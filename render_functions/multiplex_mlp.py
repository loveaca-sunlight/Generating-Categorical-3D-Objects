from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
from modules.transnerf import TransformModule
from modules.component import ResMlp, _init_linear, ResidualBlockFC
from modules.defnerf import DeformableField
from modules.util import freeze
from nerf.harmonic_embedding import HarmonicEmbedding
from nerf.linear_with_repeat import LinearWithRepeat
from .util import get_densities, cat_ray_bundles
from .registry import IMPLICIT_FUNCTIONS


@IMPLICIT_FUNCTIONS.register_module(name='multiplex_mlp')
class MultiplexMlp(torch.nn.Module):
    """
    Mlp that stores the density information of a template
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons_xyz: int,
            n_hidden_neurons_dir: int,
            n_blocks_xyz: int,
            n_blocks_dir: int,
            feature_dim: int,
            shape_code_dim: int,
            color_code_dim: int,
            density_activation: str
    ):
        super(MultiplexMlp, self).__init__()

        # Rigid transform module
        self.transformer = TransformModule(with_translation=True)

        # Deformable module shared by all functions
        self.deformer = None
        self._feature_dim = feature_dim

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        # Density layer
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        _init_linear(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        # Intermediate linear of density
        self.intermediate_linear = torch.nn.Linear(
            feature_dim, feature_dim
        )
        _init_linear(self.intermediate_linear)

        # Color layer
        color_layer = [
            LinearWithRepeat(
                feature_dim + embedding_dim_dir + color_code_dim, n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True)
        ]
        if n_blocks_dir > 0:
            color_layer.extend([
                ResidualBlockFC(n_hidden_neurons_dir, n_hidden_neurons_dir, n_hidden_neurons_dir)
                for _ in range(n_blocks_dir)
            ])
        color_layer.extend([
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        ])
        self.color_layer = torch.nn.Sequential(*color_layer)

        self._shape_code_dim = shape_code_dim
        self._color_code_dim = color_code_dim

        assert density_activation in ['softplus', 'relu']
        self._density_activation = density_activation

        self._latest_points_deformation = None
        self._latest_points_position = None

    def set_deformable_field(self, module: DeformableField):
        assert isinstance(module, DeformableField)
        assert self._feature_dim == module.feature_dim
        assert self.deformer is None, 'Deformable field has been set.'

        self.deformer = module

    def freeze_template_layers(self):
        freeze(self.mlp_xyz)
        freeze(self.density_layer)

    @property
    def latest_points_deformation(self):
        return self._latest_points_deformation

    @property
    def latest_points_position(self):
        return self._latest_points_position

    def _get_densities(
            self,
            features: torch.Tensor,
            depth_values: torch.Tensor,
            density_noise_std: float,
    ) -> torch.Tensor:
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        densities = get_densities(raw_densities, depth_values, density_noise_std,
                                  densities_activation=self._density_activation)
        return densities

    def _get_colors(
            self, features: torch.Tensor, rays_directions: torch.Tensor, color_latent_code: torch.Tensor,
    ) -> torch.Tensor:
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions. (n, i, d)
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # Cat with color latent code (1, d')
        n, i, _ = rays_embedding.shape
        color_input = torch.cat([rays_embedding, color_latent_code.unsqueeze(1).expand(n, i, -1)], dim=-1)

        return self.color_layer((self.intermediate_linear(features), color_input))

    def forward(
            self,
            ray_bundle: RayBundle,

            shape_latent_code: torch.Tensor,
            color_latent_code: torch.Tensor,
            transform_codes: torch.Tensor,
            enable_deformation: bool = True,

            density_noise_std: float = 0.0,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param shape_latent_code: (1, ds)
        :param color_latent_code: (1, dc)
        :param transform_codes: (m, 6)
        :param enable_deformation:
        :param density_noise_std:
        :param kwargs:
        :return:
        """
        # split shape and color latent code
        if shape_latent_code is not None:
            assert shape_latent_code.shape[-1] == self._shape_code_dim
        if color_latent_code is not None:
            assert color_latent_code.shape[-1] == self._color_code_dim

        # transform ray bundle with multiple transform parameters
        if transform_codes is not None:
            n_multiplex = transform_codes.shape[0]
            transforms = torch.chunk(transform_codes, n_multiplex, dim=0)  # [(1, d), ... x m]
            ray_bundles = [self.transformer(ray_bundle, transform) for transform in transforms]  # [bundle, ... x m]
            ray_bundle = cat_ray_bundles(ray_bundles)

        # Convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`. (m * n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # Store latest points position
        self._latest_points_position = rays_points_world

        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(rays_points_world, shape_latent_code)
            rays_points_template = rays_points_world + points_deformation

            # Store latest points deformation
            self._latest_points_deformation = points_deformation

            # ray direction, (m * n, i, 3)
            rays_directions_world = ray_bundle.directions

            # rays colors from features of deformable field and latent code
            rays_colors = self._get_colors(deformer_features, rays_directions_world, color_latent_code)
        else:
            assert not self.training, 'Deformation can not be disabled when training.'
            rays_points_template = rays_points_world
            rays_colors = rays_points_template.new_zeros((*rays_points_template.shape[: 3], 3))

        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_template)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz)

        # rays densities from template
        rays_densities = self._get_densities(features, ray_bundle.lengths, density_noise_std)

        # (m * n, i, p, 1), (m * n, i, p, 3)
        return rays_densities, rays_colors
