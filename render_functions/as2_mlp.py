import typing

import torch
import torch.nn.functional as F
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp, ResBlocks, LinearBlock
from modules.defnerf import HyperDeformationField
from modules.hyper_modules import HyperNetwork, HyperLinear, HyperBlocks
from modules.transform_new import TransformModule
from modules.util import freeze
from nerf.harmonic_embedding import HarmonicEmbedding
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities


def _init_weight(linear):
    torch.nn.init.kaiming_uniform_(linear.weight.data, mode='fan_in', nonlinearity='linear')


@IMPLICIT_FUNCTIONS.register_module(name='as2_mlp')
class AS2Mlp(HyperNetwork):
    """
    An adapted version that ignores view direction
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            dim_hidden_density: int,
            dim_hidden_diffuse: int,
            dim_hidden_deformer: int,
            dim_middle_diffuse: int,
            n_blocks_density: int,
            n_blocks_diffuse: int,
            norm: bool,
            shape_code_dim: int,
            color_code_dim: int,
            hyper_dim_hidden_diffuse: int,
            hyper_norm: bool,
            **kwargs
    ):
        """
        Initialize
        """
        super(AS2Mlp, self).__init__()

        # Rigid transform module
        self.transformer = TransformModule()

        # Deformable module shared by all functions
        self.deformer = None
        self._dim_hidden_deformer = dim_hidden_deformer  # feature dims of deformation output

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        #
        # Density layers for storing template shape, don't modify the module name
        #
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=dim_hidden_density,
            dim_out=dim_hidden_density,
            n_blocks=n_blocks_density,
            norm=True,  # the template uses norm,
            act='elu'  # the template uses elu
        )

        self.density_layer = torch.nn.Linear(dim_hidden_density, 1)
        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0

        #
        # Color layers
        #
        self.mlp_diffuse = HyperBlocks(
            dim_in=(dim_hidden_density + dim_hidden_deformer + embedding_dim_dir),
            dim_hidden=dim_hidden_diffuse,
            dim_middle=dim_middle_diffuse,
            n_blocks=n_blocks_diffuse,
            norm=norm,
            hyper_dim_in=color_code_dim,
            hyper_dim_hidden=hyper_dim_hidden_diffuse,
            hyper_norm=hyper_norm
        )

        self.diffuse_layer = HyperLinear(
            dim_in=dim_hidden_diffuse,
            dim_out=3,
            dim_middle=3,
            hyper_dim_in=color_code_dim,
            hyper_dim_hidden=hyper_dim_hidden_diffuse,
            hyper_norm=hyper_norm
        )

        self._shape_code_dim = shape_code_dim
        self._color_code_dim = color_code_dim

        self._density_activation = 'relu'  # fix

        self._latest_points_deformation = None
        self._latest_points_position = None
        self._latest_points_specular = None

        print(f'{type(self).__name__}: norm: {norm}, hyper_norm: {hyper_norm}.')

    def set_deformable_field(self, module: HyperDeformationField):
        assert isinstance(module, HyperDeformationField)
        assert self._dim_hidden_deformer == module.feature_dim
        assert self.deformer is None, 'Deformable field has been set.'

        self.deformer = module

    def freeze_template_layers(self):
        freeze(self.mlp_xyz)
        freeze(self.density_layer)

    @property
    def latest_points_deformation(self):
        """
        Return latest points deformation
        :return:
        """
        return self._latest_points_deformation

    @property
    def latest_points_position(self):
        """
        Return latest points position
        :return:
        """
        return self._latest_points_position

    @property
    def latest_points_specular(self):
        """
        Return latest points specular
        :return:
        """
        return self._latest_points_specular

    def _get_densities(
            self,
            features: torch.Tensor,
            depth_values: torch.Tensor,
            density_noise_std: float,
            return_raw_density: bool = False
    ) -> torch.Tensor:
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)

        if return_raw_density:
            return raw_densities

        densities = get_densities(raw_densities, depth_values, density_noise_std,
                                  densities_activation=self._density_activation)
        return densities

    def _get_colors(
            self, features: torch.Tensor, directions: torch.Tensor
    ):
        """
        This function predicts the diffuse color of each points in 3d space
        """
        # Normalize the ray_directions to unit l2 norm. detach
        rays_directions_normed = torch.nn.functional.normalize(directions.detach(), dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions. (n, i, dr)
        dir_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # inputs
        n, i, p, _ = features.shape
        inputs = torch.cat(
            [
                features,
                dir_embedding[:, :, None, :].expand(-1, -1, p, -1)
            ], dim=-1
        )

        diffuse_features = self.mlp_diffuse(
            inputs
        )
        raw_diffuse = self.diffuse_layer(diffuse_features)

        color = torch.sigmoid(raw_diffuse)

        return color

    def produce_parameters(self, latent_code: torch.Tensor) -> typing.List[torch.Tensor]:
        parameters = []
        for module in (self.mlp_diffuse, self.diffuse_layer):
            parameters.extend(module.produce_parameters(latent_code))
        return parameters

    def hyper_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        for module in (self.mlp_diffuse, self.diffuse_layer):
            yield from module.parameters()

    def compute_density(
            self,
            points: torch.Tensor,
            enable_deformation: bool = True,
    ):
        """
        Compute density of given points for matching cubes
        :param points: (p, 3)
        :param enable_deformation:
        :return:
        """
        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(points)
            # use translation for points deformation
            rays_points_template = points + points_deformation
        else:
            rays_points_template = points

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_template)

        # self.mlp maps each harmonic embedding to a latent feature space.
        xyz_features = self.mlp_xyz(embeds_xyz)

        # get densities
        rays_densities = self._get_densities(xyz_features, None, None, return_raw_density=True)
        rays_densities = torch.relu(rays_densities)

        return rays_densities

    def forward(
            self,
            ray_bundle: RayBundle,
            transform_code: torch.Tensor,
            enable_deformation: bool = True,
            density_noise_std: float = 0.0,
            return_raw_density: bool = False,
            **kwargs,
    ):
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param transform_code:
        :param enable_deformation:
        :param density_noise_std:
        :param return_raw_density:
        :param kwargs:
        :return:
        """
        # rigid transform
        if transform_code is not None:
            ray_bundle = self.transformer(ray_bundle, transform_code)
        else:
            assert not self.training

        # Convert the ray parametrizations to world coordinates with `ray_bundle_to_ray_points`. (n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # Store latest points position
        self._latest_points_position = rays_points_world

        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(rays_points_world)
            # use translation for points deformation
            rays_points_template = rays_points_world + points_deformation

            # Store latest points deformation
            self._latest_points_deformation = points_deformation
        else:
            assert not self.training, 'Deformation can not be disabled when training.'
            rays_points_template = rays_points_world
            deformer_features = None

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_template)

        # self.mlp maps each harmonic embedding to a latent feature space.
        xyz_features = self.mlp_xyz(embeds_xyz)

        # rays densities from template
        assert not (self.training and return_raw_density)
        rays_densities = self._get_densities(xyz_features, ray_bundle.lengths, density_noise_std,
                                             return_raw_density=return_raw_density)

        # get rays colors when deformation is enabled
        if enable_deformation:
            # concatenate the features from mlp_xyz and deformer as input
            cat_features = torch.cat([deformer_features, xyz_features], dim=-1)  # (n, i, p, dx + dd)

            # get color
            rays_colors = self._get_colors(cat_features, ray_bundle.directions)  # (n, i, p, 3)

            self._latest_points_specular = 0
        else:
            rays_colors = rays_points_template.new_zeros((*rays_points_template.shape[: 3], 3))  # (n, i, p, 3)

        # cat with depths
        rays_features = torch.cat(
            [
                rays_colors,
                ray_bundle.lengths.detach()[..., None]
            ], dim=-1
        )

        return rays_densities, rays_features
