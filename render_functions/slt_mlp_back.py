from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp
from modules.transform_new import TransformModule
from modules.util import freeze
from nerf.harmonic_embedding import HarmonicEmbedding
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities, cat_ray_bundles


@IMPLICIT_FUNCTIONS.register_module(name='slt_mlp_b')
class SilhouetteBackMlp(torch.nn.Module):
    """
    Mlp that stores the density information of a template
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_hidden_neurons_xyz: int,
            n_blocks_xyz: int,
            n_transforms: int,
            density_activation: str,
            norm: bool
    ):
        super(SilhouetteBackMlp, self).__init__()

        # Transform Module
        self._n_transforms = n_transforms
        self.transformer = TransformModule()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        # Density layer
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz,
            norm=norm,
            act='elu'
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)

        assert density_activation in ['softplus', 'relu']
        self._density_activation = density_activation

        self._transformed_key_points = None

    @property
    def transformed_key_points(self):
        return self._transformed_key_points

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

        # # check density
        # if (densities.max() > 1.0) or (densities.min() < 0.0):
        #     print(f'Min: {densities.min()}, Max: {densities.max()}.')

        return densities

    def freeze_template_layers(self):
        freeze(self.mlp_xyz)
        freeze(self.density_layer)

    def forward(
            self,
            ray_bundle: RayBundle,
            transform_codes: torch.Tensor,
            key_points: torch.Tensor,
            density_noise_std: float = 0.0,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param transform_codes:
        :param key_points: (1, p, 3)
        :param density_noise_std:
        :param kwargs:
        :return:
        """
        # transform ray bundle with multiple transform parameters
        if transform_codes is not None:
            n_transforms = self._n_transforms if self.training else 1
            assert transform_codes.shape[0] == n_transforms, \
                f'Excepted number of transforms is {n_transforms}, but given {transform_codes.shape[0]}.'

            n_multiplex = transform_codes.shape[0]
            transforms = torch.chunk(transform_codes, n_multiplex, dim=0)  # [(1, d), ... x m]
            # transform ray bundle and key points
            transformed = [self.transformer(ray_bundle, transform, key_points=key_points)
                           for transform in transforms]  # [(bundle, (points)), ... x m]
            # if key points is not None
            if key_points is not None:
                # transformed ray bundles
                ray_bundle = cat_ray_bundles([t[0] for t in transformed])
                # transformed key points
                self._transformed_key_points = torch.cat(
                    [t[1] for t in transformed],
                    dim=0
                )  # (m, p, 3)
            else:
                # transformed ray bundles
                ray_bundle = cat_ray_bundles(transformed)

        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        # (n, i, p, 1)
        rays_densities = self._get_densities(features, ray_bundle.lengths, density_noise_std)

        # set colors to zero
        rays_colors = rays_densities.new_zeros(size=(*rays_densities.shape[: 3], 3))

        return rays_densities, rays_colors
