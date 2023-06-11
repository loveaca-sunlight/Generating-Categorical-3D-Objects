from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp
from nerf.harmonic_embedding import HarmonicEmbedding
from nerf.linear_with_repeat import LinearWithRepeat
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities


@IMPLICIT_FUNCTIONS.register_module(name='nerf_mlp')
class NerfMlp(torch.nn.Module):
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
            densities_only: bool,
            density_activation: str,
            norm: bool = False #此处的设置由cfg修改
    ):
        super(NerfMlp, self).__init__()

        self._densities_only = densities_only

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3 #63

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3 #39

        # Density layer #前三层坐标差值之前的f值 输出feature 输入坐标
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz,
            norm=norm
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        torch.nn.init.kaiming_uniform_(self.density_layer.weight.data, nonlinearity='linear')

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        # Color layer
        if not self._densities_only:
            # Intermediate linear of density
            self.intermediate_linear = torch.nn.Linear(
                n_hidden_neurons_xyz, n_hidden_neurons_xyz
            )
            torch.nn.init.kaiming_uniform_(self.intermediate_linear.weight.data, nonlinearity='linear')

            color_layer = [
                LinearWithRepeat(
                    n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
                ),
                torch.nn.LayerNorm(n_hidden_neurons_dir) if norm else torch.nn.Identity(),
                torch.nn.ELU(inplace=True)
                # torch.nn.ReLU(True)
            ]

            assert n_blocks_dir == 0
            # if n_blocks_dir > 0:
            #     color_layer.extend([
            #         ResidualBlockFC(n_hidden_neurons_dir, n_hidden_neurons_dir, n_hidden_neurons_dir)
            #         for _ in range(n_blocks_dir)
            #     ])

            color_layer.extend([
                torch.nn.Linear(n_hidden_neurons_dir, 3),
                torch.nn.Sigmoid(),
            ])
            self.color_layer = torch.nn.Sequential(*color_layer)

        assert density_activation in ['softplus', 'relu']
        self._density_activation = density_activation

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
            self, features: torch.Tensor, rays_directions: torch.Tensor
    ) -> torch.Tensor:
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions. (n, i, 3)
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer((self.intermediate_linear(features), rays_embedding))

    def forward(
            self,
            ray_bundle: RayBundle,
            density_noise_std: float = 0.0,

            return_raw_density: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param density_noise_std:
        :param return_raw_density:
        :param kwargs:
        :return:
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world) #16*600*64*63
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz) 
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        # (n, i, p, 1)
        assert not (self.training and return_raw_density)
        rays_densities = self._get_densities(features, ray_bundle.lengths, density_noise_std,
                                             return_raw_density=return_raw_density)

        if not self._densities_only:
            rays_colors = self._get_colors(features, ray_bundle.directions)
        else:
            rays_colors = rays_densities.new_zeros(size=(*rays_densities.shape[: 3], 3))

        return rays_densities, rays_colors
