import torch

from nerf.harmonic_embedding import HarmonicEmbedding
from modules.component import ResMlp, _init_linear, ResidualBlockFC
from nerf.linear_with_repeat import LinearWithRepeat
from .util import get_densities
from typing import Tuple
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points


class MlpArchBase(torch.nn.Module):
    """
    Base class for Mlps
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons_xyz: int,
            n_hidden_neurons_dir: int,
            n_blocks_xyz: int,
            n_blocks_dir: int,
            shape_code_dim: int,
            color_code_dim: int
    ):
        super(MlpArchBase, self).__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        # Density layer
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz + shape_code_dim,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz
        )

        # Intermediate linear of density
        self.intermediate_linear = torch.nn.Linear(
            n_hidden_neurons_xyz, n_hidden_neurons_xyz
        )
        _init_linear(self.intermediate_linear)

        # Color layer
        color_layer = [
            LinearWithRepeat(
                n_hidden_neurons_xyz + embedding_dim_dir + color_code_dim, n_hidden_neurons_dir
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


class CodeMlp(MlpArchBase):
    """
    Mlp with shape and color latent code
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons_xyz: int,
            n_hidden_neurons_dir: int,
            n_blocks_xyz: int,
            n_blocks_dir: int,
            shape_code_dim: int,
            color_code_dim: int
    ):
        """
        Initialize a CodeMlp instance
        :param n_harmonic_functions_xyz:
        :param n_harmonic_functions_dir:
        :param n_hidden_neurons_xyz:
        :param n_hidden_neurons_dir:
        :param n_blocks_xyz:
        :param n_blocks_dir:
        :param shape_code_dim:
        :param color_code_dim:
        """
        super(CodeMlp, self).__init__(
            n_harmonic_functions_xyz,
            n_harmonic_functions_dir,
            n_hidden_neurons_xyz,
            n_hidden_neurons_dir,
            n_blocks_xyz,
            n_blocks_dir,
            shape_code_dim,
            color_code_dim
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        _init_linear(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

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
        densities = get_densities(raw_densities, depth_values, density_noise_std, densities_activation='softplus')
        return densities

    def _get_densities_and_colors(
            self, features: torch.Tensor, lengths: torch.Tensor, directions: torch.Tensor,
            color_latent_code: torch.Tensor, density_noise_std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The second part of the forward calculation.

        Args:
            features: the output of the common mlp (the prior part of the
                calculation), shape
                (minibatch x ... x self.n_hidden_neurons_xyz).
            density_noise_std:  As for forward().

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # Same calculation as above, just serial.
        rays_densities = self._get_densities(
            features, lengths, density_noise_std
        )
        # add color latent code
        rays_colors = self._get_colors(features, directions, color_latent_code)

        return rays_densities, rays_colors

    def forward(
            self,
            ray_bundle: RayBundle,
            shape_latent_code: torch.Tensor,
            color_latent_code: torch.Tensor,
            density_noise_std: float = 0.0,
    ):
        """
        Compute ray densities and colors
        :param ray_bundle:
        :param shape_latent_code: (n, ds)
        :param color_latent_code: (n, dc)
        :param density_noise_std:
        :return:
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`. (n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # ray direction, (n, i, 3)
        rays_directions_world = ray_bundle.directions

        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # input
        n, i, p, _ = embeds_xyz.shape
        mlp_input = torch.cat([embeds_xyz, shape_latent_code[:, None, None, :].expand(n, i, p, -1)], dim=-1)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(mlp_input)

        # output
        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle.lengths, rays_directions_world, color_latent_code, density_noise_std
        )

        return rays_densities, rays_colors
