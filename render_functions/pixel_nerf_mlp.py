# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points, PerspectiveCameras
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import get_world_to_view_transform

from nerf.implicit_function import MLPWithInputSkips, _xavier_init
from nerf.harmonic_embedding import HarmonicEmbedding
from nerf.linear_with_repeat import LinearWithRepeat
from modules.pixelnerf import PixelEncoder
from .util import get_densities


class PixelNerfMlp(torch.nn.Module):
    def __init__(
            self,
            n_harmonic_functions_xyz: int = 10,
            n_harmonic_functions_dir: int = 4,
            n_hidden_neurons_xyz: int = 256,
            n_hidden_neurons_dir: int = 128,
            n_layers_xyz: int = 8,
            append_xyz: Tuple[int] = (5,),
            encoder_net: PixelEncoder = None,
            **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons: dimensions of hidden neurons.
            n_blocks: number of residual blocks.
            encoder_net: pixel encoder attached to this renderer
            parallel: whether to compute densities and color in parallel
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.encoder_net = encoder_net
        assert self.encoder_net is not None, f'encoder_net is None.'

        self.mlp_xyz = MLPWithInputSkips(
            n_layers=n_layers_xyz,
            input_dim=embedding_dim_xyz + self.encoder_net.latent_dim,
            output_dim=n_hidden_neurons_xyz,
            skip_dim=embedding_dim_xyz,
            hidden_dim=n_hidden_neurons_xyz,
            input_skips=append_xyz
        )

        self.intermediate_linear = torch.nn.Linear(
            n_hidden_neurons_xyz, n_hidden_neurons_xyz
        )
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )

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
        densities = get_densities(raw_densities, depth_values, density_noise_std)
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

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer((self.intermediate_linear(features), rays_embedding))

    def _get_densities_and_colors(
        self, features: torch.Tensor, lengths: torch.Tensor, directions: torch.Tensor, density_noise_std: float
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
        rays_colors = self._get_colors(features, directions)

        return rays_densities, rays_colors

    def forward(
            self,
            ray_bundle: RayBundle,
            density_noise_std: float,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std:
        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`. (n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # ray direction, (n, i, 3)
        rays_directions_world = ray_bundle.directions

        # fetch latent code, encode first, (n, i, p, d)
        latent_code = self.encoder_net(rays_points_world)

        # project points to view coordinates of source view
        source_camera = self.encoder_net.source_camera
        n, i, p, _ = rays_points_world.shape
        R: Transform3d = get_world_to_view_transform(R=source_camera.R, T=rays_points_world.new_zeros((1, 3)))
        RT: Transform3d = get_world_to_view_transform(R=source_camera.R, T=source_camera.T)

        rays_points_view = RT.transform_points(rays_points_world.view(n, i * p, 3), eps=1.0e-8).view(n, i, p, 3)
        rays_directions_view = R.transform_points(rays_directions_world, eps=1.0e-8)

        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_view)

        # input
        mlp_input = torch.cat([embeds_xyz, latent_code], dim=-1)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(mlp_input, embeds_xyz)

        # output
        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle.lengths, rays_directions_view, density_noise_std
        )

        return rays_densities, rays_colors
