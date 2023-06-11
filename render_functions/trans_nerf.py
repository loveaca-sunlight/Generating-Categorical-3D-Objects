# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.pixelnerf import ResnetFC
from modules.transnerf import TransformModule
from nerf.harmonic_embedding import HarmonicEmbedding
from .util import get_densities
from .registry import IMPLICIT_FUNCTIONS


@IMPLICIT_FUNCTIONS.register_module(name='transnerf_resnet')
class TransNerfResNet(torch.nn.Module):
    def __init__(
            self,
            n_harmonic_functions_xyz: int = 6,
            n_harmonic_functions_dir: int = None,
            n_hidden_neurons: int = 256,
            n_blocks: int = 5,
            with_translation: bool = False,
            latent_code_dim: int = 512,
            parallel: bool = False,
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
            with_translation: whether to contain translation in transform
            parallel: whether to compute densities and color in parallel
        """
        super().__init__()

        self.parallel = parallel

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        if n_harmonic_functions_dir is not None:
            self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
            embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3
        else:
            self.harmonic_embedding_dir = None
            embedding_dim_dir = 3

        # self.transform = TransformModule(with_translation)

        self.resnet = ResnetFC(
            d_in=embedding_dim_xyz + embedding_dim_dir,
            d_out=4,
            n_blocks=n_blocks,
            d_latent=latent_code_dim,
            d_hidden=n_hidden_neurons,
        )

    def _get_direction_embedding(self, rays_directions):
        # Normalize the ray_directions to unit l2 norm.
        rays_embedding = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        if self.harmonic_embedding_dir is not None:
            rays_embedding = self.harmonic_embedding_dir(rays_embedding)

        return rays_embedding

    def _get_rgb_and_sigma(self, inputs: torch.Tensor):
        resnet = self.resnet  # torch.nn.DataParallel(self.resnet) if self.parallel else self.resnet

        mlp_output = resnet(inputs)
        rgb = mlp_output[..., : 3]
        sigma = mlp_output[..., 3: 4]

        return rgb, sigma

    def forward(
            self,
            ray_bundle: RayBundle,

            transform_code: torch.Tensor,
            source_latent_code: torch.Tensor,
            density_noise_std: float = 0.0,
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
            transform_code: a latent code that stores the transform parameters, (1, 3 or 6).
            source_latent_code: a latent code that describes the information of source view, (1, d).
            density_noise_std: float = 0.0

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # # transform the ray bundle
        # ray_bundle = self.transform(ray_bundle, transform_code)

        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`. (n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # directions, (n, i, 3)
        embeds_dir = self._get_direction_embedding(ray_bundle.directions)

        # input
        x = torch.cat([embeds_xyz, embeds_dir[:, :, None, :].expand(-1, -1, embeds_xyz.shape[2], -1)],
                      dim=-1)  # (n, i, p, 3)
        inputs = torch.cat([source_latent_code[:, None, None, :].expand(*x.shape[:3], -1), x], dim=-1)  # latent + x

        # output
        rgb, sigma = self._get_rgb_and_sigma(inputs)

        rays_densities = get_densities(sigma, ray_bundle.lengths, density_noise_std, densities_activation='softplus')
        rays_colors = torch.sigmoid(rgb)

        return rays_densities, rays_colors
