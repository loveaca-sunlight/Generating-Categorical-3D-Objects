# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle

from .mlp_arch import CodeMlp
from .registry import IMPLICIT_FUNCTIONS


@IMPLICIT_FUNCTIONS.register_module(name='sepnerf_mlp')
class SepNerfMlp(torch.nn.Module):
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons_xyz: int,
            n_hidden_neurons_dir: int,
            n_blocks_xyz: int,
            n_blocks_dir: int,
            shape_code_dim: int,
            color_code_dim: int,
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

        self.mlp = CodeMlp(
            n_harmonic_functions_xyz,
            n_harmonic_functions_dir,
            n_hidden_neurons_xyz,
            n_hidden_neurons_dir,
            n_blocks_xyz,
            n_blocks_dir,
            shape_code_dim,
            color_code_dim,
        )

        self._shape_code_dim = shape_code_dim
        self._color_code_dim = color_code_dim

    def forward(
            self,
            ray_bundle: RayBundle,

            shape_latent_code: torch.Tensor,
            color_latent_code: torch.Tensor,

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
            shape_latent_code: latent code for representing shape, (1, ds)
            color_latent_code: latent code for representing color, (1, dc)
            density_noise_std: float = 0.0

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # split shape and color latent code
        assert shape_latent_code.shape[-1] == self._shape_code_dim
        assert color_latent_code.shape[-1] == self._color_code_dim

        # outputs
        rays_densities, rays_colors = self.mlp(
            ray_bundle,
            shape_latent_code,
            color_latent_code,
            density_noise_std
        )

        return rays_densities, rays_colors
