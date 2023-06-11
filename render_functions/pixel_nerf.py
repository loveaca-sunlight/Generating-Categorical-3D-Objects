# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
from pytorch3d.renderer.cameras import get_world_to_view_transform, PerspectiveCameras, _get_sfm_calibration_matrix
from pytorch3d.transforms import Transform3d

from modules.pixelnerf import PixelEncoder, ResnetFC, repeat_first_dim
from nerf.harmonic_embedding import HarmonicEmbedding
from .util import get_densities
from .registry import IMPLICIT_FUNCTIONS


@IMPLICIT_FUNCTIONS.register_module(name='pixel_resnet')
class PixelNerfResNet(torch.nn.Module):
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons: int,
            n_blocks: int,
            latent_code_dim: int,
            combine_layer: int = 3,
            combine_type: str = 'average',
            density_activation: str = 'relu',
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
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        if n_harmonic_functions_dir is not None:
            self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
            embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3
        else:
            print('Position embedding is not used for direction.')
            self.harmonic_embedding_dir = None
            embedding_dim_dir = 3

        self.encoder_net = None
        self._latent_code_dim = latent_code_dim

        self.resnet = ResnetFC(
            d_in=embedding_dim_xyz + embedding_dim_dir,
            d_out=4,
            n_blocks=n_blocks,
            d_latent=latent_code_dim,
            d_hidden=n_hidden_neurons,
            combine_layer=combine_layer,
            combine_type=combine_type
        )

        assert density_activation in ['softplus', 'relu']
        self._density_activation = density_activation

    def set_encoder_net(self, net: PixelEncoder):
        assert isinstance(net, PixelEncoder)
        assert net.latent_dim == self._latent_code_dim
        assert self.encoder_net is None, 'encoder_net has been set.'

        self.encoder_net = net

    def _get_direction_embedding(self, rays_directions):
        # Normalize the ray_directions to unit l2 norm.
        rays_embedding = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        if self.harmonic_embedding_dir is not None:
            rays_embedding = self.harmonic_embedding_dir(rays_embedding)

        return rays_embedding

    def forward(
            self,
            ray_bundle: RayBundle,
            n_sources: int,
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
            n_sources: number of source views
            density_noise_std:
        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`. (b, i, p, 3)
        # b - batch size of all views, n - number of source views
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        batch_size = rays_points_world.shape[0]
        rays_points_world = repeat_first_dim(rays_points_world, dim=1, n_repeats=n_sources)  # (b * n, i, p, 3)

        # ray direction, (b, i, 3)
        rays_directions_world = ray_bundle.directions
        rays_directions_world = repeat_first_dim(rays_directions_world, dim=1, n_repeats=n_sources)  # (b * n, i, 3)

        # # fetch latent code, (b * n, i, p, d)
        # latent_code = self.encoder_net(rays_points_world)

        # project points to coordinates of source view
        source_camera: PerspectiveCameras = self.encoder_net.source_camera
        b_n, i, p, _ = rays_points_world.shape
        # reshape camera parameters from (n, 4, 4) to (b * n, 4, 4)
        cam_R = repeat_first_dim(source_camera.R, dim=0, n_repeats=batch_size)
        cam_T = repeat_first_dim(source_camera.T, dim=0, n_repeats=batch_size)
        cam_F = repeat_first_dim(source_camera.focal_length, dim=0, n_repeats=batch_size)
        cam_P = repeat_first_dim(source_camera.principal_point, dim=0, n_repeats=batch_size)

        # world to view transform
        R: Transform3d = get_world_to_view_transform(R=cam_R, T=rays_points_world.new_zeros((b_n, 3)))
        RT: Transform3d = get_world_to_view_transform(R=cam_R, T=cam_T)

        # projection transform (view to ndc)
        K = _get_sfm_calibration_matrix(
            b_n,
            source_camera.device,
            cam_F,
            cam_P,
            orthographic=False,
        )
        projection = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=source_camera.device
        )

        # full transform (world to ndc)
        full_transform = RT.compose(projection)

        # transform points from world coordinates to ndc coordinates
        # note that this operation changes parameters in source cameras
        uv = full_transform.transform_points(rays_points_world.view(b_n, i * p, 3), eps=1.0e-8).view(b_n, i, p, 3)
        # to coordinates of grid_sample
        uv = -1.0 * uv[..., : 2]
        # fetch latent code
        latent_code = self.encoder_net(uv)

        rays_points_view = RT.transform_points(rays_points_world.view(b_n, i * p, 3), eps=1.0e-8).view(b_n, i, p, 3)
        rays_directions_view = R.transform_points(rays_directions_world, eps=1.0e-8)  # (b * n, i, 3)

        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_view)

        # directions
        # embeds_dir = self._get_direction_embedding(rays_directions_world)
        embeds_dir = self._get_direction_embedding(rays_directions_view)

        # input
        xyz_dir = torch.cat([embeds_xyz, embeds_dir[:, :, None, :].expand(-1, -1, p, -1)], dim=-1)  # (b * n, i, p, 6)
        inputs = torch.cat(
            [
                latent_code,
                xyz_dir
            ],
            dim=-1
        )

        # output
        features = self.resnet(
            inputs,
            combine_inner_dims=(n_sources,)
        )
        rgb, sigma = features[..., : 3], features[..., 3:]

        rays_densities = get_densities(sigma, ray_bundle.lengths, density_noise_std,
                                       densities_activation=self._density_activation)
        rays_colors = torch.sigmoid(rgb)

        return rays_densities, rays_colors
