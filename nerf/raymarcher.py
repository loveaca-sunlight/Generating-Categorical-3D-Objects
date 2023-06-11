# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dis import dis
from mimetypes import init
import torch
from typing import Optional, List, Union
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _shifted_cumprod,
)
import torch.nn as nn
import torch.nn.functional as F

def _check_raymarcher_inputs(
    rays_densities: torch.Tensor,
    rays_features: Optional[List[torch.Tensor]],
    rays_z: Optional[torch.Tensor],
    features_can_be_none: bool = False,
    z_can_be_none: bool = False,
    density_1d: bool = True,
):
    """
    Checks the validity of the inputs to raymarching algorithms.
    An adapted version that rays_features can be either Tensor or List of Tensor
    """
    if not torch.is_tensor(rays_densities):
        raise ValueError("rays_densities has to be an instance of torch.Tensor.")

    if not z_can_be_none and not torch.is_tensor(rays_z):
        raise ValueError("rays_z has to be an instance of torch.Tensor.")

    if not features_can_be_none:
        for feature in rays_features:
            if not torch.is_tensor(feature):
                raise ValueError("each element of rays_features has to be an instance of torch.Tensor.")

    if rays_densities.ndim < 1:
        raise ValueError("rays_densities have to have at least one dimension.")

    if density_1d and rays_densities.shape[-1] != 1:
        raise ValueError(
            "The size of the last dimension of rays_densities has to be one."
        )

    rays_shape = rays_densities.shape[:-1]

    # pyre-fixme[16]: `Optional` has no attribute `shape`.
    if not z_can_be_none and rays_z.shape != rays_shape:
        raise ValueError("rays_z have to be of the same shape as rays_densities.")

    if not features_can_be_none:
        for feature in rays_features:
            if feature.shape[:-1] != rays_shape:
                raise ValueError(
                    "The first to previous to last dimensions of rays_features"
                    " have to be the same as all dimensions of rays_densities."
                )


class EmissionAbsorptionNeRFRaymarcher(EmissionAbsorptionRaymarcher):
    """
    This is essentially the `pytorch3d.renderer.EmissionAbsorptionRaymarcher`
    which additionally returns the rendering weights. It also skips returning
    the computation of the alpha-mask which is, in case of NeRF, equal to 1
    everywhere.

    The weights are later used in the NeRF pipeline to carry out the importance
    ray-sampling for the fine rendering pass.

    For more details about the EmissionAbsorptionRaymarcher please refer to
    the documentation of `pytorch3d.renderer.EmissionAbsorptionRaymarcher`.

    Adapt this implementation to support multiple rays_features
    """

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: Union[torch.Tensor, List[torch.Tensor]],
        eps: float = 1e-10,
        **kwargs,
    ):
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific emission-absorption distribution.
                Each ray distribution `(..., :)` is a valid probability
                distribution, i.e. it contains non-negative values that integrate
                to 1, such that `weights.sum(dim=-1)==1).all()` yields `True`.
        """
        if torch.is_tensor(rays_features):
            rays_features = [rays_features]

        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption

        features = [
            (weights[..., None] * feature).sum(dim=-2)
            for feature in rays_features
        ]

        if len(features) == 1:
            features = features[0]

        return features, weights

class SDFNeRFRaymarcher(EmissionAbsorptionRaymarcher):

    def forward(
        self,
        rays_sdf: torch.Tensor,
        rays_features: Union[torch.Tensor, List[torch.Tensor]],
        inv_s: torch.Tensor,
        gradient: torch.Tensor,
        pts: torch.Tensor,
        dists: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ):
        batch_size, chunk, n_samples, _ = rays_sdf.shape 
        inv_s = inv_s.expand(batch_size * chunk * n_samples, 1)
        
        ray_bundle = kwargs['ray_bundle']
        cos_anneal_ratio = 1.0
        
        if gradient != None:
            dirs = ray_bundle.directions[...,None,:].expand(pts.shape)
            pts = pts.reshape(-1,3)
            dirs = dirs.reshape(-1,3)

            gradients = gradient.squeeze()
            gradients = gradients.reshape(-1,3)
            
            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            #现在默认1.0
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = rays_sdf.reshape(-1,1) + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = rays_sdf.reshape(-1,1) - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, chunk, n_samples).clip(0.0, 1.0)

            pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, chunk, n_samples)
            inside_sphere = (pts_norm < 1.0).float().detach()
            relax_inside_sphere = (pts_norm < 1.2).float().detach()

            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, chunk, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[..., :-1]
            weights_sum = weights.sum(dim=-1, keepdim=True)

            color = (rays_features * weights[..., None]).sum(dim=-2)

            # Eikonal loss
            gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, chunk, n_samples, 3), ord=2,
                                                dim=-1) - 12.0) ** 2
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

            return color, weights, gradient_error

        else:
            if torch.is_tensor(rays_features):
                rays_features = [rays_features]

            _check_raymarcher_inputs(
                rays_sdf,
                rays_features,
                None,
                z_can_be_none=True,
                features_can_be_none=False,
                density_1d=True,
            )
            _check_density_bounds(rays_sdf)
            rays_densities = rays_sdf[..., 0]
            absorption = _shifted_cumprod(
                (1.0 + eps) - rays_densities, shift=self.surface_thickness
            )
            weights = rays_densities * absorption

            features = [
                (weights[..., None] * feature).sum(dim=-2)
                for feature in rays_features
            ]

            if len(features) == 1:
                features = features[0]

            return features, weights
