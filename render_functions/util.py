import torch
import torch.nn.functional as F
import functools

from pytorch3d.renderer import RayBundle
from typing import List


def get_densities(
        raw_densities: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
        densities_activation: str = 'softplus',
        beta: float = 1.0
) -> torch.Tensor:
    """
    `raw_densities` are re-weighted using the depth step sizes
    and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
    """
    depths = depth_values.detach()  # detach

    deltas = torch.cat(
        (
            depths[..., 1:] - depths[..., :-1],
            1e10 * torch.ones_like(depths[..., :1]),  # infinite depth
        ),
        dim=-1,
    )[..., None]

    # deltas is the distance between two points, thus should be a non-negative value
    assert torch.all(deltas >= 0)

    if density_noise_std > 0.0:
        raw_densities = (
                raw_densities + torch.randn_like(raw_densities) * density_noise_std
        )

    if densities_activation == 'relu':
        act = torch.relu
    elif densities_activation == 'softplus':
        act = functools.partial(F.softplus, beta=beta)
    elif densities_activation == 'none':
        act = lambda x: x
    else:
        raise ValueError(f'Unsupported densities_activation: {densities_activation}.')

    densities = 1 - (-deltas * act(raw_densities)).exp()

    return densities


def cat_ray_bundles(ray_bundles: List[RayBundle]):
    """
    Cat ray bundles along the first dimension
    :param ray_bundles:
    :return: (n * m, ...)
    """
    return RayBundle(
        origins=torch.cat([bundle.origins for bundle in ray_bundles], dim=0),  # (m * n, i, 3)
        directions=torch.cat([bundle.directions for bundle in ray_bundles], dim=0),  # (m * n, i, 3)
        lengths=torch.cat([bundle.lengths for bundle in ray_bundles], dim=0),  # (m * n, i, p)
        xys=torch.cat([bundle.xys for bundle in ray_bundles], dim=0),  # (m * n, i, 2)
    )
