import warnings

import torch
import torch.nn as nn
import torch.nn.functional

from pytorch3d.renderer import RayBundle
from .utils import _get_scale_transform, _get_translation_transform, _get_rotation_transform


class TransformModuleBase(nn.Module):
    """
    Base module for transform
    """
    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor, **kwargs) -> RayBundle:
        raise NotImplementedError

    def forward(self, ray_bundle: RayBundle, transform_code: torch.Tensor, **kwargs) -> RayBundle:
        """
        Transform input ray bundle
        :param ray_bundle:
        :param transform_code: (n, d)
        :return:
        """
        # return transformed ray bundle
        return self._transform(ray_bundle, transform_code, **kwargs)


class TransformModule(TransformModuleBase):
    """
    A module can transform point to canonical coordinates
    """
    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor, **kwargs):
        """
        Transform a ray bundle from world coordinates to template coordinates using 6-DOF parameters
        :param ray_bundle:
        :param transform_code: (n, 6)
        :return:
        """
        n, _6 = transform_code.shape
        assert _6 == 6, f'Except transform code with 6 elements, but given {_6}.'

        # chunk, (n, 3)
        r_xyz, t_xyz = torch.chunk(transform_code, 2, dim=1)

        # rotation, using radians to avoid large number when using degrees
        R = _get_rotation_transform(r_xyz)

        # translation
        T = _get_translation_transform(t_xyz)

        # full transform
        RT = R.compose(T)

        # transform
        origins = RT.transform_points(ray_bundle.origins)  # (n, i, 3)
        directions = R.transform_points(ray_bundle.directions)  # (n, i, 3)

        return RayBundle(origins, directions, ray_bundle.lengths, ray_bundle.xys)


class FullTransformModule(TransformModuleBase):
    """
    Transform a ray bundle from world coordinates to template coordinates using 6-DOF parameters, as well as a isotropic
    scale coefficient
    """
    def __init__(self):
        super(FullTransformModule, self).__init__()
        self.softplus = nn.Softplus(beta=10)

    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor, **kwargs):
        """
        Transform ray bundle according to transform code
        :param ray_bundle:
        :param transform_code: (n, 7) - [r(x,y,z), t(x, y, z), s]
        :return:
        """
        n, _7 = transform_code.shape
        assert _7 == 7, f'Excepted inputs with 7 dims, but given {_7}.'

        # split from code
        r_xyz = transform_code[:, : 3]   # (n, 3)
        t_xyz = transform_code[:, 3: 6]  # (n, 3)
        scale = transform_code[:, 6]    # (n,)
        # use softplus to prevent negative scale value
        scale = self.softplus(scale)

        # transforms R, T and S
        R = _get_rotation_transform(r_xyz)
        T = _get_translation_transform(t_xyz)
        S = _get_scale_transform(scale)

        # apply full transforms to origins
        RST = R.compose(S).compose(T)
        origins = RST.transform_points(ray_bundle.origins)  # (n, i, 3)
        # apply R to directions
        directions = R.transform_points(ray_bundle.directions)  # (n, i, 3)
        # compute updated lengths
        lengths = ray_bundle.lengths * scale.view(n, 1, 1)  # (n, i, p)

        # transformed ray bundle
        ray_bundle_new = RayBundle(
            origins=origins,
            directions=directions,
            lengths=lengths,
            xys=ray_bundle.xys
        )

        # transform key points if offered
        key_points = kwargs.get('key_points', None)
        if key_points is None:
            return ray_bundle_new
        else:
            n, p, _3 = key_points.shape
            assert _3 == 3
            key_points_new = RST.transform_points(key_points)
            return ray_bundle_new, key_points_new
