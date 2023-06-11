import warnings

import torch
import torch.nn as nn
import torch.nn.functional
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
from pytorch3d.transforms import Translate, Transform3d, euler_angles_to_matrix, Scale, Rotate, RotateAxisAngle


def _get_rotation_transform(angles: torch.Tensor):
    """
    Return a rotation transform
    :param angles: (x, y, z) in radiance
    :return:
    """
    n, _3 = angles.shape
    assert _3 == 3

    rotation_matrix = euler_angles_to_matrix(-angles, 'XYZ')  # (n, 3, 3)
    R = Rotate(
        R=rotation_matrix,
        device=angles.device
    )
    return R


def _get_translation_transform(translations: torch.Tensor):
    """
    Return a translation transform
    :param translations: (x, y, z)
    :return:
    """
    n, _3 = translations.shape
    assert _3 == 3

    return Translate(
            translations,
            device=translations.device
        )


class TransformModuleBase(nn.Module):
    """
    Base module for transform
    """
    def __init__(self):
        super(TransformModuleBase, self).__init__()
        warnings.warn('Please use the transform module in "modules/transform" packet.')

    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor) -> RayBundle:
        raise NotImplementedError

    def forward(self, ray_bundle: RayBundle, transform_code: torch.Tensor) -> RayBundle:
        """
        Transform input ray bundle
        :param ray_bundle:
        :param transform_code: (n, d)
        :return:
        """
        # return transformed ray bundle
        return self._transform(ray_bundle, transform_code)


class TransformModule(TransformModuleBase):
    """
    A module can transform point to canonical coordinates
    """
    def __init__(self, with_translation: bool = True):
        super(TransformModule, self).__init__()

        # self.with_translation = True

    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor):
        """
        Transform a ray bundle from world coordinates to canonical coordinates
        :param ray_bundle:
        :param transform_code: (n, d)
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
    Transform module that applies rotations, translations and scales to ray bundle
    """
    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor):
        """
        Transform ray bundle according to transform code
        :param ray_bundle:
        :param transform_code: (n, 9) - [r(x,y,z), t(x, y, z), s(x, y, z)]
        :return:
        """
        n, _9 = transform_code.shape
        assert _9 == 9, f'Excepted inputs with 9 dims, but given {_9}.'

        # split from code
        r_xyz, t_xyz, s_xyz = torch.chunk(transform_code, 3, dim=1)  # (n, 3)
        device = transform_code.device

        # transforms R, T and S
        R = _get_rotation_transform(r_xyz)

        T = _get_translation_transform(t_xyz)

        S = Scale(
            s_xyz,
            device=device
        )

        # apply full transforms to origins
        RST = R.compose(S).compose(T)
        origins = RST.transform_points(ray_bundle.origins)  # (n, i, 3)
        # apply R and S to directions
        RS = R.compose(S)
        directions = RS.transform_points(ray_bundle.directions)  # (n, i, 3)
        directions = torch.nn.functional.normalize(directions, p=2, dim=-1)
        # compute updated lengths
        points = ray_bundle.directions[:, :, None, :] * ray_bundle.lengths[:, :, :, None]  # (n, i, p, 3)
        n, i, p, _3 = points.shape
        points = S.transform_points(points.view(n, i * p, 3)).view(n, i, p, 3)
        lengths = points.norm(dim=-1)  # (n, i, p)

        return RayBundle(
            origins=origins,
            directions=directions,
            lengths=lengths,
            xys=ray_bundle.xys
        )


class RigidTransformModule(TransformModuleBase):
    """
    Rigid transform module that transform a ray bundle with 3x4 transform matrix
    """
    def __init__(self):
        super(RigidTransformModule, self).__init__()

        zeros_one = torch.tensor([0., 0., 0., 1.], dtype=torch.float).view(1, 1, 4)
        self.register_buffer('_zeros_one', zeros_one)

    def _transform(self, ray_bundle: RayBundle, transform_code: torch.Tensor):
        """
        Transform a ray bundle
        :param ray_bundle: input ray bundle
        :param transform_code: flatten transform matrix of 3x4, (n, 12)
        :return:
        """
        n, m = transform_code.shape
        assert m == 12, f'transform code is a vector of dim 12, but given {m}.'

        # to transform3d
        matrix = transform_code.view(n, 3, 4)
        matrix = torch.cat([matrix, self._zeros_one.expand(n, -1, -1)], dim=1)  # (n, 4, 4)
        transform = Transform3d(
            device=transform_code.device,
            matrix=torch.transpose(matrix, 1, 2)
        )

        # to points
        rays_points = ray_bundle_to_ray_points(ray_bundle)  # (n, i, p, 3)
        n, i, p, _3 = rays_points.shape

        # transform points
        rays_points = transform.transform_points(rays_points.view(n, i * p, 3), eps=1.0e-8).view(n, i, p, 3)

        # transform input origins
        origins = transform.transform_points(ray_bundle.origins, eps=1.0e-8)  # (n, i, 3)
        # the direction is the unit vector from one points to origins
        directions = rays_points[:, :, -1, :] - origins
        directions = torch.nn.functional.normalize(directions, dim=-1)  # (n, i, 3)
        # the length is the distance to origin points
        lengths = (rays_points - origins[:, :, None, :]).norm(dim=-1)  # (n, i, p)

        return RayBundle(
            origins=origins,
            directions=directions,
            lengths=lengths,
            xys=ray_bundle.xys
        )
