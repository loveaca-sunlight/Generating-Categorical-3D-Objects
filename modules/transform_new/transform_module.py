import torch

from pytorch3d.renderer import RayBundle
from .utils import vec6d_to_matrix


class TransformModule(torch.nn.Module):
    """
    Transform a ray-bundle according to given transform code.
    """

    def __init__(self):
        super(TransformModule, self).__init__()

        self.softplus = torch.nn.Softplus(beta=10)

        print('Using new transform module...')

    def forward(self, rays: RayBundle, transform_code: torch.Tensor, **kwargs):
        """
        Forward step
        :param rays: a ray bundle
        :param transform_code: (n, 10)
        :param kwargs:
        :return:
        """
        n, _10 = transform_code.shape
        assert _10 == 10
        # split
        vec_r, translation, vec_s = torch.split(transform_code, [6, 3, 1], dim=-1)  # (n, d)
        # rotation
        rotation = vec6d_to_matrix(vec_r)  # (n, 3, 3)
        # scale
        scale = self.softplus(vec_s)
        # origin
        origins = rays.origins[..., None]  # (n, i, 3, 1)
        origins_ = (rotation[:, None, :, :] @ origins + translation[:, None, :, None]) * scale[:, None, :, None]  # (n, i, 3, 1)
        # direction
        directions = rays.directions[..., None]  # (n, i, 3, 1)
        directions_ = rotation[:, None, :, :] @ directions  # (n, i, 3, 1)
        # lengths
        lengths = rays.lengths  # (n, i, p)
        lengths_ = lengths * scale[:, None, :]

        new_rays = RayBundle(
            origins=origins_.squeeze(-1),
            directions=directions_.squeeze(-1),
            lengths=lengths_,
            xys=rays.xys
        )

        # transform key points if offered
        key_points = kwargs.get('key_points', None)
        if key_points is None:
            return new_rays
        else:
            n, p, _3 = key_points.shape
            assert _3 == 3
            key_points_new = (rotation[:, None, :, :] @ key_points[..., None] + translation[:, None, :, None])\
                             * scale[:, None, :, None]
            return new_rays, key_points_new.squeeze(-1)
