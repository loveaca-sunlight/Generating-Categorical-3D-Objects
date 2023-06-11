import torch

from pytorch3d.transforms import Translate, euler_angles_to_matrix, Scale, Rotate

EULER_CONVENTION = 'XYZ'


def _get_rotation_transform(angles: torch.Tensor):
    """
    Return a rotation transform
    :param angles: (x, y, z) in radiance
    :return:
    """
    n, _3 = angles.shape
    assert _3 == 3

    rotation_matrix = euler_angles_to_matrix(-angles, EULER_CONVENTION)  # (n, 3, 3)
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


def _get_scale_transform(scales: torch.Tensor):
    """
    Return a scale transform
    :param scales: (s or xyz)
    :return:
    """
    if scales.dim() == 2:
        assert scales.shape[-1] == 3
    elif scales.dim() == 1:
        pass
    else:
        raise ValueError(f'The input tensor must be of 1 or 2 dims, but given {scales.shape}.')

    return Scale(
        scales,
        device=scales.device
    )
