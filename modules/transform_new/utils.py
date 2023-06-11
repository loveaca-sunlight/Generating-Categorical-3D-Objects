import torch
import torch.nn.functional as F

from pytorch3d.transforms import euler_angles_to_matrix


def vec6d_to_matrix(vec6d: torch.Tensor):
    """
    Transform a 6d vector to a rotation matrix
    :param vec6d: (n, 6)
    :return:
    """
    n, _6 = vec6d.shape
    assert _6 == 6

    a1, a2 = torch.chunk(vec6d, 2, dim=-1)  # (n, 3)

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1)  # (n, 3, 3)


def matrix_to_vec6d(matrix: torch.Tensor):
    """
    Transform a matrix to vec6d
    :param matrix: (n, 3, 3)
    :return:
    """
    n, _3, _3 = matrix.shape
    vec6d = matrix[:, :, : 2].transpose(-1, -2).reshape(n, 6)
    return vec6d


def euler_to_vec6d(euler_angles: torch.Tensor, convention: str):
    """
    Transform euler angles to vec6d
    :param euler_angles: (n, 3)
    :param convention:
    :return:
    """
    matrix = euler_angles_to_matrix(euler_angles, convention)  # (n, 3, 3)
    return matrix_to_vec6d(matrix)
