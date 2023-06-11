import math
import numpy as np
import mmcv
from typing import Tuple, List, Dict

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform


def generate_eval_video_cameras(
    train_cameras: List[PerspectiveCameras],
    n_eval_cams: int = 100,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> List[PerspectiveCameras]:
    """
    Generate a camera circular trajectory for visualizing a NeRF model.

    Args:
        train_cameras: Cameras used in training
        n_eval_cams: Number of cameras in the trajectory.
        up: The "up" vector of the scene (=the normal of the scene floor).
            Active for the `trajectory_type="circular"`.
        scene_center: The center of the scene in world coordinates which all
            the cameras from the generated trajectory look at.
    Returns:
        Dictionary of camera instances which can be used as the test dataset
    """
    cam_centers = torch.cat(
        [e.get_camera_center() for e in train_cameras]
    )

    # fit plane to the camera centers
    plane_mean = cam_centers.mean(dim=0)
    cam_centers_c = cam_centers - plane_mean[None]

    if up is not None:
        # us the up vector instead of the plane through the camera centers
        plane_normal = torch.FloatTensor(up)
    else:
        cov = (cam_centers_c.t() @ cam_centers_c) / cam_centers_c.shape[0]
        _, e_vec = torch.symeig(cov, eigenvectors=True)
        plane_normal = e_vec[:, 0]

    plane_dist = (plane_normal[None] * cam_centers_c).sum(dim=-1)
    cam_centers_on_plane = cam_centers_c - plane_dist[:, None] * plane_normal[None]

    cov = (
        cam_centers_on_plane.t() @ cam_centers_on_plane
    ) / cam_centers_on_plane.shape[0]
    _, e_vec = torch.symeig(cov, eigenvectors=True)
    traj_radius = (cam_centers_on_plane ** 2).sum(dim=1).sqrt().mean()
    angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams)
    traj = traj_radius * torch.stack(
        (torch.zeros_like(angle), angle.cos(), angle.sin()), dim=-1
    )
    # traj = traj_radius * torch.stack(
    #     (angle.sin(), angle.cos(), torch.zeros_like(angle)), dim=-1
    # )
    traj = traj @ e_vec.t() + plane_mean[None]

    # point all cameras towards the center of the scene
    R, T = look_at_view_transform(
        eye=traj,
        at=(scene_center,),  # (1, 3)
        up=(up,),  # (1, 3)
        device=traj.device,
    )

    # get the average focal length and principal point
    focal = torch.cat([e.focal_length for e in train_cameras]).mean(dim=0)
    p0 = torch.cat([e.principal_point for e in train_cameras]).mean(dim=0)

    # assemble the dataset
    test_dataset = [
        PerspectiveCameras(
                focal_length=focal[None],
                principal_point=p0[None],
                R=R_[None],
                T=T_[None],
            )
        for i, (R_, T_) in enumerate(zip(R, T))
    ]

    return test_dataset


def generate_circular_cameras(
    train_cameras: List[PerspectiveCameras],
    n_eval_cams: int = 100,
    up_scale: float = 1.0
):
    """
    Generate cameras
    :param train_cameras:
    :param n_eval_cams:
    :param up_scale:
    :return:
    """
    cam_centers = torch.cat(
        [e.get_camera_center() for e in train_cameras]
    )  # (n, 3)

    plane_mean = cam_centers.mean(dim=0)[None]

    cam_vecs = cam_centers - plane_mean  # (n, 3)

    dist_mean = torch.norm(cam_vecs, p=2, dim=-1).mean()

    # position
    view_center = plane_mean.clone()
    view_center[..., -1] *= up_scale
    # view_center[..., : -1] = 0.0
    angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams)
    traj = dist_mean * torch.stack(
        [angle.cos(), angle.sin(), torch.zeros_like(angle)], dim=-1
    ) + view_center

    # point all cameras towards the center of the scene
    R, T = look_at_view_transform(
        eye=traj,
        device=traj.device,
        up=((0.1, -0.7, -0.6),)
    )

    # get the average focal length and principal point
    focal = torch.cat([e.focal_length for e in train_cameras]).mean(dim=0)
    p0 = torch.cat([e.principal_point for e in train_cameras]).mean(dim=0)

    # assemble the dataset
    test_dataset = [
        PerspectiveCameras(
            focal_length=focal[None],
            principal_point=p0[None],
            R=R_[None],
            T=T_[None],
        )
        for i, (R_, T_) in enumerate(zip(R, T))
    ]

    return test_dataset


def generate_inference_cameras(
    focal_length: Tuple,
    principal_point: Tuple,
    n_eval_cams: int = 100,
    high: float = 8.0,
    radius: float = 8.0,
):
    """
    Generate cameras used for inference
    :param n_eval_cams:
    :param high:
    :param radius:
    :param focal_length: (2,)
    :param principal_point: (2,)
    :return:
    """
    plane_mean = torch.tensor(
        [0.0, 0.0, high], dtype=torch.float
    ).view(1, 3)

    dist_mean = radius

    # position
    view_center = plane_mean
    angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams)
    traj = dist_mean * torch.stack(
        [angle.cos(), angle.sin(), torch.zeros_like(angle)], dim=-1
    ) + view_center

    # point all cameras towards the center of the scene
    R, T = look_at_view_transform(
        eye=traj,
        device=traj.device,
        up=((0, 0, 1),)
    )

    # get the average focal length and principal point
    focal = torch.tensor(focal_length, dtype=torch.float).expand(n_eval_cams, -1)
    p0 = torch.tensor(principal_point, dtype=torch.float).expand(n_eval_cams, -1)

    # assemble the dataset
    test_cameras = PerspectiveCameras(
        focal_length=focal,
        principal_point=p0,
        R=R,
        T=T,
    )

    return test_cameras


def tensors2images(tensors: List[torch.Tensor], tensor_format: str, reverse_channel: bool):
    images = [np.round(255.0 * tensor2image(tensor, tensor_format, reverse_channel)).astype(np.uint8)
              for tensor in tensors]
    return images


def tensors2depths(tensors: List[torch.Tensor], tensor_format: str):
    depths = [tensor2image(tensor, tensor_format, False, clamp=False).squeeze() for tensor in tensors]  # [(h, w), ...]

    depths = [np.clip(np.round(10.0 * depth), a_min=0.0, a_max=255.0).astype(np.uint8)
              for depth in depths]

    return depths


def tensor2image(tensor: torch.Tensor, tensor_format: str, reverse_channel: bool, clamp: bool = True):
    """
    Convert tensor to numpy image
    :param tensor:
    :param tensor_format:
    :param reverse_channel:
    :param clamp:
    :return:
    """
    if clamp:
        tensor = tensor.clamp(min=0.0, max=1.0)

    img = tensor.cpu().numpy()

    if tensor_format == 'HWC':
        pass
    elif tensor_format == 'CHW':
        img = np.moveaxis(img, 0, -1)
    else:
        raise ValueError(f'Unknown tensor_format: {tensor_format}.')

    if reverse_channel:
        img = img[:, :, ::-1]

    return img


def save_image(img: np.ndarray, path: str):
    if img.ndim == 3:
        img = img[:, :, ::-1]  # to BGR
    mmcv.imwrite(img, path)


def convert_from_old_version(state_dict):
    """
    Convert state dict from old version
    :param state_dict: state dict of old version
    :return:
    """
    new_version = {}
    for k, v in state_dict.items():
        if k.startswith('model._implicit_function'):
            parts: list = k.split('.')
            parts.insert(3, 'mlp')
            name = '.'.join(parts)
        else:
            name = k

        new_version[name] = v
    return new_version


def dicts_to_csv(data: List[Dict[str, object]], fn: str):
    assert len(data) > 0, 'Given data is empty.'

    keys = data[0].keys()

    to_save = [','.join(keys) + '\n']
    for d in data:
        item = [
            str(d[k]) for k in keys
        ]
        to_save.append(','.join(item) + '\n')

    with open(fn, 'w') as f:
        f.writelines(to_save)


def depth_to_mask(depths: List[torch.Tensor], min_depth: float = 0.01, max_depth: float = 50.0):
    return [
        ((dpt > min_depth) & (dpt < max_depth)).float()
        for dpt in depths
    ]
