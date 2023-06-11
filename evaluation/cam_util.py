import math
from typing import Tuple

import torch
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform


def generate_eval_video_cameras(
    ref_cameras: PerspectiveCameras,
    n_eval_cams: int = 100,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    focal_scale: float = 1.0,
    device='cpu'
) -> PerspectiveCameras:
    """
    Generate a camera trajectory for visualizing a NeRF model.

    Args:
        ref_cameras: reference cameras
        n_eval_cams: Number of cameras in the trajectory.
        up: The "up" vector of the scene (=the normal of the scene floor).
            Active for the `trajectory_type="circular"`.
        scene_center: The center of the scene in world coordinates which all
            the cameras from the generated trajectory look at.
        focal_scale:
        device:
    Returns:
        Dictionary of camera instances which can be used as the test dataset
    """
    cam_centers = ref_cameras.get_camera_center()  # (n, 3)
    # fit plane to the camera centers
    plane_mean = cam_centers.mean(dim=0)
    cam_centers_c = cam_centers - plane_mean[None]

    if up is not None:
        # us the up vector instead of the plane through the camera centers
        plane_normal = torch.FloatTensor(up).to(device)
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
    angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams, device=device)
    traj = traj_radius * torch.stack(
        (torch.zeros_like(angle, device=device), angle.cos(), angle.sin()), dim=-1
    )
    traj = traj @ e_vec.t() + plane_mean[None]

    # point all cameras towards the center of the scene
    R, T = look_at_view_transform(
        eye=traj,
        at=(scene_center,),  # (1, 3)
        up=plane_normal[None],  # (up,),  # (1, 3)
        device=traj.device,
    )

    # get the average focal length and principal point
    focal = ref_cameras.focal_length.mean(dim=0) * focal_scale
    p0 = ref_cameras.principal_point.mean(dim=0)

    val_cameras = PerspectiveCameras(
                    focal_length=focal[None],
                    principal_point=p0[None],
                    R=R,
                    T=T,
                    device=device
                )

    return val_cameras
