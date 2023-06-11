import torch
import numpy as np

from pytorch3d.renderer import RayBundle
from pytorch3d.renderer.cameras import CamerasBase
from .util import get_bounding_points


class NerfMonteCarloRaysampler(torch.nn.Module):
    """
    An adapted raysampler that accept dynamic min_depth and max_depth
    """
    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        n_rays_per_image: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        edge_thr: float = 1.,
        negative_z: bool = False,
    ) -> None:
        """
        Args:
            min_x: The smallest x-coordinate of each ray's source pixel.
            max_x: The largest x-coordinate of each ray's source pixel.
            min_y: The smallest y-coordinate of each ray's source pixel.
            max_y: The largest y-coordinate of each ray's source pixel.
            n_rays_per_image: The number of rays randomly sampled in each camera.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of each ray-point.
            max_depth: The maximum depth of each ray-point. Set min_depth and max_depth to None to enable bounds mode
        """
        super().__init__()
        self.register_buffer('_min_x', torch.tensor(min_x, dtype=torch.float))
        self.register_buffer('_max_x', torch.tensor(max_x, dtype=torch.float))
        self.register_buffer('_min_y', torch.tensor(min_y, dtype=torch.float))
        self.register_buffer('_max_y', torch.tensor(max_y, dtype=torch.float))
        self._n_rays_per_image = n_rays_per_image
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._negative_z = negative_z
        self._edge_thr = edge_thr

    def forward(self, cameras: CamerasBase, fg_probability: torch.Tensor = None,
                fg_thr: float = None, bounds: torch.Tensor = None, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            fg_probability: foreground probability.
            fg_thr: threshold for mask.
            bounds: (n, 2)
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, n_rays_per_image, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, n_rays_per_image, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, n_rays_per_image, 2)`
                containing the 2D image coordinates of each ray.
        """
        batch_size = cameras.R.shape[0]

        device = cameras.device

        # check parameters for dynamic sampling
        assert fg_probability is not None, 'fg_probability can not be None when using dynamic ray sampling.'
        assert fg_thr is not None, 'fg_thr can not be None when using dynamic ray sampling.'

        # calculate min and max depths w.r.t cameras
        with torch.no_grad():
            if bounds is not None:
                assert (self._min_depth is None) and (self._max_depth is None)
                min_depths = bounds[:, 0]  # (n,)
                max_depths = bounds[:, 1]  # (n,)
            else:
                cameras_center = cameras.get_camera_center()
                distances = torch.norm(cameras_center, p=2, dim=-1, keepdim=False)  # (n,)
                min_depths = torch.clamp_min(distances - self._min_depth, 0.0)
                max_depths = distances + self._max_depth

            # get the initial grid of image xy coords
            # of shape (batch_size, n_rays_per_image, 2)

            # get dynamic bounds
            bound_x, bound_y = get_bounding_points(fg_probability, fg_thr)  # (n, 2)

            # switch to ndc coordinates
            h, w = fg_probability.shape[-2:]
            bound_x = -1.0 * (bound_x / (w - 1.0) * 2.0 - 1.0)
            bound_y = -1.0 * (bound_y / (h - 1.0) * 2.0 - 1.0)
            max_x, min_x = bound_x[:, 0], bound_x[:, 1]
            max_y, min_y = bound_y[:, 0], bound_y[:, 1]

            # clamp the value, (n,)
            min_x = torch.maximum(min_x, self._min_x)[:, None, None]
            max_x = torch.minimum(max_x, self._max_x)[:, None, None]
            min_y = torch.maximum(min_y, self._min_y)[:, None, None]
            max_y = torch.minimum(max_y, self._max_y)[:, None, None]

            assert torch.all(max_x - min_x >= 0.0) and torch.all(max_y - min_y >= 0.0)

            xrand = torch.rand(
                        size=(batch_size, self._n_rays_per_image, 1),
                        dtype=torch.float32,
                        device=device,
                    )
            xrand = torch.cat([xrand[:, :int(self._n_rays_per_image * self._edge_thr), ] * (max_x - min_x) + min_x,
                                xrand[:, int(self._n_rays_per_image * self._edge_thr):, ]],dim=1)
            yrand = torch.rand(
                        size=(batch_size, self._n_rays_per_image, 1),
                        dtype=torch.float32,
                        device=device,
                    )
            yrand = torch.cat([yrand[:, :int(self._n_rays_per_image * self._edge_thr), ] * (max_y - min_y) + min_y,
                                yrand[:, int(self._n_rays_per_image * self._edge_thr):, ]],dim=1)
            
            # dynamically choose boundaries according to foreground mask 根据前景蒙版动态选择边界
            rays_xy = torch.cat(
                [
                    xrand,yrand,
                ],
                dim=2
            )

        return _xy_to_ray_bundle(
            cameras, rays_xy, min_depths, max_depths, self._n_pts_per_ray, negative_z=self._negative_z
        )


@torch.no_grad()
def _xy_to_ray_bundle(
    cameras: CamerasBase,
    xy_grid: torch.Tensor,
    min_depths: torch.Tensor,
    max_depths: torch.Tensor,
    n_pts_per_ray: int,
    negative_z: bool = False,
) -> RayBundle:
    """
    Extends the `xy_grid` input of shape `(batch_size, ..., 2)` to rays.
    This adds to each xy location in the grid a vector of `n_pts_per_ray` depths
    uniformly spaced between `min_depth` and `max_depth`.

    The extended grid is then unprojected with `cameras` to yield
    ray origins, directions and depths.

    An adapted version that can handle different depth ranges.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()  # pyre-ignore

    # ray z-coords
    assert min_depths.shape[0] == batch_size and max_depths.shape[0] == batch_size
    depths = [torch.linspace( #分割
        min_d, max_d, n_pts_per_ray, dtype=xy_grid.dtype, device=xy_grid.device
    ) for min_d, max_d in zip(min_depths, max_depths)]

    rays_zs = torch.stack(depths, dim=0)[:, None, :].expand(batch_size, n_rays_per_image, n_pts_per_ray)

    # make two sets of points at a constant depth=1 and 2 or depth=1 and 0
    to_unproject = torch.cat(
        (
            xy_grid.view(batch_size, 1, n_rays_per_image, 2)
            .expand(batch_size, 2, n_rays_per_image, 2)
            .reshape(batch_size, n_rays_per_image * 2, 2),
            torch.cat(
                (
                    xy_grid.new_ones(batch_size, n_rays_per_image, 1),  # pyre-ignore
                    2.0 * xy_grid.new_ones(batch_size, n_rays_per_image, 1),
                ),
                dim=1,
            ),
        ),
        dim=-1,
    )

    # unproject the points
    unprojected = cameras.unproject_points(to_unproject)  # pyre-ignore

    # split the two planes back
    rays_plane_1_world = unprojected[:, :n_rays_per_image]
    rays_plane_2_world = unprojected[:, n_rays_per_image:]

    # directions are the differences between the two planes of points
    rays_directions_world = rays_plane_2_world - rays_plane_1_world

    # origins are given by subtracting the ray directions from the first plane
    rays_origins_world = rays_plane_1_world - rays_directions_world

    return RayBundle(
        rays_origins_world.view(batch_size, *spatial_size, 3),
        rays_directions_world.view(batch_size, *spatial_size, 3) * (-1.0 if negative_z else 1.0),
        rays_zs.view(batch_size, *spatial_size, n_pts_per_ray),
        xy_grid,
    )


class NerfGridRaysampler(torch.nn.Module):
    def __init__(
            self,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            image_width: int,
            image_height: int,
            n_pts_per_ray: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            negative_z: bool = False
    ) -> None:
        """
        Args:
            min_x: The leftmost x-coordinate of each ray's source pixel's center.
            max_x: The rightmost x-coordinate of each ray's source pixel's center.
            min_y: The topmost y-coordinate of each ray's source pixel's center.
            max_y: The bottommost y-coordinate of each ray's source pixel's center.
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point. The radius when object locates in origin.
            max_depth: The maximum depth of a ray-point. The radius when object locates in origin. Set min_depth and
                       max_depth to None to enable bounds mode
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._n_rays_per_image = n_rays_per_image
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._negative_z = negative_z
        self._image_height = image_height
        self._image_width = image_width

        # get the initial grid of image xy coords
        _xy_grid = torch.stack(
            tuple(
                reversed(
                    torch.meshgrid(
                        torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
                        torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
                    )
                )
            ),
            dim=-1,
        )
        self.register_buffer("_xy_grid", _xy_grid, persistent=False)

    def forward(self, cameras: CamerasBase, fg_probability: torch.Tensor = None,
                fg_thr: float = None, bounds: torch.Tensor = None, ray_sampler_config: str = None, **kwargs) -> RayBundle:
        """
        Args:
            cameras: A batch of `batch_size` cameras from which the rays are emitted.
            bounds:
        Returns:
            A named tuple RayBundle with the following fields:
            origins: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the locations of ray origins in the world coordinates.
            directions: A tensor of shape
                `(batch_size, image_height, image_width, 3)`
                denoting the directions of each ray in the world coordinates.
            lengths: A tensor of shape
                `(batch_size, image_height, image_width, n_pts_per_ray)`
                containing the z-coordinate (=depth) of each ray in world units.
            xys: A tensor of shape
                `(batch_size, image_height, image_width, 2)`
                containing the 2D image coordinates of each ray.
        """

        batch_size = cameras.R.shape[0]

        device = cameras.device

        # calculate min and max depths (near and far bounds) w.r.t cameras
        with torch.no_grad():
            if bounds is not None:
                assert (self._min_depth is None) and (self._max_depth is None)
                min_depths = bounds[:, 0]  # (n,)
                max_depths = bounds[:, 1]  # (n,)
            else:
                cameras_center = cameras.get_camera_center()
                distances = torch.norm(cameras_center, p=2, dim=-1, keepdim=False)  # (n,)
                min_depths = torch.clamp_min(distances - self._min_depth, 0.0)
                max_depths = distances + self._max_depth

        if fg_probability == None:
            # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
            xy_grid = self._xy_grid.to(device)[None].expand(  # pyre-ignore
                batch_size, *self._xy_grid.shape
            )

        else:
            # get dynamic bounds
            #bound_x, bound_y = get_bounding_points(fg_probability, fg_thr)  # (n, 2)
            bound_x, bound_y = get_bounding_points(fg_probability, fg_thr) 
            
            # switch to ndc coordinates
            h, w = fg_probability.shape[-2:]
            bound_x = -1.0 * (bound_x / (w - 1.0) * 2.0 - 1.0)
            bound_y = -1.0 * (bound_y / (h - 1.0) * 2.0 - 1.0)
            max_x, min_x = bound_x[:, 0], bound_x[:, 1]
            max_y, min_y = bound_y[:, 0], bound_y[:, 1]

            # clamp the value, (n,)
            '''min_x_num = min_x.min()
            max_x_num = max_x.max()
            min_y_num = min_y.min()
            max_y_num = max_y.max()'''

            assert torch.all(max_x - min_x >= 0.0) and torch.all(max_y - min_y >= 0.0)

            #xy_grid = self._xy_grid.to(device)[None]
            xy_grid = []
            for i in range(0,batch_size):
                #move_x = ((max_x[i] - min_x[i])/self._n_rays_per_image).item()
                #move_y = ((max_y[i] - min_y[i])/self._n_rays_per_image).item()#间隔距离
                len_x = max_x[i] - min_x[i]
                len_y = max_y[i] - min_y[i]

                #这个暂时不处理
                #num_x, num_y = len_x.div(self._n_rays_per_image), len_y.div(self._n_rays_per_image * 1.5)

                if ray_sampler_config == None:
                    size_x = torch.rand(1).cuda() * (len_x-len_x.div(2)) + len_x.div(2) #选框大小
                    size_y = torch.rand(1).cuda() * (len_y-len_y.div(3)) + len_y.div(3)
                    #pos_x = np.random.randint(move_x) - (move_x/2)
                    #pos_x = (np.random.randint(1) * move_x) - (move_x/2)
                    #pos_y = (np.random.randint(1) * move_y) - (move_y/2)
                    #起始位置
                    pos_x = torch.rand(1).cuda() * (len_x - size_x + size_x.div(self._n_rays_per_image)) 
                    pos_y = torch.rand(1).cuda() * (len_y - size_y + size_y.div(self._n_rays_per_image * 1.2)) 
                    
                    new_xy_grid = torch.stack(
                        tuple(
                            reversed(
                                torch.meshgrid(
                                    torch.linspace((max_y[i]-pos_y).item(), (max_y[i]-pos_y-size_y).item(), int(self._n_rays_per_image * 1.2), dtype=torch.float32),
                                    torch.linspace((min_x[i]+pos_x).item(), (min_x[i]+pos_x+size_x).item(), self._n_rays_per_image, dtype=torch.float32),
                                )
                            )
                        ),
                        dim=-1,
                    )
                elif ray_sampler_config == 'square':
                    size_x = torch.rand(1).cuda() * (len_x-len_x.div(3)) + len_x.div(4) #选框大小
                    size_y = torch.rand(1).cuda() * (len_y-len_y.div(3)) + len_y.div(4)
                    #pos_x = np.random.randint(move_x) - (move_x/2)
                    #pos_x = (np.random.randint(1) * move_x) - (move_x/2)
                    #pos_y = (np.random.randint(1) * move_y) - (move_y/2)
                    #起始位置
                    pos_x = torch.rand(1).cuda() * (len_x - size_x + size_x.div(self._n_rays_per_image)) 
                    pos_y = torch.rand(1).cuda() * (len_y - size_y + size_y.div(self._n_rays_per_image)) 
                    
                    new_xy_grid = torch.stack(
                        tuple(
                            reversed(
                                torch.meshgrid(
                                    torch.linspace((max_y[i]-pos_y).item(), (max_y[i]-pos_y-size_y).item(), self._n_rays_per_image, dtype=torch.float32),
                                    torch.linspace((min_x[i]+pos_x).item(), (min_x[i]+pos_x+size_x).item(), self._n_rays_per_image, dtype=torch.float32),
                                )
                            )
                        ),
                        dim=-1,
                    )
                elif ray_sampler_config == 'full_img':
                    new_xy_grid = torch.stack(
                        tuple(
                            reversed(
                                torch.meshgrid(
                                    torch.linspace(1.0, -1.0, self._n_rays_per_image, dtype=torch.float32),
                                    torch.linspace(-1.0, 1.0, self._n_rays_per_image, dtype=torch.float32),
                                )
                            )
                        ),
                        dim=-1,
                    )
                elif ray_sampler_config == 'mask_img':
                    pos_x = torch.rand(1).cuda() * (len_x.div(self._n_rays_per_image) - (len_x.div(self._n_rays_per_image))/2)
                    pos_y = torch.rand(1).cuda() * (len_y.div(self._n_rays_per_image) - (len_y.div(self._n_rays_per_image))/2)
                    #这里即使超出了1.0也不会影响结果 三维空间内渲染超一点正常
                    new_xy_grid = torch.stack(
                        tuple(
                            reversed(
                                torch.meshgrid(
                                    torch.linspace((max_y[i]-pos_y).item(), (max_y[i]-pos_y-len_y).item(), self._n_rays_per_image, dtype=torch.float32),
                                    torch.linspace((min_x[i]+pos_x).item(), (min_x[i]+pos_x+len_x).item(), self._n_rays_per_image, dtype=torch.float32),
                                )
                            )
                        ),
                        dim=-1,
                    )
                
                
                xy_grid.append(new_xy_grid)
            
            # expand the (H, W, 2) grid batch_size-times to (B, H, W, 2)
            #xy_grid = xy_grid.expand(  # pyre-ignore
            #    batch_size, *self._xy_grid.shape
            #)
            xy_grid = torch.stack(xy_grid,dim=0).cuda()
            #57 * 48 * 2
        return _xy_to_ray_bundle(
            cameras, xy_grid, min_depths, max_depths, self._n_pts_per_ray, negative_z=self._negative_z
        )


class NerfNDCGridRaysampler(NerfGridRaysampler):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        negative_z: bool,
        n_rays_per_image: int = 0,
    ) -> None:
        """
        Args:
            image_width: The horizontal size of the image grid.
            image_height: The vertical size of the image grid.
            n_pts_per_ray: The number of points sampled along each ray.
            min_depth: The minimum depth of a ray-point.
            max_depth: The maximum depth of a ray-point.
        """
        half_pix_width = 1.0 / image_width
        half_pix_height = 1.0 / image_height
        super().__init__(
            min_x=1.0 - half_pix_width,
            max_x=-1.0 + half_pix_width,
            min_y=1.0 - half_pix_height,
            max_y=-1.0 + half_pix_height,
            image_width=image_width,
            image_height=image_height,
            n_pts_per_ray=n_pts_per_ray,
            n_rays_per_image=n_rays_per_image,
            min_depth=min_depth,
            max_depth=max_depth,
            negative_z=negative_z
        )
