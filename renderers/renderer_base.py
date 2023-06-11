import torch

from nerf.raymarcher import EmissionAbsorptionNeRFRaymarcher
from nerf.raysampler import NeRFRaysampler, ProbabilisticRaysampler
from typing import Tuple
from pytorch3d.renderer import ImplicitRenderer
from render_functions import IMPLICIT_FUNCTIONS


class RendererBase(torch.nn.Module):
    """
    Base class for renderers
    """
    def __init__(
            self,
            image_size: Tuple[int, int],
            n_pts_per_ray: int,
            n_pts_per_ray_fine: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            implicit_function_config: dict,
            mask_thr: float = 0.0,
            density_noise_std: float = 0.0,
            render_passes: Tuple[str] = ('coarse', 'fine'),
            negative_z: bool = False
    ):
        super(RendererBase, self).__init__()

        assert len(render_passes) > 0, 'render_passes is empty.'

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        for render_pass in render_passes:
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                raysampler = NeRFRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    stratified=stratified,
                    stratified_test=stratified_test,
                    n_rays_per_image=n_rays_per_image,
                    image_height=image_height,
                    image_width=image_width,
                    negative_z=negative_z
                )
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray_fine,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            else:
                raise ValueError(f"No such rendering pass {render_pass}")

            # Initialize the fine/coarse renderer.
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            # Instantiate the fine/coarse NeuralRadianceField module.
            self._implicit_function[render_pass] = IMPLICIT_FUNCTIONS.build(implicit_function_config) 

        self._mask_thr = mask_thr
        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size

        # show parameters
        print(f'Image Size: {self._image_size}, Mask Threshold: {self._mask_thr}, Negative Z: {negative_z}.')
        print(f'Implicit Function: {type(self._implicit_function[render_passes[0]]).__name__}.')
