# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch
from pytorch3d.renderer import ImplicitRenderer
from pytorch3d.renderer.cameras import CamerasBase

from modules.pixelnerf import PixelEncoder
from nerf.raymarcher import EmissionAbsorptionNeRFRaymarcher
from nerf.raysampler import NeRFRaysampler, ProbabilisticRaysampler
from nerf.utils import calc_mse, calc_psnr, sample_images_at_mc_locs
from render_functions.pixel_nerf import PixelNerfResNet
from render_functions.pixel_nerf_mlp import PixelNerfMlp


class PixelRenderer(torch.nn.Module):
    """
    Renderer of pixel nerf
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
            function_type: str,
            function_config: dict,
            latent_code_dim: int = 512,
            encoder_layers: int = 34,
            mask_thr: float = 0.4,
            density_noise_std: float = 0.0
    ):
        """
        Args:
            image_size: The size of the rendered image (`[height, width]`).
            n_pts_per_ray: The number of points sampled along each ray for the
                coarse rendering pass.
            n_pts_per_ray_fine: The number of points sampled along each ray for the
                fine rendering pass.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            min_depth: The minimum depth of a sampled ray-point for the coarse rendering.
            max_depth: The maximum depth of a sampled ray-point for the coarse rendering.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
            chunk_size_test: The number of rays in each chunk of image rays.
                Active only when `self.training==True`.
            function_type: can be 'resnet' or 'mlp'
            latent_code_dim:
            encoder_layers
        """

        super().__init__()

        # Encoder
        self.encoder = PixelEncoder(n_layers=encoder_layers, latent_dim=latent_code_dim)

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        for render_pass in ("coarse", "fine"):
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
            self._implicit_function[render_pass] = self._get_implicit_function(
                function_type, function_config
            )
            print(f'Implicit function type: {function_type}.')

        self._mask_thr = mask_thr
        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size

    def _get_implicit_function(self, function_type: str, function_config: dict):
        if function_type == 'resnet':
            cls = PixelNerfResNet
        elif function_type == 'mlp':
            cls = PixelNerfMlp
        else:
            raise ValueError(f'function_type can only be "resnet" or "mlp", but given {function_type}.')

        if 'encoder_net' in function_config:
            raise ValueError('encoder_net can not be specified.')

        return cls(encoder_net=self.encoder, **function_config)

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor,
            chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            target_camera: A batch of cameras from which the scene is rendered.
            target_image: A batch of corresponding ground truth images of shape
                (n, h, w, 3).
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=target_camera,
                volumetric_function=self._implicit_function[renderer_pass],

                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=None,

                fg_probability=target_mask,
                fg_thr=self._mask_thr
            )

            if renderer_pass == "coarse":
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                if target_image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    rgb_gt = sample_images_at_mc_locs(
                        target_image[..., :3],
                        ray_bundle_out.xys,
                    )
                else:
                    rgb_gt = None

                # render depth
                depth_coarse = torch.sum(weights * ray_bundle_out.lengths, dim=-1, keepdim=True)  # (n, i, 1)

            elif renderer_pass == "fine":
                rgb_fine = rgb

                # render depth
                depth_fine = torch.sum(weights * ray_bundle_out.lengths, dim=-1, keepdim=True)  # (n, i, 1)

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {"rgb_fine": rgb_fine, "rgb_coarse": rgb_coarse, "rgb_gt": rgb_gt}

        out.update({
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine
        })

        return out

    def forward(
            self,
            source_camera: CamerasBase,
            source_image: torch.Tensor,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor
    ) -> Tuple[dict, dict]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            source_camera: A batch of cameras from which the scene is rendered.
            source_image: A batch of corresponding ground truth images of shape
                (1, 3, h, w)
            target_camera:
            target_image: can be None
            target_mask: probability of one pixel can be foreground or mask to indicate valid area
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.

                The shape of `rgb_coarse`, `rgb_fine`, `rgb_gt` depends on the
                `self.training` flag:
                    If `==True`, all 3 tensors are of shape
                    `(batch_size, n_rays_per_image, 3)` and contain the result
                    of the Monte Carlo training rendering pass.
                    If `==False`, all 3 tensors are of shape
                    `(batch_size, image_size[0], image_size[1], 3)` and contain
                    the result of the full image rendering pass.
            metrics: `dict` containing the error metrics comparing the fine and
                coarse renders to the ground truth:
                `mse_coarse`: Mean-squared error between the coarse render and
                    the input `image`
                `mse_fine`: Mean-squared error between the fine render and
                    the input `image`
                `psnr_coarse`: Peak signal-to-noise ratio between the coarse render and
                    the input `image`
                `psnr_fine`: Peak signal-to-noise ratio between the fine render and
                    the input `image`
        """
        batch_size = target_camera.R.shape[0]

        # encode source image first
        self.encoder.encode(source_image=source_image, source_camera=source_camera)

        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                batch_size,
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                target_camera,
                None if target_image is None else target_image.permute(0, 2, 3, 1).contiguous(),
                target_mask,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }

            out.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 1)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("depth_fine", "depth_coarse")
            })
        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if target_image is not None:
            for render_pass in ("coarse", "fine"):
                for metric_name, metric_fun in zip(
                        ("mse", "psnr"), (calc_mse, calc_psnr)
                ):
                    metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                        target_mask.permute(0, 2, 3, 1).contiguous() if not self.training else None
                    )

        return out, metrics
