import typing

import torch
from pytorch3d.renderer.cameras import CamerasBase

from nerf.utils import calc_mse, calc_psnr, sample_images_at_mc_locs
from .renderer_base import RendererBase


class SceneRenderer(RendererBase):
    def __init__(
            self,
            image_size: typing.Tuple[int, int],
            n_pts_per_ray: int,
            n_pts_per_ray_fine: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            function_config: dict,
            mask_thr: float,
            density_noise_std: float
    ):
        super().__init__(
            image_size=image_size,
            n_pts_per_ray=n_pts_per_ray,
            n_pts_per_ray_fine=n_pts_per_ray_fine,
            n_rays_per_image=n_rays_per_image,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified=stratified,
            stratified_test=stratified_test,
            chunk_size_test=chunk_size_test,
            implicit_function_config=function_config,
            mask_thr=mask_thr,
            density_noise_std=density_noise_std
        )

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
                    with torch.no_grad():
                        rgb_gt = sample_images_at_mc_locs(
                            target_image.permute(0, 2, 3, 1).contiguous(),
                            ray_bundle_out.xys,
                        )
                else:
                    rgb_gt = None

                # Sum of weights along a ray, (n, i)
                weights_coarse = weights.sum(dim=-1)

                # render depth
                depth_coarse = torch.sum(weights * ray_bundle_out.lengths, dim=-1, keepdim=True)  # (n, i, 1)

            elif renderer_pass == "fine":
                rgb_fine = rgb

                # Sum of weights along a ray, (n, i)
                weights_fine = weights.sum(dim=-1)

                # render depth
                depth_fine = torch.sum(weights * ray_bundle_out.lengths, dim=-1, keepdim=True)  # (n, i, 1)

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {"rgb_fine": rgb_fine, "rgb_coarse": rgb_coarse, "rgb_gt": rgb_gt}

        out.update({
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine,
            'weights_coarse': weights_coarse,
            'weights_fine': weights_fine
        })

        return out

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor,
    ) -> typing.Tuple[dict, dict]:
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
            target_camera:
            target_image: can be None
            target_mask: probability of one pixel can be foreground or mask to indicate valid area
        """
        batch_size = target_camera.R.shape[0]

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
                target_camera=target_camera,
                target_image=target_image,
                target_mask=target_mask,
                chunk_idx=chunk_idx,
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
                ).view(-1, *self._image_size, 3)
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
