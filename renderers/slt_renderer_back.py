import typing

import torch
from pytorch3d.renderer.cameras import CamerasBase

from loss_functions import calc_weights_reg, KeyPointMatchLoss
from modules.multiplex_new import TransformMultiplexDict
from nerf.utils import sample_images_at_mc_locs
from .renderer_base import RendererBase
from .util import select_template_weights


class SilhouetteBackRenderer(RendererBase):
    """
    A renderer that only renders silhouette
    """
    def __init__(
            self,
            sequences: typing.List[str],
            image_size: typing.Tuple[int, int],
            n_pts_per_ray: int,
            # n_pts_per_ray_fine: int,  only coarse
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            function_config: dict,
            mask_thr: float,
            n_transform_ways: int,
            template_key_points: torch.Tensor,
            match_tol: float,
            density_noise_std: float
    ):
        super().__init__(
            image_size=image_size,
            n_pts_per_ray=n_pts_per_ray,
            n_pts_per_ray_fine=0,
            n_rays_per_image=n_rays_per_image,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified=stratified,
            stratified_test=stratified_test,
            chunk_size_test=chunk_size_test,
            implicit_function_config=function_config,
            mask_thr=mask_thr,
            density_noise_std=density_noise_std,
            render_passes=('coarse',)
        )

        # transform multiplexes
        self.transforms = TransformMultiplexDict(sequences, n_transform_ways)

        # key point match loss
        self.match_loss = KeyPointMatchLoss(
            target_points=template_key_points,
            tol=match_tol
        )

    def load_template_weights(self, state_dict: dict):
        """
        Only suit for template_mlp
        :param state_dict:
        :return:
        """
        # Select Weights from fine pass
        weights = select_template_weights(state_dict, fine_only=True)

        # Load Weights from fine pass
        with torch.no_grad():
            for render_pass in ['coarse']:   # load to coarse function only
                for module_name, module in [('mlp_xyz', self._implicit_function[render_pass].mlp_xyz),
                                            ('density_layer', self._implicit_function[render_pass].density_layer)]:
                    for name, param in module.named_parameters():
                        param.data.copy_(weights[f'_implicit_function.fine.{module_name}.{name}'])

    def freeze_template_weights(self):
        for f in self._implicit_function.values():
            f.freeze_template_layers()

    def transform_parameters(self):
        """
        Return parameters of transforms
        :return:
        """
        return self.transforms.parameters()

    def _get_transform_code(self, sequence_name: str):
        # (m, d)
        return self.transforms(sequence_name)

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            transform_codes: torch.Tensor,
            key_points: torch.Tensor,
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
        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse",):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=target_camera,
                volumetric_function=self._implicit_function[renderer_pass],

                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=0.0,
                camera_hash=None,

                transform_codes=transform_codes,
                key_points=key_points,

                fg_probability=target_mask,
                fg_thr=self._mask_thr
            )

            n_multiplexes = self.transforms.num_multiplexes

            if renderer_pass == "coarse":
                if target_image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    with torch.no_grad():
                        rgb_gt = sample_images_at_mc_locs(
                            target_image.permute(0, 2, 3, 1).contiguous(),
                            ray_bundle_out.xys,
                        ).repeat(n_multiplexes, 1, 1)  # (m * n, i, 3)
                else:
                    rgb_gt = None

                # Sample the ground truth fg_probability
                if target_fg_probability is not None:
                    with torch.no_grad():
                        fg_gt = sample_images_at_mc_locs(
                            target_fg_probability.permute(0, 2, 3, 1).contiguous(),
                            ray_bundle_out.xys
                        ).repeat(n_multiplexes, 1, 1)  # (m * n, i, 1)
                else:
                    fg_gt = None

                # Sum of weights along a ray, (m * n, i)
                weights_coarse = weights.sum(dim=-1)

                # render depth
                depth_coarse = torch.sum(weights * ray_bundle_out.lengths.repeat(n_multiplexes, 1, 1),
                                         dim=-1, keepdim=True)  # (m * n, i, 1)

                # key points
                transformed_key_points = self._implicit_function[renderer_pass].transformed_key_points
            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {"rgb_gt": rgb_gt}

        out.update({
            'depth_coarse': depth_coarse,
            'fg_gt': fg_gt,
            'weights_coarse': weights_coarse,
            'key_points': transformed_key_points
        })

        return out

    def update_multiplex_score(self, sequence_name: str, scores: torch.Tensor):
        self.transforms.update_scores(sequence_name, scores)

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            key_points: torch.Tensor,
            key_ids: typing.List[int],
            sequence_name: str
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
            target_fg_probability:
            target_mask: probability of one pixel can be foreground or mask to indicate valid area
            key_points: (1, 2, 3)
            key_ids: the indexes of key points
            sequence_name:
        """
        batch_size = target_camera.R.shape[0]
        n_multiplexes = self.transforms.num_multiplexes

        # get rigid transform codes
        rigid_codes = self._get_transform_code(sequence_name)

        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                batch_size,  # n_multiplexes = 1 when self.training == false
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                target_camera=target_camera,
                target_image=target_image,
                target_fg_probability=target_fg_probability,
                target_mask=target_mask,
                transform_codes=rigid_codes,
                key_points=(key_points if chunk_idx == 0 else None),
                chunk_idx=chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            outs = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_gt",)
            }

            outs.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 1)  # n_multiplexes = 1 when self.training == false
                if chunk_outputs[0][k] is not None
                else None
                for k in ("depth_coarse", 'fg_gt')
            })

            outs.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size)  # n_multiplexes = 1 when self.training == false
                if chunk_outputs[0][k] is not None
                else None
                for k in ("weights_coarse", )
            })

            outs['key_points'] = chunk_outputs[0]['key_points']
        else:
            outs = chunk_outputs[0]

        #
        # Calc error metrics
        #

        # Dict[Tensor] to List[Dict[Tensor]]
        outs = {
            name: torch.chunk(tensor, n_multiplexes, dim=0) if tensor is not None else None
            for name, tensor in outs.items()
        }
        outs = [
            {name: (tensors[i] if tensors is not None else None) for name, tensors in outs.items()}
            for i in range(n_multiplexes)
        ]

        # Compute metrics for each transform multiplex
        multiplex_metrics = []
        for out in outs:
            metrics = {}

            # Calc weights reg
            fg_probability = out['fg_gt']
            for render_pass in ("coarse", ):
                metrics[f'weights_reg_{render_pass}'] = calc_weights_reg(out[f'weights_{render_pass}'],
                                                                         fg_probability.squeeze(-1))

            # Calc match loss
            transformed_points = out['key_points']
            metrics['match_loss'] = self.match_loss(
                input_points=transformed_points,
                input_ids=key_ids
            )

            # append to list
            multiplex_metrics.append(metrics)

        # each element stores all outputs and metrics
        outs = outs[0] if len(outs) == 1 else outs
        multiplex_metrics = multiplex_metrics[0] if len(multiplex_metrics) == 1 else multiplex_metrics

        return outs, multiplex_metrics
