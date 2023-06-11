# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import typing
from typing import Tuple

import torch
from pytorch3d.renderer.cameras import CamerasBase

from loss_functions import calc_l2_reg
from modules.transnerf import LatentCodeEncoder
from nerf.utils import calc_mse, calc_psnr, sample_images_at_mc_locs
from .renderer_base import RendererBase
from modules.util import freeze, unfreeze
from torch.optim import Adam, SGD
from tqdm import trange
from .util import get_name_idx_mapping


class TransRenderer(RendererBase):
    """
    Renderer of pixel nerf
    """

    def __init__(
            self,
            sequences: typing.List[str],
            image_size: Tuple[int, int],
            n_pts_per_ray: int,
            n_pts_per_ray_fine: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            function_config: dict,
            encoder_layers: int,
            mask_thr: float = 0.0,
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
        """

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

        # Encoder
        self.encoder = LatentCodeEncoder(
            n_layers=encoder_layers,
            shape_code_dim=function_config.get('shape_code_dim'),
            color_code_dim=function_config.get('color_code_dim'),
            single_input=False,
            reduction=True,
            pretrained=True
        )

        # Transform Embeddings
        if len(set(sequences)) != len(sequences):
            raise ValueError(f'The are duplicates among sequences.')
        self._set_transform_embedding(
            name_idx_mapping=get_name_idx_mapping(sequences),
            with_translation=function_config.get('with_translation')
        )

    def _set_transform_embedding(self, name_idx_mapping: dict, with_translation: bool):
        assert isinstance(name_idx_mapping, dict), f'name_idx_mapping must be a dict, ' \
                                                   f'but given {type(name_idx_mapping)}.'
        self.name_idx_mapping = name_idx_mapping
        self.embedding = torch.nn.Embedding(
            len(self.name_idx_mapping),
            embedding_dim=(6 if with_translation else 3)
        )
        # initialize parameters in transform embedding to zero
        for param in self.embedding.parameters():
            param.data.zero_()

    def _get_sequence_index(self, sequences: typing.List[str], device):
        # check if all sequences are same
        name = list(set(sequences))
        assert len(name) == 1, f'Not all frame comes from the same sequence.'

        idx = self.name_idx_mapping[name[0]]
        idx_tensor = torch.tensor([idx], dtype=torch.int, device=device)

        return idx_tensor

    def _get_transform_code(self, sequences: typing.List[str], device):
        idx_tensor = self._get_sequence_index(sequences, device)
        # (1, d)
        return self.embedding(idx_tensor)

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor,
            shape_latent_code: torch.Tensor,
            color_latent_code: torch.Tensor,
            transform_code: torch.Tensor,
            chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            shape_latent_code: (1, d)
            color_latent_code:
            transform_code: (1, 3 or 6)
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

                transform_code=transform_code,
                shape_latent_code=shape_latent_code,
                color_latent_code=color_latent_code,

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

    def adapt(self,
              source_camera: CamerasBase,
              source_image: torch.Tensor,
              source_mask: torch.Tensor,
              reset: bool,
              max_iters: int = 100,
              learning_rate: float = 1.0e-4,
              transform_reg_coef: float = 1.0e-3
              ):
        """
        Find the proper transform embedding
        :param source_camera:
        :param source_image:
        :param source_mask:
        :param reset:
        :param max_iters:
        :param learning_rate:
        :param transform_reg_coef:
        :return:
        """
        # freeze modules
        freeze(self)

        # set new embedding module
        if reset:
            self._set_transform_embedding(
                name_idx_mapping={'eval': 0},
                with_translation=(self.embedding.embedding_dim == 6)
            )
            self.embedding.to(source_camera.device)
            self.embedding.train()
        else:
            # initialize parameters in transform embedding to zero
            for param in self.embedding.parameters():
                param.data.zero_()
            unfreeze(self.embedding)

        # use adam as optimizer
        optimizer = SGD(self.embedding.parameters(), lr=learning_rate, momentum=0.99)

        # begin adaptation
        pbar = trange(max_iters)
        for iter_idx in pbar:
            pbar.set_description(f'Iter {iter_idx + 1}')

            _, metrics = self(
                source_camera,
                source_image,
                source_mask,
                source_image,
                ['eval']
            )

            mse_coarse, mse_fine, transform_reg = \
                [metrics[idx] for idx in ['mse_coarse', 'mse_fine', 'transform_reg']]
            loss = mse_coarse + mse_fine + transform_reg_coef * transform_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix_str("loss={:05f}".format(loss.item()))

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor,
            source_image: torch.Tensor,
            sequence_name: typing.List[str],
            enable_transform: bool = True
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
            source_image: A batch of corresponding ground truth images of shape
                (1, 3, h, w)
            target_camera:
            target_image: can be None
            target_mask: probability of one pixel can be foreground or mask to indicate valid area
            source_image:
            sequence_name: sequence name of target images
            enable_transform:
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

        # get shape and color code
        if source_image is None:
            source_image = target_image
        shape_latent_code, color_latent_code = self.encoder(source_image)  # (1, d)

        # get transform code
        transform_code = self._get_transform_code(sequence_name, target_camera.device) if enable_transform else None

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
                shape_latent_code,
                color_latent_code,
                transform_code,
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

        if self.training:
            # Calc transformation regularization
            metrics['transform_reg'] = calc_l2_reg(transform_code)

            # Calc latent code regularization
            metrics['shape_code_reg'] = calc_l2_reg(shape_latent_code)
            metrics['color_code_reg'] = calc_l2_reg(color_latent_code)

        return out, metrics
