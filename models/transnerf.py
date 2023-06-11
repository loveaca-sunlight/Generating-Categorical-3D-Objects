import typing
import warnings

import pytorch_lightning as pl
import torch

from renderers import TransRenderer
from torch.optim.lr_scheduler import MultiStepLR
from .util import mask_image
from pytorch3d.renderer.cameras import CamerasBase
from tqdm import tqdm
from .registry import MODELS


@MODELS.register_module(name='transnerf')
class TransNerfModel(pl.LightningModule):
    def __init__(self, cfg, sequences: typing.List[str]):
        super(TransNerfModel, self).__init__()

        self.cfg = cfg

        self.model = TransRenderer(
            sequences=sequences,

            image_size=(self.cfg.model.height, self.cfg.model.width),
            n_pts_per_ray=self.cfg.model.n_pts_per_ray,
            n_pts_per_ray_fine=self.cfg.model.n_pts_per_ray_fine,
            n_rays_per_image=self.cfg.model.n_rays_per_image,
            min_depth=self.cfg.model.min_depth,
            max_depth=self.cfg.model.max_depth,
            stratified=self.cfg.model.stratified,
            stratified_test=self.cfg.model.stratified_test,
            chunk_size_test=self.cfg.model.chunk_size_test,
            density_noise_std=self.cfg.model.density_noise_std,

            function_config=self.cfg.model.implicit_function,
            encoder_layers=self.cfg.model.encoder_layers,

            mask_thr=self.cfg.dataset.mask_thr
        )

        self._mask_image = self.cfg.model.mask_image

    def configure_optimizers(self):
        # Initialize the optimizer.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
        )

        if self.cfg.optimizer.gamma < 1.0:
            if self.cfg.optimizer.gamma < 0.5:
                warnings.warn(f'The gamma parameter is too small: {self.cfg.optimizer.gamma}.')
            # The learning rate scheduling
            lr_scheduler = MultiStepLR(
                optimizer=optimizer,
                milestones=self.cfg.optimizer.milestones,
                gamma=self.cfg.optimizer.gamma,
            )
            print(f'Using MultiStepLR with gamma = {self.cfg.optimizer.gamma}, '
                  f'milestones = {self.cfg.optimizer.milestones}.')
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer

    def adapt(self,
              source_camera: CamerasBase,
              source_image: torch.Tensor,
              source_mask: torch.Tensor,
              reset: bool,
              max_iters: int = 100,
              learning_rate: float = 1.0e-4,
              transform_reg_coef: float = 1.0e-3
              ):
        self.model.adapt(
            source_camera,
            source_image,
            source_mask,
            reset,
            max_iters,
            learning_rate,
            transform_reg_coef
        )

    def forward(self,
                target_camera: CamerasBase,
                target_image: torch.Tensor,
                target_mask: torch.Tensor,
                source_image: torch.Tensor,
                sequence_name: typing.List[str],
                **kwargs
                ):
        return self.model(
            target_camera,
            target_image,
            target_mask,
            source_image,
            sequence_name,
            **kwargs
        )

    def training_step(self, batch_data, batch_idx):
        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        sequence_name = batch_data.sequence_name
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # make sure all frames come from the same sequence
        assert len(set(sequence_name)) == 1, 'Not all frames come from the same sequence.'

        # select source view
        if self.cfg.model.single_source_view:
            _, source_image, _ = choose_source_view(cameras, images)
        else:
            source_image = None

        # get output and metrics
        output, metrics = self(
            cameras,
            images,
            fg_probability if self._mask_image else mask_crop,
            source_image,
            sequence_name
        )

        # get metrics
        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
        transform_reg = metrics['transform_reg']
        shape_code_reg = metrics['shape_code_reg']
        color_code_reg = metrics['color_code_reg']

        # log
        for key in ['mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine', 'transform_reg', 'shape_code_reg',
                    'color_code_reg']:
            self.log(f'train/{key}', metrics[key], on_step=True, on_epoch=False, logger=True)

        if batch_idx == 64:
            writer = self.logger.experiment
            for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
                img = output[k]
                writer.add_image(f'train/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='HWC')
            for k in ['depth_coarse', 'depth_fine']:
                depth = output[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_image(f'train/{k}', depth, global_step=self.current_epoch + 1, dataformats='HWC')

        # loss
        loss = mse_coarse + mse_fine + self.cfg.loss.latent_reg_coef * (shape_code_reg + color_code_reg) + \
               self.cfg.loss.transform_reg_coef * transform_reg

        return loss

    def validation_step(self, batch_data, batch_idx):
        # get tensorboard writer
        writer = self.logger.experiment

        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        sequence_name = batch_data.sequence_name
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # make sure all frames come from the same sequence
        assert len(set(sequence_name)) == 1, 'Not all frames come from the same sequence.'

        # chose source view
        source_camera, source_image, source_idx = choose_source_view(cameras, images)

        # get output and metrics
        val_nerf_out, val_metrics = self(
            cameras,
            images,
            fg_probability if self._mask_image else mask_crop,
            source_image,
            sequence_name
        )

        # write images
        if batch_idx == 0:
            for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
                img = val_nerf_out[k]

                # mark source image
                src_img = img[source_idx]
                fg_probability = torch.all(src_img == 0.0, dim=-1, keepdim=True).float()  # (n, h, w, 1)
                src_img = src_img + fg_probability * 0.5
                img[source_idx] = src_img

                writer.add_images(f'val/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            for k in ['depth_coarse', 'depth_fine']:
                depth = val_nerf_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val/{k}', depth, global_step=self.current_epoch + 1, dataformats='NHWC')

        # log current steps
        self.log('val/epochs', self.current_epoch, sync_dist=True)
