import random
import warnings

import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from torch.optim.lr_scheduler import MultiStepLR

from renderers import PixelRenderer
from .registry import MODELS
from .util import mask_image, choose_views, mask_images_background


@MODELS.register_module(name='pixel_nerf_model')
class PixelNerfModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(PixelNerfModel, self).__init__()

        self.cfg = cfg

        self.model = PixelRenderer(
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
            encoder_config=self.cfg.model.encoder_config,

            mask_thr=self.cfg.dataset.mask_thr
        )

        self._mask_image = self.cfg.model.mask_image

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr
        )

        if self.cfg.optimizer.gamma < 1.0:
            if self.cfg.optimizer.gamma < 0.5:
                warnings.warn(f'The gamma parameter is too small: {self.cfg.optimizer.gamma}.')
            # The learning rate scheduling
            lr_scheduler = MultiStepLR(
                optimizer=optimizer,
                milestones=self.cfg.optimizer.milestones,
                gamma=self.cfg.optimizer.gamma
            )
            print(f'Using MultiStepLR with gamma = {self.cfg.optimizer.gamma}, '
                  f'milestones = {self.cfg.optimizer.milestones}.')
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor,
            source_image: torch.Tensor,
            source_camera: CamerasBase,
            n_sources: int
    ):
        return self.model(
            target_camera=target_camera,
            target_image=target_image,
            target_mask=target_mask,
            source_image=source_image,
            source_camera=source_camera,
            n_sources=n_sources
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

        # choose source views
        n_sources = random.choice(self.cfg.model.n_sources)
        source_ids = random.sample(list(range(images.shape[0])), n_sources)
        source_images, source_cameras = choose_views(source_ids, images, cameras)

        # get output and metrics
        output, metrics = self(
            target_camera=cameras,
            target_image=images,
            target_mask=(fg_probability if self._mask_image else mask_crop),
            source_image=source_images,
            source_camera=source_cameras,
            n_sources=n_sources
        )

        # get metrics
        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']

        # log
        for key in ['mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine']:
            self.log(f'train/{key}', metrics[key], on_step=True, on_epoch=False, logger=True, batch_size=1)

        if (batch_idx == 64) and ((self.current_epoch + 1) % 5 == 0):  # log every 5 epochs
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
        loss = mse_coarse + mse_fine

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

        # choose source views
        n_sources = random.choice(self.cfg.model.n_sources)
        source_ids = random.sample(list(range(images.shape[0])), n_sources)
        source_images, source_cameras = choose_views(source_ids, images, cameras)

        # get output and metrics
        val_nerf_out, val_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_mask=(fg_probability if self._mask_image else mask_crop),
            source_image=source_images,
            source_camera=source_cameras,
            n_sources=n_sources
        )

        # write images
        if batch_idx <= 1:  # log first two samples
            for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
                img = val_nerf_out[k]
                # mask source images
                mask_images_background(img, source_ids)
                writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            for k in ['depth_coarse', 'depth_fine']:
                depth = val_nerf_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val_{batch_idx}/{k}', depth, global_step=self.current_epoch + 1,
                                  dataformats='NHWC')

        # log validation accuracy
        val_acc = val_metrics['psnr_fine']
        self.log('val/val_acc', val_acc, sync_dist=True, batch_size=1)
