import os.path
import random
import typing

import mmcv
import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from timm.scheduler.step_lr import StepLRScheduler
from torch.optim import Optimizer

from renderers import BaseRenderer
from .registry import MODELS
from .util import mask_image, choose_views, augment_source_images


@MODELS.register_module(name='base_nerf_model')
class BaseNeRFModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(BaseNeRFModel, self).__init__()

        self.cfg = cfg

        self.model = BaseRenderer(
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

            mask_thr=self.cfg.dataset.mask_thr,

            weighted_mse=self.cfg.model.get('weighted_mse', False)
        )

        self._mask_image = self.cfg.model.mask_image

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        """
        Override the default behavior of zero_grad
        """
        # It is very important to set set_to_none to True to avoid wrong gradient information caused by zero gradiant.
        optimizer.zero_grad(set_to_none=True)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self):
        # Configure optimizer
        param_groups = [
            {'params': self.model.encoder_parameters(),
             'lr': self.cfg.optimizer.encoder_lr},
            {'params': self.model.hyper_parameters(),
             'lr': self.cfg.optimizer.hyper_lr},
            {'params': self.model.rest_parameters(),
             'lr': self.cfg.optimizer.lr},
        ]
        optimizer = torch.optim.Adam(
            param_groups
        )

        # Configure scheduler
        scheduler = StepLRScheduler(
            optimizer=optimizer,
            **self.cfg.scheduler
        )
        print('Using StepLR with ' + ', '.join([f'{k}={v}' for k, v in self.cfg.scheduler.items()]))
        return [optimizer], [scheduler]

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            source_image: torch.Tensor,
            sequence_name: str,
            enable_deformation: bool = True,
            enable_specular: bool = True,
            shape_code: torch.Tensor = None,
            color_code: torch.Tensor = None,
            transform_code: torch.Tensor = None,
            produce_parameters: bool = True
    ):
        return self.model(
            target_camera=target_camera,
            target_image=target_image,
            target_fg_probability=target_fg_probability,
            target_mask=target_mask,
            source_image=source_image,
            sequence_name=sequence_name,
            enable_deformation=enable_deformation,
            enable_specular=enable_specular,
            shape_code=shape_code,
            color_code=color_code,
            transform_code=transform_code,
            produce_parameters=produce_parameters
        )

    def prepare_input(self, src_img: torch.Tensor, src_fg: torch.Tensor):
        fg = src_fg.clone()
        fg[fg < self.cfg.dataset.mask_thr] = 0.0
        inputs = torch.cat([src_img, fg], dim=1)  # (n, 4, h, w)
        return inputs

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
        src_img, src_fg = choose_views(source_ids, images, fg_probability)

        # cat input (mask first)
        source_images = self.prepare_input(src_img, src_fg)

        # data augment
        source_images = augment_source_images(source_images)

        # get output and metrics
        output, metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=(fg_probability if self._mask_image else mask_crop),
            source_image=source_images,
            sequence_name=sequence_name[0],
            enable_deformation=True,
            enable_specular=True
        )

        # get metrics
        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
        weights_reg_coarse, weights_reg_fine = metrics['weights_reg_coarse'], metrics['weights_reg_fine']
        # weights_norm = metrics['weights_norm']

        # log
        for key in ['mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine', 'weights_reg_coarse',
                    'weights_reg_fine', 'weights_norm']:
            self.log(f'train/{key}', metrics[key], on_step=True, on_epoch=False, logger=True)

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
        loss = mse_coarse + mse_fine \
               + self.cfg.loss.weights_reg_coef * (weights_reg_coarse + weights_reg_fine)
               # + self.cfg.loss.weights_norm_coef * weights_norm

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
        src_img, src_fg = choose_views(source_ids, images, fg_probability)

        # cat input (mask first)
        source_images = self.prepare_input(src_img, src_fg)

        # data augment
        source_images = augment_source_images(source_images)

        # get output and metrics
        val_nerf_out, val_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=(fg_probability if self._mask_image else mask_crop),
            source_image=source_images,
            sequence_name=sequence_name[0],
            enable_deformation=True,
            enable_specular=True
        )

        # write images
        if batch_idx <= 1:  # log first two samples
            # log augmented source images
            writer.add_images(f'val_{batch_idx}/src_imgs', torch.clamp(source_images[:, : 3, :, :], min=0.0, max=1.0),
                              global_step=self.current_epoch + 1, dataformats='NCHW')
            writer.add_images(f'val_{batch_idx}/src_msks', source_images[:, 3: 4, :, :],
                              global_step=self.current_epoch + 1, dataformats='NCHW')

            for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
                img = val_nerf_out[k]
                writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            for k in ['depth_coarse', 'depth_fine']:
                depth = val_nerf_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val_{batch_idx}/{k}', depth, global_step=self.current_epoch + 1,
                                  dataformats='NHWC')

        # log validation accuracy
        val_acc = val_metrics['psnr_fine']  # only consider fine pass
        self.log('val/val_acc', val_acc, sync_dist=True, batch_size=1)
