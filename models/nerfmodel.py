import warnings
import os.path as osp

import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from torch.optim.lr_scheduler import MultiStepLR

from renderers import NerfRenderer
from .util import mask_image
from .registry import MODELS


@MODELS.register_module(name='nerf_model')
class NerfModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(NerfModel, self).__init__()

        self.cfg = cfg

        self.model = NerfRenderer(
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

            mask_thr=self.cfg.dataset.mask_thr,
        )

        self._mask_image = self.cfg.model.mask_image

        # save key points
        self._seq_name = cfg.dataset.seq_name

    def on_train_start(self) -> None:
        """
        Save key point file 在一开始保存手动标的那个key
        :return:
        """
        kp_path = osp.join(self.cfg.dataset.root, self.cfg.dataset.category, 'keypoints/', f'{self._seq_name}.pt')
        # save key point file if exist
        if osp.exists(kp_path):
            key_points = torch.load(kp_path, map_location='cpu')
            print(key_points)
            torch.save(key_points, osp.join(self.trainer.default_root_dir, 'keypoints.pt'))

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
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

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor
    ):
        return self.model(
            target_camera=target_camera,
            target_image=target_image,
            target_fg_probability=target_fg_probability,
            target_mask=target_mask
        )

    def training_step(self, batch_data, batch_idx):
        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop
        sequence_name = list(set(batch_data.sequence_name))

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # check sequence
        assert len(sequence_name) == 1, 'Not all frames come from the same sequence.'
        assert sequence_name[0] == self._seq_name

        # get output and metrics
        output, metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=fg_probability if self._mask_image else mask_crop
        )

        # get metrics
        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
        weights_reg_coarse, weights_reg_fine = metrics['weights_reg_coarse'], metrics['weights_reg_fine']

        # log
        for key in ['mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine', 'weights_reg_coarse', 'weights_reg_fine']:
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
        loss = mse_coarse + mse_fine + self.cfg.loss.weights_reg_coef * (weights_reg_coarse + weights_reg_fine)

        return loss

    def validation_step(self, batch_data, batch_idx):
        # get tensorboard writer
        writer = self.logger.experiment

        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # get output and metrics
        val_nerf_out, val_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=fg_probability if self._mask_image else mask_crop
        )

        # write images
        if batch_idx <= 1:  # log first two samples
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
        val_acc = (val_metrics['psnr_fine'] + val_metrics['psnr_coarse']) / 2.0
        self.log('val/val_acc', val_acc, sync_dist=True, batch_size=1)
