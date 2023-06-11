import warnings

import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from .util import mask_image, choose_views
from renderers import SceneRenderer
from .registry import MODELS


@MODELS.register_module(name='test_model')
class TestModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(TestModel, self).__init__()

        self.cfg = cfg

        self.model = SceneRenderer(
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

        self._mask_image = self.cfg.model.get('mask_image', False)
        self._seq_name = cfg.dataset.seq_name

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
        )

        # The learning rate scheduling
        lr_scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.cfg.optimizer.step_size,
            gamma=self.cfg.optimizer.gamma,
        )
        print(f'Using StepLR with step_size={self.cfg.optimizer.step_size}, gamma={self.cfg.optimizer.gamma}.')
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_mask: torch.Tensor
    ):
        return self.model(
            target_camera=target_camera,
            target_image=target_image,
            target_mask=target_mask
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

        # check sequence
        assert len(set(sequence_name)) == 1, f'Not all frames come from the same sequence: {sequence_name}.'
        assert sequence_name[0] == self._seq_name

        # choose target view
        tgt_ids = list(range(1, images.shape[0]))
        tgt_imgs, tgt_cams, tgt_masks = choose_views(tgt_ids, images, cameras, mask_crop)

        # get output and metrics
        output, metrics = self(
            target_camera=tgt_cams,
            target_image=tgt_imgs,
            target_mask=tgt_masks
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
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # get output and metrics
        val_nerf_out, val_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_mask=mask_crop
        )

        # write images
        if batch_idx <= 1:  # log first two samples
            for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
                img = val_nerf_out[k]
                fg_prob = fg_probability.permute(0, 2, 3, 1)
                img = fg_prob * img + (1.0 - fg_prob) * 1.0
                writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            for k in ['depth_coarse', 'depth_fine']:
                depth = val_nerf_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val_{batch_idx}/{k}', depth, global_step=self.current_epoch + 1,
                                  dataformats='NHWC')

        # log validation accuracy
        val_acc = self.current_epoch / 100.0
        self.log('val/val_acc', val_acc, sync_dist=True, batch_size=1)
