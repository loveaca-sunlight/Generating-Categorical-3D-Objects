import imp
import warnings
import os.path as osp

import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import os
import trimesh

from renderers import NeusRenderer
from .util import mask_image
from .registry import MODELS
from torchvision import utils as vutils

@MODELS.register_module(name='neus_model')
class NeusModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(NeusModel, self).__init__()
        torch.cuda.set_device(0)
        self.cfg = cfg

        self.model = NeusRenderer(
            image_size=(self.cfg.model.height, self.cfg.model.width),
            n_pts_per_ray=self.cfg.model.n_pts_per_ray,
            n_rays_per_image=self.cfg.model.n_rays_per_image,
            up_sample_steps=self.cfg.model.up_sample_steps,
            n_pts_importance=self.cfg.model.n_pts_importance,
            min_depth=self.cfg.model.min_depth,
            max_depth=self.cfg.model.max_depth,
            stratified=self.cfg.model.stratified,
            stratified_test=self.cfg.model.stratified_test,
            chunk_size_test=self.cfg.model.chunk_size_test,
            density_noise_std=self.cfg.model.density_noise_std,

            function_config=self.cfg.model.implicit_function,
            raymarcher_config=self.cfg.model.raymarcher,

            mask_thr=self.cfg.dataset.mask_thr,
            edge_thr=self.cfg.dataset.edge_thr,
        )

        self._mask_image = self.cfg.model.mask_image

        self._vali_mesh = self.cfg.vali_mesh
        self.vali_near = self.cfg.vali_near
        self.vali_far = self.cfg.vali_far

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
            target_mask: torch.Tensor,
            valida: bool,
    ):
        return self.model(
            target_camera=target_camera,
            target_image=target_image,
            target_fg_probability=target_fg_probability,
            target_mask=target_mask,
            valida = valida
        )

    def training_step(self, batch_data, batch_idx):
        self.model.train()
        torch.set_grad_enabled(True)

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
            target_mask=fg_probability if self._mask_image else mask_crop,
            valida = True
        )

        # get metrics
        mse_fine = metrics['mse_fine']
        mask_loss = metrics['mask_loss']
        eikonal_loss = metrics['gradient_error']

        # log
        for key in ['mse_fine','psnr_fine', 'mask_loss','gradient_error']:
            self.log(f'train/{key}', metrics[key], on_step=True, on_epoch=False, logger=True)

        if (batch_idx == 64) and ((self.current_epoch + 1) % 5 == 0):  # log every 5 epochs
            writer = self.logger.experiment
            for k in ['rgb_fine', 'rgb_gt']:
                img = output[k]
                writer.add_image(f'train/{k}', torch.clamp(img, min=0.0, max=1.0),
                                 global_step=self.current_epoch + 1, dataformats='HWC')
            for k in ['depth_fine']:
                depth = output[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_image(f'train/{k}', depth, global_step=self.current_epoch + 1, dataformats='HWC')

        # loss
        loss = mse_fine + self.cfg.loss.weights_mask * (mask_loss) + self.cfg.loss.weights_igr * (eikonal_loss)

        self.log(f'train/loss', loss, on_step=True, on_epoch=False, logger=True)

        return loss

    def validation_step(self, batch_data, batch_idx):
        if not os.path.exists(self.cfg.base_log_dir):
            os.mkdir(self.cfg.base_log_dir)
        # get tensorboard writer
        #self.model.train()
        #torch.set_grad_enabled(True)
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
            target_mask=fg_probability if self._mask_image else mask_crop,
            valida = False
        )

        # write images
        if batch_idx <= 1:  # log first two samples
            for k in ['rgb_fine', 'rgb_gt']:
                img = val_nerf_out[k]
                writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            for k in ['depth_fine']:
                depth = val_nerf_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val_{batch_idx}/{k}', depth, global_step=self.current_epoch + 1,
                                  dataformats='NHWC')

        # log validation accuracy
        val_acc = (val_metrics['psnr_fine']) / 2.0
        self.log('val/val_acc', val_acc, sync_dist=True, batch_size=1)

        if self._vali_mesh:
            resolution= 64#300
            threshold=0.0
            bound_min = torch.tensor(self.cfg.vali_near, dtype=torch.float32)
            bound_max = torch.tensor(self.cfg.vali_far, dtype=torch.float32)

            vertices, triangles =\
                self.model.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
            os.makedirs(os.path.join(self.cfg.base_log_dir, 'meshes'), exist_ok=True)

            #if world_space:
            #    vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.cfg.base_log_dir, 'meshes', '{:0>8d}.ply'.format(self.current_epoch)))
        
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.current_epoch / self.anneal_end])