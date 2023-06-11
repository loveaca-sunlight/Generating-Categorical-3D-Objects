import os.path
import typing
import warnings

import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from torch.optim.lr_scheduler import MultiStepLR
from .util import mask_image
from renderers import SepRenderer
from tqdm import tqdm


class TestModel(pl.LightningModule):
    def __init__(self, cfg, sequences: typing.List[str]):
        super(TestModel, self).__init__()

        # manual optimization
        self.automatic_optimization = False

        self.cfg = cfg

        self.model = SepRenderer(
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

    def forward(self,
                target_camera: CamerasBase,
                target_image: torch.Tensor,
                target_mask: torch.Tensor,
                shape_sequence_name: str,
                color_sequence_name: str = None,
                ):
        return self.model(
            target_camera,
            target_image,
            target_mask,
            shape_sequence_name,
            color_sequence_name
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

        # get output and metrics
        output, metrics = self(
            cameras,
            images,
            fg_probability if self._mask_image else mask_crop,
            sequence_name[0],
            None
        )

        # get metrics
        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
        shape_code_reg = metrics['shape_code_reg']
        color_code_reg = metrics['color_code_reg']

        # log
        for key in ['mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine', 'shape_code_reg', 'color_code_reg']:
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
        loss = mse_coarse + mse_fine + self.cfg.loss.latent_code_reg_coef * (shape_code_reg + color_code_reg)

        # optimizers
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # backward
        self.manual_backward(loss)

        optimizer.step()

        # check grad and write to csv
        items = []
        if batch_idx == 0:
            items.append(
                ','.join(['step', 'name', 'min', 'max', 'median', 'mean']) + '\n'
            )
        for name, param in self.model.named_parameters():
            grad_norm = param.grad.abs()
            # 'step', 'name', 'min', 'max', 'median', 'mean'
            items.append(
                f'{self.global_step},{name},{grad_norm.min()},{grad_norm.max()},{grad_norm.median()},'
                f'{grad_norm.mean()}\n'
            )
        with open(os.path.join('outputs', f'epoch_{self.current_epoch}.csv'), 'a') as f:
            f.writelines(items)
