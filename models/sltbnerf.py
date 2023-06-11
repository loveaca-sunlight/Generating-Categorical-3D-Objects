import os.path as osp
import typing
import warnings

import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from torch.optim.lr_scheduler import MultiStepLR

from modules.multiplex_new import TransformMultiplex
from renderers import SilhouetteBackRenderer
from .registry import MODELS
from .util import mask_image, select_key_points
from torch.optim import Optimizer


@MODELS.register_module(name='sltb_nerf_model')
class SltbNerfModel(pl.LightningModule):
    def __init__(self, cfg, sequences: typing.List[str], resume: bool = False):
        super(SltbNerfModel, self).__init__()

        self.cfg = cfg

        # load template key points
        template_weights_path = self.cfg.model.template_weights
        key_point_file = osp.join(osp.dirname(template_weights_path), 'keypoints.pt')
        key_points = torch.load(key_point_file, map_location='cpu')

        self._match_keys = self.cfg.model.match_keys
        template_key_points = torch.stack(
            [key_points[k] for k in self._match_keys],
            dim=0
        )  # (p, 3)

        # # load base transforms
        # if not resume:
        #     transform_path = osp.join(cfg.dataset.root, cfg.dataset.category, 'keypoints/transforms/init_transforms.pt')
        #     if osp.exists(transform_path):
        #         base_transforms = torch.load(transform_path, map_location='cpu')
        #         print(f'Base transforms are loaded from {transform_path}.')
        #     else:
        #         base_transforms = None
        #         print('Base transforms are not found.')
        # else:
        #     base_transforms = None

        self.model = SilhouetteBackRenderer(
            sequences=sequences,

            image_size=(self.cfg.model.height, self.cfg.model.width),
            n_pts_per_ray=self.cfg.model.n_pts_per_ray,
            # n_pts_per_ray_fine=self.cfg.model.n_pts_per_ray_fine,
            n_rays_per_image=self.cfg.model.n_rays_per_image,
            min_depth=self.cfg.model.min_depth,
            max_depth=self.cfg.model.max_depth,
            stratified=self.cfg.model.stratified,
            stratified_test=self.cfg.model.stratified_test,
            chunk_size_test=self.cfg.model.chunk_size_test,
            density_noise_std=self.cfg.model.density_noise_std,

            function_config=self.cfg.model.implicit_function,

            mask_thr=self.cfg.dataset.mask_thr,

            n_transform_ways=self.cfg.model.n_transform_ways,

            template_key_points=template_key_points,
            match_tol=self.cfg.model.match_tol,
        )

        self._mask_image = self.cfg.model.mask_image

        # load template weights or whole weights
        if not resume:
            pretrained_path = self.cfg.model.get('pretrained_weights', None)
            assert pretrained_path is None, 'The keyword pretrained_path is deprecated.'  # Note, only load template weights
            if pretrained_path is None:
                # load template weights
                state_dict = torch.load(template_weights_path, map_location='cpu')['state_dict']
                self.model.load_template_weights(state_dict)
                print(f'Template weights are loaded from {template_weights_path}.')
            else:
                # load all weights
                state_dict = torch.load(pretrained_path, map_location='cpu')['state_dict']
                self.load_state_dict(state_dict)
                print(f'Model weights are loaded from {pretrained_path}.')

        # freeze template weights
        self.model.freeze_template_weights()
        print('Template weights are frozen. ')

        # resume information
        print(f'Resume: {resume}.')

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.SGD(
            self.model.transform_parameters(),
            lr=self.cfg.optimizer.lr,
            momentum=self.cfg.optimizer.momentum
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
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def on_train_start(self) -> None:
        """
        On the start of fitting, save name_idx_mapping of latent embedding
        :return:
        """
        sequence_names = self.model.transforms.sequence_names
        fp = osp.join(self.trainer.default_root_dir, 'sequence_names.txt')
        with open(fp, 'w') as f:
            f.writelines('\n'.join(sequence_names))

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        """
        Override the default behavior of zero_grad
        """
        # It is very important to set set_to_none to True to avoid wrong gradient information caused by zero gradiant.
        optimizer.zero_grad(set_to_none=True)

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            key_points: torch.Tensor,
            key_ids: typing.List[int],
            sequence_name: str,
    ):
        return self.model(
            target_camera=target_camera,
            target_image=target_image,
            target_fg_probability=target_fg_probability,
            target_mask=target_mask,
            key_points=key_points,
            key_ids=key_ids,
            sequence_name=sequence_name,
        )

    def training_step(self, batch_data, batch_idx):
        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        sequence_name = batch_data.sequence_name
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop
        kp_dict = batch_data.meta['key_points']  # dict(str, (n, 3))

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # make sure all frames come from the same sequence
        assert len(set(sequence_name)) == 1, 'Not all frames come from the same sequence.'

        # get key points
        key_points, sel_ids = select_key_points(kp_dict, self._match_keys)
        key_points = key_points.type_as(images)  # (n, p, 3)

        # get output and metrics
        outputs, multiplex_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=fg_probability if self._mask_image else mask_crop,
            key_points=key_points,
            key_ids=sel_ids,
            sequence_name=sequence_name[0],
        )

        # get metrics
        weight_reg_list = []
        match_loss_list = []
        for mp_idx, (output, metrics) in enumerate(zip(outputs, multiplex_metrics)):
            weights_reg_coarse = metrics['weights_reg_coarse']
            match_loss = metrics['match_loss']

            # log
            for key in ['weights_reg_coarse', 'match_loss']:
                self.log(f'train_mp{mp_idx}/{key}', metrics[key], on_step=True, on_epoch=False, logger=True)

            # append to list
            weight_reg_list.append(weights_reg_coarse)
            match_loss_list.append(match_loss)

        # stack losses
        weight_reg_losses = torch.stack(weight_reg_list)  # (m,)
        match_losses = torch.stack(match_loss_list)

        # update score
        with torch.no_grad():
            multiplex_loss = weight_reg_losses.detach() + self.cfg.loss.val_match_loss_coef * match_losses.detach()
            scores = TransformMultiplex.losses_to_scores(multiplex_loss)
            self.model.update_multiplex_score(sequence_name[0], scores)

        # loss for transforms
        losses = weight_reg_losses + self.cfg.loss.match_loss_coef * match_losses
        transform_loss = losses.mean()

        return transform_loss

    def validation_step(self, batch_data, batch_idx):
        # get tensorboard writer
        writer = self.logger.experiment

        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        sequence_name = batch_data.sequence_name
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop
        kp_dict = batch_data.meta['key_points']  # dict(str, (n, 3))

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # make sure all frames come from the same sequence
        assert len(set(sequence_name)) == 1, 'Not all frames come from the same sequence.'

        # get key points
        key_points, sel_ids = select_key_points(kp_dict, self._match_keys)
        key_points = key_points.type_as(images)  # (n, p, 3)

        # get output and metrics
        val_nerf_out, val_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=fg_probability if self._mask_image else mask_crop,
            key_points=key_points,
            key_ids=sel_ids,
            sequence_name=sequence_name[0],
        )

        # write images
        if batch_idx <= 1:  # log first two samples
            for k in ['rgb_gt']:
                img = val_nerf_out[k]
                writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            for k in ['depth_coarse']:
                depth = val_nerf_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val_{batch_idx}/{k}', depth, global_step=self.current_epoch + 1,
                                  dataformats='NHWC')
            # log sequence name
            writer.add_text(f'val_{batch_idx}/seq_name', sequence_name[0], global_step=self.current_epoch + 1)

            # log losses
            for k in ('weights_reg_coarse', 'match_loss'):
                writer.add_scalar(f'val_{batch_idx}/{k}', val_metrics[k], global_step=self.current_epoch + 1)

        # log validation accuracy
        val_acc = val_metrics['weights_reg_coarse'] + self.cfg.loss.match_loss_coef * val_metrics['match_loss']
        self.log('val/val_loss', val_acc, sync_dist=True, batch_size=1)
