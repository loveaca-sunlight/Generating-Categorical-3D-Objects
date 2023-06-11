import os.path
import random
import typing

import mmcv
import pytorch_lightning as pl
import torch
from pytorch3d.renderer.cameras import CamerasBase
from timm.scheduler.step_lr import StepLRScheduler
from torch.optim import Optimizer
import trimesh
import numpy as np
from PIL import Image 
from io import BytesIO
from torchvision import transforms

from renderers import StyleNeusRenderer
from .registry import MODELS
from .util import mask_image, choose_views, augment_source_images, find_files, load_image, resize_image, load_mask, get_bbox_from_mask, get_clamp_bbox, crop_around_box

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(256/480, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
])

@MODELS.register_module(name='style_neus_model')
class StyleNeusModel(pl.LightningModule):
    def __init__(self, cfg, sequences: typing.List[str], load_pretrained_weights=True, style_data='train', **kwargs):
        super(StyleNeusModel, self).__init__()

        self.cfg = cfg

        # try to load sequence data from pre-trained slt model
        if load_pretrained_weights:
            seq_fp = os.path.join(os.path.dirname(self.cfg.model.best_transforms), 'sequence_names.txt')
            if os.path.exists(seq_fp):
                sequences = mmcv.list_from_file(seq_fp)
                print(f'Sequence data is found at {seq_fp}.')

        # load style image TMP
        style_img_path = {}
        for name in ['train','validation','test']:
            style_img_path[name] = os.path.join(self.cfg.dataset.style_path, name)

        #风格图片路径
        self.style_img_files = find_files(style_img_path[style_data], exts=['*.png', '*.jpg'])

        '''style = load_image(self.cfg.dataset.style_path)

        style, scale, mask_crop = resize_image(style, shape=[3,128,128])

        if self.cfg.dataset.mask_path != 'None':
            mask = load_mask(self.cfg.dataset.mask_path)
            bbox_xywh = torch.tensor(get_bbox_from_mask(mask, 0.4))

            clamp_bbox_xyxy = get_clamp_bbox(bbox_xywh, 0.3)
            mask = crop_around_box(mask, clamp_bbox_xyxy, self.cfg.dataset.mask_path)

            fg_probability, _, _ = resize_image(mask, shape=[3,128,128], mode="nearest")

            style = crop_around_box(style, clamp_bbox_xyxy, self.cfg.dataset.style_path)
            style, scale, mask_crop = resize_image(style, shape=[3,128,128])

            style = mask_image(style, fg_probability, self.cfg.dataset.mask_thr)

            style = self.prepare_input(style[None,], fg_probability[None,])
        else:
            style = style[None,]

        self.style_image = style.cuda()'''

        self.model = StyleNeusRenderer(
            sequences=sequences,

            image_size=(self.cfg.model.height, self.cfg.model.width),
            n_pts_per_ray=self.cfg.model.n_pts_per_ray,
            n_pts_importance=self.cfg.model.n_pts_importance,
            up_sample_steps=self.cfg.model.up_sample_steps,
            n_rays_per_image=self.cfg.model.n_rays_per_image,
            min_depth=self.cfg.model.min_depth,
            max_depth=self.cfg.model.max_depth,
            stratified=self.cfg.model.stratified,
            stratified_test=self.cfg.model.stratified_test,
            chunk_size_test=self.cfg.model.chunk_size_test,
            density_noise_std=self.cfg.model.density_noise_std,

            function_config=self.cfg.model.implicit_function,
            deformer_config=self.cfg.model.deformable_config,
            encoder_config=self.cfg.model.encoder_config,

            mask_thr=self.cfg.dataset.mask_thr,

            pairs_per_image=self.cfg.model.pairs_per_image,
            epsilon=self.cfg.model.epsilon,

            weighted_mse=self.cfg.model.get('weighted_mse', False),
        )

        self._mask_image = self.cfg.model.mask_image
        self._vali_mesh = self.cfg.vali_mesh
        if self.cfg.model.get('ray_sampler_config',None) == None:
            self.ray_sampler_config = None
        else:
            self.ray_sampler_config = self.cfg.model.ray_sampler_config

        # Set pretrained to false for inference, true for training
        # load best transforms
        if sequences is not None and load_pretrained_weights:
            no_pre_alignment = self.cfg.model.get('no_pre_alignment', False)
            if not no_pre_alignment:
                transforms_path = self.cfg.model.best_transforms
                best_transforms = torch.load(transforms_path, map_location='cpu')
                self.model.load_transforms(best_transforms)
                print(f'The best transforms are loaded from {transforms_path}.')
            else:
                print('The transform is not loaded, since "no_pre_alignment" is true.')

        # load template and hyper weights
        if load_pretrained_weights:
            hyper_path = self.cfg.model.hyper_weights
            hyper_weights = torch.load(hyper_path, map_location='cpu')['state_dict']
            self.model.load_weights(hyper_weights)
            print(f'The HyperNeuS weights are loaded from {hyper_path}.')
    
        

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
            {'params': self.model.template_parameters(),
             'lr': self.cfg.optimizer.template_lr},
            {'params': self.model.transform_parameters(),
             'lr': self.cfg.optimizer.transform_lr},
             {'params': self.model.deviation_parameters(),
             'lr': self.cfg.optimizer.template_lr},
            {'params': self.model.encoder_parameters(),
             'lr': self.cfg.optimizer.encoder_lr},
             {'params': self.model.encoder2_parameters(),
             'lr': self.cfg.optimizer.encoder2_lr},
            {'params': self.model.hyper_color_parameters(),
             'lr': self.cfg.optimizer.hyper_color_lr},
            {'params': self.model.hyper_shape_parameters(),
             'lr': self.cfg.optimizer.hyper_shape_lr},
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
            enable_trans: bool = True,
            shape_code: torch.Tensor = None,
            color_code: torch.Tensor = None,
            style_img: torch.Tensor = None,
            transform_code: torch.Tensor = None,
            produce_parameters: bool = True,
            epoch: int = 0,
            valida: bool = True,
            ray_sampler_config: str = None,
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
            enable_trans=enable_trans,
            style_image=style_img,
            transform_code=transform_code,
            produce_parameters=produce_parameters,
            epoch=epoch,
            valida=valida,
            ray_sampler_config=ray_sampler_config,
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

        #get style_img
        style_img_path = np.random.choice(self.style_img_files, 1)[0]
        with open(style_img_path, 'rb') as f:
            #这样读就是二进制的
            f = f.read()
        #这句 就是补全数据的
        f=f+B'\xff'+B'\xd9'

        im = Image.open(BytesIO(f))
        ori_style_img = im.convert('RGB')
        style_img = data_transform(ori_style_img).cuda()
        style_img = style_img[None,]
        style_idx = torch.from_numpy(np.array([self.style_img_files.index(style_img_path)]))

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
        out, metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=(fg_probability if self._mask_image else mask_crop),
            source_image=source_images,
            sequence_name=sequence_name[0],
            enable_deformation=True,
            enable_specular=True,
            style_img=style_img,
            epoch=self.current_epoch,
            ray_sampler_config=self.ray_sampler_config,
            valida = True
        )

        # get metrics
        #mse_fine = metrics['mse_coarse'] ??
        style_loss = metrics['style_loss']
        content_loss = metrics['content_loss']
        rgb_loss = metrics['rgb_loss']
        #deformation_reg_fine = metrics['deformation_reg_fine']
        #deformation_loss_fine = metrics['deformation_loss_fine']
        #weights_reg_fine = metrics['weights_reg_fine']
        #specular_reg_fine = metrics['specular_reg_fine']
        #eikonal_loss = metrics['gradient_error']
        # weights_norm = metrics['weights_norm']

        # log
        for key in ['style_loss', 'content_loss', 'rgb_loss']:
            self.log(f'train/{key}', metrics[key], on_step=True, on_epoch=False, logger=True)

        # 没有发深度信息，后续再看
        if (batch_idx == 64) and ((self.current_epoch + 1) % 5 == 0):  # log every 5 epochs
            writer = self.logger.experiment
            for k in ['rgb_fine','rgb_gt']:
                img = out[k]
                writer.add_images(f'train/{k}', torch.clamp(img, min=0.0, max=1.0),
                                 global_step=self.current_epoch + 1, dataformats='NHWC')

        # loss
        loss = self.cfg.loss.style * style_loss \
               + self.cfg.loss.content * content_loss\
               + self.cfg.loss.rgb * rgb_loss
               #+ self.cfg.loss.deformation_reg_coef * deformation_reg_fine \
               #+ self.cfg.loss.deformation_loss_coef * deformation_loss_fine \
               #+ self.cfg.loss.weights_reg_coef * (weights_reg_fine) \
               #+ self.cfg.loss.weights_igr * (eikonal_loss)
               #+ self.cfg.loss.specular_reg_coef * (specular_reg_fine) \
               #+ self.cfg.loss.weights_norm_coef * weights_norm
        self.log(f'train/loss', loss, on_step=True, on_epoch=False, logger=True)

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

        #get style_img
        style_img_path = np.random.choice(self.style_img_files, 1)[0]
        with open(style_img_path, 'rb') as f:
            #这样读就是二进制的
            f = f.read()
        #这句 就是补全数据的
        f=f+B'\xff'+B'\xd9'

        im = Image.open(BytesIO(f))
        ori_style_img = im.convert('RGB')
        style_img = data_transform(ori_style_img).cuda()
        style_img = style_img[None,]
        style_idx = torch.from_numpy(np.array([self.style_img_files.index(style_img_path)]))

        # mask image
        if self._mask_image:
            images = mask_image(images, fg_probability, self.cfg.dataset.mask_thr)

        # make sure all frames come from the same sequence
        assert len(set(sequence_name)) == 1, 'Not all frames come from the same sequence.'

        # choose source views
        n_sources = random.choice(self.cfg.model.n_sources)
        source_ids = random.sample(list(range(images.shape[0])), n_sources)
        src_img, src_fg = choose_views(source_ids, images, fg_probability) #随机选一个source image

        # cat input (mask first)
        source_images = self.prepare_input(src_img, src_fg)

        # data augment
        source_images = augment_source_images(source_images)

        # get output and metrics
        val_out, val_metrics = self(
            target_camera=cameras,
            target_image=images,
            target_fg_probability=fg_probability,
            target_mask=(fg_probability if self._mask_image else mask_crop),
            source_image=source_images,
            sequence_name=sequence_name[0],
            style_img=style_img,
            enable_deformation=True,
            enable_specular=True,
            valida = False
        )
        # write images
        if batch_idx <= 3:  # log first two samples
            # log augmented source images
            #writer.add_images(f'val_{batch_idx}/content_imgs', torch.clamp(source_images[:, : 3, :, :], min=0.0, max=1.0),
            #                  global_step=self.current_epoch + 1, dataformats='NCHW')
            writer.add_images(f'val_{batch_idx}/style_imgs', style_img[:, : 3, :, :],
                              global_step=self.current_epoch + 1, dataformats='NCHW')

            for k in ['rgb_fine', 'rgb_gt']:
                img = val_out[k]
                writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0),
                                  global_step=self.current_epoch + 1, dataformats='NHWC')
            
            for k in ['depth_fine']:
                depth = val_out[k]
                depth = depth / (depth.max() + 1.0e-8)
                writer.add_images(f'val_{batch_idx}/{k}', depth, global_step=self.current_epoch + 1,
                                  dataformats='NHWC')

            writer.add_text(f'val_{batch_idx}/sequence', sequence_name[0], global_step=self.current_epoch + 1)


        #157_17275_32769
        #415_57142_110210
        #if sequence_name[0] == '157_17275_32769' or sequence_name[0] == '415_57142_110210':
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
            mesh.export(os.path.join(self.cfg.base_log_dir, 'meshes', '{name}_{epoch:0>5d}.ply'.format(name=sequence_name[0][:3],epoch=self.current_epoch)))