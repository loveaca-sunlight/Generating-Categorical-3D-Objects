"""
Load test split and save inferenced images to disk
"""
import os
import math
import os.path as osp
import time
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from typing import Dict, List
import random

import torch
import collections
import trimesh
from mmcv import Config
from tqdm import tqdm
import numpy as np
from PIL import Image 
from io import BytesIO
from torchvision import transforms
from torchvision import utils as vutils
import cv2

from dataset.dataloader_zoo import dataloader_zoo
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from dataset.utils import DATASET_TYPE_TRAIN, DATASET_TYPE_KNOWN
from evaluation.util import generate_circular_cameras
from models import MODELS
from models.util import mask_image, choose_views, augment_source_images
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(256/480, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_name',
        type=str,
        default='hydrant_hyper_igr',
        help='The name of configuration file.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/hydrant_hyper_igr/last.ckpt',
        help='The path of checkpoint file.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='output dir',
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=200,
        help='Number of test samples per chunk.'
    )
    parser.add_argument(
        '--start_chunk',
        type=int,
        default=0
    )
    parser.add_argument(
        '--transform',
        type=str,
        default=None
    )
    parser.add_argument(
        '--save_img',
        type=bool,
        default=True
    )

    return parser.parse_args()


@torch.no_grad()
def prepare_input(src_img: torch.Tensor, src_fg: torch.Tensor, mask_thr: float):
    fg = src_fg.clone()
    fg[fg < mask_thr] = 0.0
    inputs = torch.cat([src_img, fg], dim=1)  # (n, 4, h, w)
    return inputs


def test():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    DATASET_CONFIGS['default']['image_width'] = cfg.width
    DATASET_CONFIGS['default']['image_height'] = cfg.height
    # depth data is not needed
    DATASET_CONFIGS['default']['load_depths'] = False
    DATASET_CONFIGS['default']['load_depth_masks'] = False

    data = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
        test_on_train=True,
        load_key_points=True,  # to filter out sequences without key point
        limit_to=-1,
        limit_sequences_to=-1,
    )

    # dataloader
    loader = dataloader_zoo(
        datasets=data,
        dataset_name='co3d_multisequence',
        batch_size=cfg.batch_size,
        batch_size_val=cfg.batch_size_val,
        num_workers=cfg.num_workers,
        dataset_len=cfg.dataset.dataset_len,
        dataset_len_val=cfg.dataset.dataset_len_val,
        curriculum_config=cfg.dataset.get('curriculum_config', None),
        val_follows_train=cfg.dataset.get('val_follows_train', False)
    )
    train_loader, val_loader = [loader[key] for key in ['train', 'val']]

    # model
    sequences = list(data['train'].seq_annots.keys())
    model = MODELS.build(
        {
            'type': cfg.model.type,
            'cfg': cfg,
            'sequences': sequences,
            'load_pretrained_weights': False,
            'calibration_mode': True,
        }
    )
    state_dict = torch.load(arg.checkpoint, map_location='cpu')['state_dict']
    '''state_dict = collections.OrderedDict(
        [
            (k, state_dict[k])
            for k in filter(lambda x: '_grid_raysampler._xy_grid' not in x, state_dict.keys())
        ]
    )  # for checkpoint of old version
    state_dict = OrderedDict(filter(lambda x: 'transforms.' not in x[0], state_dict.items()))
    state_dict = OrderedDict(filter(lambda x: 'deviations.' not in x[0], state_dict.items()))'''
    #del_list = ["model.transforms.transforms.385_45782_91506", "model.transforms.transforms.386_45949_91795", "model.transforms.transforms.387_46716_93021", "model.transforms.transforms.397_49981_98390", "model.deviations.deviations.385_45782_91506", "model.deviations.deviations.386_45949_91795", "model.deviations.deviations.387_46716_93021", "model.deviations.deviations.397_49981_98390"]
    '''del_list = ["model.transforms.transforms.398_50425_98971", "model.deviations.deviations.398_50425_98971"]
    for i in del_list:
        del state_dict[i]'''
    model.load_state_dict(state_dict)
    print(f'Model weights are loaded from {arg.checkpoint}.')
    model.freeze()
    model.to(device)
    print('Model weights are frozen.')

    # load transforms
    if cfg.model.best_transforms is None:
        fn = osp.join('checkpoints/', arg.config_name, 'sequence_transform.pt')
    else:
        fn = cfg.model.best_transforms
    sequence_transform = torch.load(fn, map_location=device)
    print(f'sequence_transform is loaded from {fn}.')

    # make output directory
    output_dir = arg.output
    cate = arg.config_name.split('_')[0]
    output_dir = arg.output
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    cate = arg.config_name.split('_')[0]
    output_dir = os.path.join(output_dir,cate)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            # skip if batch_idx < start_chunk * chunk_size
            #if batch_idx > 40:
            #    break

            # to device
            batch_data = batch_data.to(device)

            # get data
            images = batch_data.image_rgb
            cameras = batch_data.camera
            sequence_name = batch_data.sequence_name
            fg_probability = batch_data.fg_probability

            # number of images
            n_images = images.shape[0]

            #获得mask 图片——————
            '''for i in range(n_images):
                src_img = images[i].masked_fill(fg_probability[i]<0.7,255)
                vutils.save_image(src_img,os.path.join(output_dir,'img_'+str(i)+'_'+sequence_name[0]+'.png'))
            continue'''

            # # only test when n_source_image == 1
            # if n_images > 2:
            #     tqdm.write(f'The number of source image is {n_images - 1}, skipping...')
            #     continue

            # source and target ids
            n_sources = random.choice([5])
            for idx,fg in enumerate(fg_probability):
                if fg.max() == 0.:
                    images = torch.cat(images[:idx],images[idx+1:])
                    fg_probability = torch.cat(fg_probability[:idx],fg_probability[idx+1:])

            source_ids = random.sample(list(range(images.shape[0])), n_sources)

            images = mask_image(images, fg_probability, cfg.dataset.mask_thr)

            # choose source and target data
            src_images, src_fgs = choose_views(source_ids, images, fg_probability)

            # 设置相机的数量和焦距
            num_cameras = 30 
            focal_length = torch.tensor([10., 10.])[None]

            # 设置相机的主点为图像平面的中心
            principal_point = torch.tensor([0., 0.])[None]

            value = batch_idx / 20 - 1

            dist = 30.0
            high = 0.6
            # view_center[..., -1] *= up_scale
            # view_center[..., : -1] = 0.0
            
            angle = torch.linspace(0, 2.0 * math.pi, num_cameras)
            traj = dist * torch.stack(
                [angle.cos(), angle.sin(), torch.tensor(high * 8 / dist).expand(angle.shape)], dim=-1
            )
            # define the vector and normalize it
            v = np.array([0.16, -0.83, -0.55])
            n = v / np.linalg.norm(v)

            # 求出v与Z轴之间的夹角
            theta = np.arccos(v[2]/np.linalg.norm(v))
            # 求出旋转轴，即v与Z轴的叉积
            k = np.cross(v,[0,0,1])
            k = k/np.linalg.norm(k) # 单位化
            # 计算三维旋转矩阵
            trans = np.cos(theta)*np.eye(3) + np.sin(theta)*np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]]) + (1-np.cos(theta))*np.outer(k,k)

            ''''# define the rotation angle in radians
            theta = math.pi / 4 # 45 degrees

            # calculate the rotation matrix using Rodrigues' formula
            nx = n[0]
            ny = n[1]
            nz = n[2]
            c = math.cos(theta)
            s = math.sin(theta)
            trans = np.array([[c + nx**2 * (1 - c), nx * ny * (1 - c) - nz * s, nx * nz * (1 - c) + ny * s],
                        [ny * nx * (1 - c) + nz * s, c + ny**2 * (1 - c), ny * nz * (1 - c) - nx * s],
                        [nz * nx * (1 - c) - ny * s, nz * ny * (1 - c) + nx * s, c + nz**2 *(  1- c)]])'''
            
            trans = torch.from_numpy(trans).float()

            for i in range(len(traj)):
                traj[i] = torch.matmul(traj[i], trans)

            at_var = tuple([x * 8 for x in (-0.004077, 0.20015, 0.13277)])
            #up = ((0.1, -1, -0.6),)
            up = ((0,-0.8,-0.6),)
            # 设置相机的RT
            R, T = look_at_view_transform(
                eye=traj,
                device=traj.device,
                #up=((0., 0., 1.),),
                up=up,
                at=(tuple([x * high for x in at_var]),)
            )

            # 创建一批PerspectiveCameras类数据
            eval_cameras = [
                PerspectiveCameras(
                    focal_length=focal_length,
                    principal_point=principal_point.clone().detach(),
                    R=R_[None],
                    T=T_[None],
                )
                for i, (R_, T_) in enumerate(zip(R, T))
            ]

            # prepare input
            src_input = prepare_input(src_images, src_fgs, cfg.dataset.mask_thr)
            src_input = augment_source_images(src_input)


            # inference
            transform_code = sequence_transform[sequence_name[0]]
            if transform_code.ndim == 1:
                transform_code = transform_code[None]

            outs = []
            render_pos = [3,9,15] #hydrant
            #render_pos = [5] # cup
            #render_pos = [26]
            for i in range(len(eval_cameras)):
                if len(render_pos)>0:
                    if i not in render_pos:
                        continue
                out, val_metrics = model(
                    target_camera=eval_cameras[i].cuda(),
                    #target_camera=cameras[i],
                    #target_image=images[i][None],
                    #target_fg_probability=fg_probability[i][None],
                    #target_mask=fg_probability[i][None],
                    target_image=None,
                    target_fg_probability=None,
                    target_mask=None,
                    source_image=src_input,
                    sequence_name=sequence_name[0],
                    enable_deformation=True,
                    enable_specular=True,
                    enable_trans=False,
                    valida = False
                )
                
                mask_bool = (out['depth_fine'] < 0.7)
                rgb = out['rgb_fine'].masked_fill(mask_bool,255)
                
                outs.append(rgb)
                if arg.save_img:
                    # fetch output images
                    '''results = torch.cat((
                        rgb.permute(0,3,1,2),
                        out['rgb_gt'].permute(0,3,1,2),),dim=0)
                    result = vutils.make_grid(results,nrow=4,padding=2)'''
                    #results = torch.cat((rgb.permute(0,3,1,2),out['rgb_gt'].permute(0,3,1,2),),dim=0)
                    #result = vutils.make_grid(results,nrow=4,padding=2)
                    # save image
                    result = rgb.permute(0,3,1,2)
                
                    fn = os.path.join(output_dir,'img_'+cate+sequence_name[0]+'_D'+str(i)+'.png')
                    #fn = os.path.join(output_dir,'img_'+'Flase'+'_C'+str(i)+'.png')
                    vutils.save_image(result,fn)
            #src = src_input[0][:3].masked_fill((src_input[0][3:4]<0.5),255)
            src_img = src_input[0][:3].masked_fill(src_input[0][3:4]<0.7,255)
            vutils.save_image(src_img,os.path.join(output_dir,'img_'+cate+sequence_name[0]+'src'+'.png'))
            
            save_vid = False
            if save_vid:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_path = os.path.join(output_dir,'video'+cate+sequence_name[0]+'.mp4')
                video = cv2.VideoWriter(vid_path, fourcc, 5.0, (128, 128))
                for tensor in outs:
                    # 将tensor转换为numpy数组，并调整维度顺序和颜色空间（opencv默认是BGR）
                    img = tensor[0].cpu().numpy() * 255 # 转换为0-255范围的整数值
                    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
                    # 写入视频帧
                    video.write(img)
                # 释放资源
                video.release()

            vali_mesh = False
            if vali_mesh:
                resolution = 300 #300
                threshold = 0.0
                bound_min = torch.tensor(cfg.vali_near, dtype=torch.float32)
                bound_max = torch.tensor(cfg.vali_far, dtype=torch.float32)

                vertices, triangles =\
                    model.model.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
                #os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)

                #if world_space:
                #    vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

                mesh = trimesh.Trimesh(vertices, triangles)
                mesh.export(os.path.join(output_dir, '{name}.ply'.format(name=sequence_name[0][:3])))


    


if __name__ == '__main__':
    test()
