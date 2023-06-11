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
        default='toybus_402',
        help='The name of configuration file.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/toybus_402/last.ckpt',
        help='The path of checkpoint file.',
    )
    parser.add_argument(
        '--output',
        type=str,
        #default='/test/CRF_img/lpips',
        default='outputs',
        help='output dir',
    )
    parser.add_argument(
        '--deform',
        type=bool,
        default=True,
    )
    parser.add_argument(
        '--val',
        type=bool,
        default=True,
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
        batch_size=100,
        batch_size_val=cfg.batch_size_val,
        num_workers=cfg.num_workers,
        dataset_len=1,
        dataset_len_val=1,
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
            'style_data': 'validation'
        }
    )
    state_dict = torch.load(arg.checkpoint, map_location='cpu')['state_dict']
    
    model.load_state_dict(state_dict)
    print(f'Model weights are loaded from {arg.checkpoint}.')

    model.freeze()
    model.to(device)
    print('Model weights are frozen.')

    # make output directory
    '''output_dir = osp.join('checkpoints/', arg.config_name, 'results/')
    if not osp.exists(osp.join('checkpoints/', arg.config_name)):
        os.mkdir(osp.join('checkpoints/', arg.config_name))
    if not osp.exists(output_dir):
        os.mkdir(output_dir)'''
    output_dir = arg.output
    cate, seq_name = arg.config_name.split('_')
    
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    #output_dir = os.path.join(output_dir,cate,'NeRF')
    if arg.val:
        output_dir = os.path.join(output_dir,cate,'%s_val'%seq_name)
    else:
        output_dir = os.path.join(output_dir,cate,seq_name)
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
            mask_crop = batch_data.mask_crop
            fg_probability = batch_data.fg_probability

            #if not (sequence_name[0] == '194_20878_39742' or sequence_name[0]=='415_57152_110225' or sequence_name[0]=='411_55961_107847'): #hydrant
            #    continue
            #if not (sequence_name[0] == '14_164_1113' or sequence_name[0] == '14_175_1075'): cup
            #    continue
            #if not (sequence_name[0] == '45_2396_6862' or sequence_name[0] == '49_2826_7701'): #banana
            #    continue

            images = mask_image(images, fg_probability, cfg.dataset.mask_thr)

            # 设置相机的数量和焦距
            num_cameras = 100 
            focal_length = torch.tensor([10., 10.])[None]

            # 设置相机的主点为图像平面的中心
            principal_point = torch.tensor([0., 0.])[None]

            value = batch_idx / 20 - 1

            #dist = 45.0
            #high = 2.0

            dist = 42
            high = 1.5
            at_rate = -0.4
            # view_center[..., -1] *= up_scale
            # view_center[..., : -1] = 0.0
            
            
            deform = arg.deform
            # define the vector and normalize it
            render_pos = []
            if cate == 'hydrant':
                #v = np.array([0.16, -0.83, -0.55]) #hyrant
                if deform: #415
                    v = np.array([-0.2355, -3.1086, -1.8523])
                    high, dist, at_rate = 0.5, 30, 1
                else:
                    v = np.array([0.01, 1.2, -1.72]) #194
                    high, dist, at_rate = 1.5, 42, -0.4
                render_pos = []
            elif cate == 'cup':
                dist = 35
                if deform:
                    v = np.array([0.42, -1.05, -1.95])
                else:
                    v = np.array([0.19, -1.56, -0.97])
                    seq_name = '34'
            elif cate == 'banana':
                dist = 40
                high = 0.5
                v = np.array([0.42, -2, -1.65])
                if not deform:
                    seq_name = '12'
            elif cate == 'vase':
                dist = 42
                high = 1.5
                at_rate = -0.4
                if deform:
                    v = np.array([0.12, -1.98, -0.82])
                else:
                    v = np.array([-0.56, -1.74, -1.78])
            elif cate == 'toybus':
                dist = 30
                high = 2.0
                at_rate = -0.4
                if deform:
                    #v = np.array([0.21,-1.39,-1.24]) 386
                    v = np.array([0.26,-0.68,-2.12]) #402
                    high = 3.0
                else:
                    v = np.array([0.19, -1.71, -1.38])
                    seq_name = '309'
            elif cate == 'mouse':
                dist = 30
                high = 1.0
                at_rate = 0
                if deform:
                    v = np.array([0.37,-0.38,-1.11])
                else:
                    v = np.array([0.13,-1.01,-1.55])
                    seq_name = '251'
            render_pos = []
            v = v / np.linalg.norm(v)

            angle = torch.linspace(0, 2.0 * math.pi, num_cameras)
            traj = dist * torch.stack(
                [angle.cos(), angle.sin(), torch.tensor(high * 8 / dist).expand(angle.shape)], dim=-1
            )

            # 求出v与Z轴之间的夹角
            theta = np.arccos(v[2]/np.linalg.norm(v))
            # 求出旋转轴，即v与Z轴的叉积
            k = np.cross(v,[0,0,1])
            k = k/np.linalg.norm(k) # 单位化
            # 计算三维旋转矩阵
            trans = np.cos(theta)*np.eye(3) + np.sin(theta)*np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]]) + (1-np.cos(theta))*np.outer(k,k)
            
            trans = torch.from_numpy(trans).float()

            for i in range(len(traj)):
                traj[i] = torch.matmul(traj[i], trans)

            at_var = tuple([x * at_rate for x in v])
            #up = ((0.1, -1, -0.6),)
            up = (v,)
            # 设置相机的RT
            R, T = look_at_view_transform(
                eye=traj,
                device=traj.device,
                #up=((0., 0., 1.),),
                up=up,
                at=(tuple([x for x in at_var]),)
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

            outs = []
            #render_pos = [18,23,26]  #hydrant
            #render_pos = [2,14,24]
            #render_pos = [5] # cup
            render_pos = []
            if arg.val:
                cameras = eval_cameras
            #for i in range(len(eval_cameras)):
            for i in range(len(cameras)):
                if len(render_pos)>0:
                    if i not in render_pos:
                        continue
                # get output and metrics
                if arg.val:
                    out, val_metrics = model(
                        target_camera=cameras[i].cuda(),
                        target_image=images[i][None],
                        target_fg_probability=fg_probability[i][None],
                        target_mask=fg_probability[i][None],
                        valida = False
                    )
                else:
                    out, val_metrics = model(
                        target_camera=cameras[i],
                        target_image=images[i][None],
                        target_fg_probability=fg_probability[i][None],
                        target_mask=fg_probability[i][None],
                        valida = False
                    )
                
                mask_bool = (out['depth_fine'] < 0.7)
                rgb = out['rgb_fine']#.masked_fill(mask_bool,255)
                
                outs.append(rgb)
                save_img = True
                if save_img:
                    # fetch output images
                    result = rgb.permute(0,3,1,2)
                    '''results = torch.cat((
                        rgb.permute(0,3,1,2),
                        style_imgs[0],),dim=0)
                    result = vutils.make_grid(results,nrow=4,padding=2)'''
                    # results = torch.cat((rgb.permute(0,3,1,2),out['rgb_gt'].permute(0,3,1,2),),dim=0)
                    # result = vutils.make_grid(results,nrow=4,padding=2)
                    # save image
                    
                    # 保存数据集形式
                    fn = os.path.join(output_dir, 'img_'+str(i))
                    if arg.val:
                        vutils.save_image(result,fn+'.png')
                    else:
                        vutils.save_image(out['rgb_gt'].permute(0,3,1,2), fn+'.png')
                    #
                    vutils.save_image(out['depth_fine'].permute(0,3,1,2), fn+'_depth.png')
                    np.save(fn+'_rays_o.npy',out['rays_o'][0].cpu().numpy())
                    np.save(fn+'_rays_d.npy',out['rays_d'][0].cpu().numpy())
                    pos = np.eye(4)
                    pos[:3,:3] = cameras[i].R.cpu()
                    pos[:3,3] = (cameras[i].T/8).cpu()
                    np.save(fn+'_pos.npy',pos)

                    '''#nerf生成图
                    fn = os.path.join(output_dir,'img_'+cate+'_'+seq_name+'_NeRF_'+str(i))
                    vutils.save_image(result,fn+'.png')'''

                    #深度图
                    #fn = os.path.join(output_dir,'img_'+cate+'_'+seq_name+'_Mask_'+str(i))
                    #vutils.save_image(out['depth_fine'].permute(0,3,1,2),fn+'.png')
            
            save_vid = False
            if save_vid:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_path = os.path.join(output_dir,'video'+cate+arg.config_name+'.mp4')
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
                mesh.export(os.path.join(output_dir, '{name}.ply'.format(arg.config_name[:3])))


    


if __name__ == '__main__':
    test()
