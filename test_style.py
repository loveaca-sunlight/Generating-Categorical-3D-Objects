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
from models.util import mask_image, choose_views, augment_source_images, find_files
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
        default='mouse_style_test',
        help='The name of configuration file.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        #default='checkpoints/banana_style_pre/epoch=519-step=25680.ckpt',
        #default='checkpoints/hydrant_style_lr1/last.ckpt',
        default='checkpoints/mouse_style_pre/epoch=1599-step=143000.ckpt',
        #default='checkpoints/vase_style_pre2/epoch=2149-step=47050.ckpt',
        #default='checkpoints/toybus_style_pre/last.ckpt',
        help='The path of checkpoint file.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/test/CRF_img/lpips',
        #default='outputs',
        help='output dir',
    )
    parser.add_argument(
        '--style_path',
        type=str,
        default='wikiart/crf2',
        help='style dir',
    )
    parser.add_argument(
        '--deform',
        type=bool,
        default=True,
    )
    parser.add_argument(
        '--render_pos',
        nargs='*',
        type=int,
    )
    parser.add_argument(
        '--show_img',
        type=bool,
        default=True,
    )
    parser.add_argument(
        '--show_video',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--show_mesh',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--show_lpips',
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

    cate = arg.config_name.split('_')[0]

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
            'style_data': 'validation'
        }
    )
    state_dict = torch.load(arg.checkpoint, map_location='cpu')['state_dict']
    #   #del_list = []
    if cate == 'vase':
        del_list = ["model.transforms.transforms.270_28782_57216", "model.transforms.transforms.374_42307_84569", "model.transforms.transforms.377_43385_86507", "model.transforms.transforms.396_49453_97605", "model.transforms.transforms.398_50425_98971", "model.transforms.transforms.401_52181_102270", "model.deviations.deviations.270_28782_57216", "model.deviations.deviations.374_42307_84569", "model.deviations.deviations.377_43385_86507", "model.deviations.deviations.396_49453_97605", "model.deviations.deviations.398_50425_98971", "model.deviations.deviations.401_52181_102270"]
    elif cate == 'toybus':
        del_list = ["model.transforms.transforms.385_45782_91506", "model.transforms.transforms.386_45949_91795", "model.transforms.transforms.387_46716_93021", "model.transforms.transforms.397_49981_98390", "model.deviations.deviations.385_45782_91506", "model.deviations.deviations.386_45949_91795", "model.deviations.deviations.387_46716_93021", "model.deviations.deviations.397_49981_98390"]
    else:
        del_list = []
    
    for i in del_list:
        del state_dict[i]

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
    '''output_dir = osp.join('checkpoints/', arg.config_name, 'results/')
    if not osp.exists(osp.join('checkpoints/', arg.config_name)):
        os.mkdir(osp.join('checkpoints/', arg.config_name))'''
    
    output_dir = arg.output
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    
    output_dir = os.path.join(output_dir,cate)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    if not osp.exists(output_dir+'/OurStyle'):
        os.mkdir(output_dir+'/OurStyle')

    style_img_files = find_files(arg.style_path, exts=['*.png', '*.jpg'])

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
            #cameras = torch.load('toy_camera_309.pt').cuda()
            sequence_name = batch_data.sequence_name
            fg_probability = batch_data.fg_probability

            #if not (sequence_name[0] == '349_36568_68004' or sequence_name[0]=='415_57152_110225' or sequence_name[0]=='411_55961_107847'): #hydrant
            #    continue
            #if not (sequence_name[0] == '14_164_1113' or sequence_name[0] == '14_175_1075'): cup
            #    continue
            #if not (sequence_name[0] == '45_2396_6862' or sequence_name[0] == '49_2826_7701'): #banana
            #    continue
            #if not (batch_idx <50):
            #if not (sequence_name[0] == '194_20878_39742' or sequence_name[0]=='415_57152_110225'):
            #    continue
            # number of images
            n_images = images.shape[0]

            save_source = True
            if save_source:
                #for i in range(n_images):
                for i in range(8):
                    src_img = images[i].masked_fill(fg_probability[i]<0.7,255)
                    vutils.save_image(src_img,os.path.join(output_dir,'img_'+str(i)+'_'+sequence_name[0]+'.png'))

            # source and target ids
            n_sources = random.choice([5])
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

            dist = 42 #hydrant
            #dist = 45 banana
            #high = 0.6 #hydrant
            high = 1.5
            at_rate = -0.4
            #high = 2.0 #banana
            #high = 5

            #415_57152_110225 high 0.5 dist 30  at_var 1
            #194_20878_39742 high 1.5 dist 42 at_rate -0.4
            
            deform = arg.deform
            # define the vector and normalize it
            render_pos = []
            if cate == 'hydrant':
                #v = np.array([0.16, -0.83, -0.55]) #hyrant
                if deform: 
                    #415
                    v = np.array([-0.2355, -3.1086, -1.8523])
                    high, dist, at_rate = 0.5, 30, 1
                    #427
                    #v = np.array([0.3,-1.84,-1.14])
                    #high, dist, at_rate = 1.5, 35, -0.2
                    #349
                    #v = np.array([-0.014, -1.7896, -1.8956])
                    #high, dist, at_rate = 1.0, 30, -0.2
                else:
                    v = np.array([0.01, 1.2, -1.72]) #194
                    high, dist, at_rate = 1.5, 42, -0.4
            elif cate == 'cup':
                dist = 35
                if deform:
                    v = np.array([0.42, -1.05, -1.95])
                else:
                    v = np.array([0.19, -1.56, -0.97])
            elif cate == 'banana':
                #dist = 40
                dist = 45
                #high = 0.5
                high = 2.0
                v = np.array([0.42, -2, -1.65])
                #v = np.array([-0.639, -1.3134, -2.1108])
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
                    v = np.array([0.21,-1.39,-1.24])
                else:
                    v = np.array([0.19, -1.71, -1.38])
                render_pos = []
            elif cate == 'mouse':
                dist = 35
                high = 1.5
                at_rate = 0
                if deform:
                    v = np.array([0.37,-0.38,-1.11])
                else:
                    v = np.array([0.13,-1.01,-1.55])

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
                #at=(tuple([x * high for x in at_var]),)
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

            # prepare input
            src_input = prepare_input(src_images, src_fgs, cfg.dataset.mask_thr)
            src_input = augment_source_images(src_input)

            for style_img_path in style_img_files:
                #style_img_path = "wikiart/style/brushstrokes.jpg"
                #style_img_path = "wikiart/style/sketch.jpg"
                #style_img_path = "wikiart/style/antimonocromatismo.jpg"
                #style_img_path = "wikiart/style/en_campo_gris.jpg"
                #np.random.choice(model.style_img_files, 1)[0]
                with open(style_img_path, 'rb') as f:
                    #这样读就是二进制的
                    f = f.read()
                #这句 就是补全数据的
                f=f+B'\xff'+B'\xd9'

                im = Image.open(BytesIO(f))
                ori_style_img = im.convert('RGB')
                style_img = data_transform(ori_style_img).cuda()
                #style_idx = torch.from_numpy(np.array([model.style_img_files.index(style_img_path)]))
                style_name = style_img_path.split('/')[-1].split('.')[0]
                vutils.save_image(style_img,os.path.join(output_dir,'img_'+cate+'_'+style_name+'.png'))

                outs = []
                #render_pos = [18]
                #render_pos = [6] # cup
                
                #render_pos = [2,14,24]
                render_pos = [] 
                for i in range(len(eval_cameras)):
                    if len(render_pos)>0:
                        if i not in render_pos:
                            continue
                    out, val_metrics = model(
                        target_camera=eval_cameras[i].cuda(),
                        #target_camera=cameras[i],
                        target_image=None,
                        target_fg_probability=None,
                        target_mask=None,
                        source_image=src_input,
                        sequence_name=sequence_name[0],
                        #sequence_name="402_52524_102918",
                        style_img=style_img[None,],
                        enable_deformation=True,
                        enable_specular=True,
                        enable_trans=True,
                        valida = False
                    )
                    
                    mask_bool = (out['depth_fine'] < 0.7)
                    rgb = out['rgb_fine'].masked_fill(mask_bool,255)
                    
                    outs.append(rgb)
                    save_img = arg.show_img
                    if save_img:
                        # fetch output images
                        '''results = torch.cat((
                            rgb.permute(0,3,1,2),
                            style_imgs[0],),dim=0)
                        result = vutils.make_grid(results,nrow=4,padding=2)'''
                        #results = torch.cat((rgb.permute(0,3,1,2),out['rgb_gt'].permute(0,3,1,2),),dim=0)
                        #result = vutils.make_grid(results,nrow=4,padding=2)
                        # save image
                        result = rgb.permute(0,3,1,2)
                    
                        if arg.show_lpips:
                            fn = os.path.join(output_dir,'OurStyle','img_'+cate+'_'+style_name+'_'+sequence_name[0].split('_')[0]+'_OurStyle_'+str(i))
                            vutils.save_image(result,fn+'.png')

                            fn = os.path.join(output_dir,'img_'+cate+'_'+sequence_name[0].split('_')[0]+'_Coor_'+str(i))
                            np.save(fn+'.npy',out['coor'][0].cpu().numpy())

                            fn = os.path.join(output_dir,'img_'+cate+'_'+sequence_name[0].split('_')[0]+'_Mask_'+str(i))
                            vutils.save_image((out['depth_fine'] > 0.7).float().permute(0,3,1,2),fn+'.png')
                        else:
                            fn = os.path.join(output_dir,'img_'+cate+'_'+style_name+'_'+sequence_name[0]+'_OurStyle_'+str(i)) #.split('_')[0]
                            #fn = os.path.join(output_dir,'img_'+'Flase'+'_C'+str(i)+'.png')
                            vutils.save_image(result,fn+'.png')
                #src = src_input[0][:3].masked_fill((src_input[0][3:4]<0.5),255)
                src_img = src_input[0][:3].masked_fill(src_input[0][3:4]<0.7,255)
                #vutils.save_image(src_img,os.path.join(output_dir,'img_'+cate+sequence_name[0]+'src'+'.png'))
                
                save_vid = arg.show_video
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

            vali_mesh = arg.show_mesh
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
