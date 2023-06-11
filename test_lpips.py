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
from jt_camera import Camera

import torch
import torch.nn as nn
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
import lpips

from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(256/480, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_name',
        type=str,
        #default='hydrant_brushstrokes_OurStyle',
        #default='hydrant_brushstrokes_ReReVST',
        default='mouse_brushstrokes_ReReVST',
        help='The name of configuration file.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/test/CRF_img',
        help='output dir',
    )
    parser.add_argument(
        '--path',
        type=str,
        default='/test/CRF_img/lpips',
        #default='outputs/lpips',
        help='style dir',
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

def test(deform:bool):
    arg = parse_args()
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_name = arg.config_name
    cate, style, method = config_name.split('_')

    dir = os.path.join(arg.path,cate)

    # 设置相机的数量和焦距
    num_cameras = 30 
    focal_length = torch.tensor([10., 10.])[None]

    # 设置相机的主点为图像平面的中心
    principal_point = torch.tensor([0., 0.])[None]

    high = 1.5
    at_rate = -0.4

    #415_57152_110225 high 0.5 dist 30  at_var 1
    #194_20878_39742 high 1.5 dist 42 at_rate -0.4
    
    # define the vector and normalize it
    render_pos = []
    if cate == 'hydrant':
        #v = np.array([0.16, -0.83, -0.55]) #hyrant
        if deform: 
            v = np.array([-0.2355, -3.1086, -1.8523])#415
            high, dist, at_rate = 0.5, 30, 1
            seq = '415'
        else:
            v = np.array([0.01, 1.2, -1.72]) #194
            high, dist, at_rate = 1.5, 42, -0.4
            seq = '194'
    elif cate == 'cup':
        dist = 35
        if deform:
            v = np.array([0.42, -1.05, -1.95])
            seq = '40'
        else:
            v = np.array([0.19, -1.56, -0.97])
            seq = '34'
    elif cate == 'banana':
        dist = 40
        high = 0.5
        v = np.array([0.42, -2, -1.65])
        at_rate = -0.4
        if deform:
            seq = "41"
        else:
            seq = "12"
    elif cate == 'vase':
        dist = 42
        high = 1.5
        at_rate = -0.4
        if deform:
            v = np.array([0.12, -1.98, -0.82])
            seq = '372'
        else:
            v = np.array([-0.56, -1.74, -1.78])
            seq = '374'
    elif cate == 'toybus':
        dist = 30
        high = 2.0
        at_rate = -0.4
        if deform:
            v = np.array([0.21,-1.39,-1.24])
            seq = '386'
        else:
            v = np.array([0.19, -1.71, -1.38])
            seq = '309'
        render_pos = []
    elif cate == 'mouse':
        dist = 30
        high = 1.0
        at_rate = 0
        if deform:
            v = np.array([0.37,-0.38,-1.11])
            seq = '30'
        else:
            v = np.array([0.13,-1.01,-1.55])
            seq = '251'

    angle = torch.linspace(0, 2.0 * math.pi, num_cameras)
    traj = dist * torch.stack(
        [angle.cos(), angle.sin(), torch.tensor(high * 8 / dist).expand(angle.shape)], dim=-1
    )
    v = v / np.linalg.norm(v)

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

    transform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        ])

    def rasterize(target_pos: int, source_pos: int, style):
        target_camera = eval_cameras[target_pos].clone()
        target_camera.T = target_camera.T / 8

        if method in ['MCCNet','ReReVST','AdaIN']:
            fn_img = os.path.join(arg.path, method, "img_%s_%s_%s_%s_%s"%(cate,style,seq,method,str(source_pos)))
        else:
            fn_img = os.path.join(dir, method, "img_%s_%s_%s_%s_%s"%(cate,style,seq,method,str(source_pos)))
        rgb = Image.open(fn_img+'.png')
        rgb = transform(rgb).permute(1,2,0).reshape([-1, 3])
        fn = os.path.join(dir,"img_%s_%s_%s_%s"%(cate,seq,'Coor',str(source_pos)))
        coor = torch.tensor(np.load(fn+'.npy')).reshape([-1, 3])

        # 将世界坐标转换到相机坐标系下
        coor_cvv = target_camera.get_full_projection_transform().transform_points(coor)
        w=128
        h=128
        umax, vmax = PixeltoCvv(h=h, w=w, hid=0, wid=0)
        umin, vmin = PixeltoCvv(h=h, w=w, hid=h-1, wid=w-1)
        cvv_backup = coor_cvv.clone()
        coor_cvv[..., 0] = (coor_cvv[..., 0] + 1) / 2 * (umax - umin) + umin
        coor_cvv[..., 1] = (coor_cvv[..., 1] + 1) / 2 * (vmax - vmin) + vmin

        batch_size = 1
        point_num = rgb.shape[-2]
        rgb = rgb.reshape([1, point_num, rgb.shape[-1]])
        rgb_coor = torch.cat([rgb, coor.unsqueeze(0)], dim=-1).expand([batch_size, point_num, 6])  # (4, point, 6)
        
        k=1.5
        z=1

        pts3D = Pointclouds(points=coor_cvv.reshape([batch_size, point_num, 3]), features=rgb_coor)
        radius = float(2. / max(w, h) * k)
        idx, _, _ = rasterize_points(pts3D, [h, w], radius, z)
        alphas = torch.ones_like(idx.float())
        img = compositing.alpha_composite(
            idx.permute(0, 3, 1, 2).long(),
            alphas.permute(0, 3, 1, 2),
            pts3D.features_packed().permute(1, 0),
        )
        img = img.permute([0, 2, 3, 1]).contiguous()  # (batch, h, w, 6)
        rgb_map, coor_map = img[..., :3], img[..., 3:]  # (batch, h, w, 3)
        msk = (idx[:, :, :, :1] != -1).float()  # (batch, h, w, 1)

        return rgb_map, coor_map, msk


    lpips_model = LPIPS_M(net='vgg', version='0.1').cuda()

    outs_dist = []
    gap_list = [1,5]
    style_list = ['brushstrokes','fernand','la_muse','sketch']
    #style_list = ['la_muse','sketch']
    sum_dist = 0.
    for gap in gap_list:
        for style in style_list:
            for i in range(30):
                t_pos = i % 30
                s_pos = (i + gap) % 30
                r_rgb, coor_map, r_mask = rasterize(t_pos, s_pos, style)

                fn = os.path.join(dir,"img_%s_%s_Mask_%s"%(cate,seq,str(t_pos)))
                o_mask = Image.open(fn+'.png').convert("L")
                o_mask = transform(o_mask).unsqueeze(0)

                mask = r_mask.permute(0,3,1,2) * o_mask

                if method in ['MCCNet','ReReVST','AdaIN']:
                    fn_img = os.path.join(arg.path, method, "img_%s_%s_%s_%s_%s"%(cate,style,seq,method,str(t_pos)))
                else:
                    fn_img = os.path.join(dir, method, "img_%s_%s_%s_%s_%s"%(cate,style,seq,method,str(t_pos)))
                s_rgb = Image.open(fn_img+'.png')
                s_rgb = transform(s_rgb).unsqueeze(0)

                #s_rgb_mask = s_rgb * mask
                #r_rgb_mask = r_rgb.permute(0,3,1,2) * mask
        
                dist = lpips_model.forward(s_rgb.cuda(), r_rgb.permute(0,3,1,2).cuda(),mask.cuda()).item()
                
                outs_dist.append(dist)
                sum_dist += dist
                #print(cate+" "+method+":"+dist)

        print(cate+" "+method+" "+str(deform) +" "+seq+" gap%d"%(gap)+" average:"+str(sum_dist/120))         

def PixeltoCvv(h, w, hid=0, wid=0):
    cvv = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]]).float()
    pts = Pointclouds(points=cvv, features=cvv)
    idx, _, dist2 = rasterize_points(pts, [h, w], 1e10, 3) #idx 每个像素最近点的索引 
    a2, b2, c2 = (dist2.cpu().numpy())[0, hid, wid]
    x2 = (a2 + b2) / 2 - 1
    cosa = (x2 + 1 - a2) / (2 * x2**0.5)
    sina_abs = (1 - cosa**2)**0.5
    u = (x2 ** 0.5) * cosa
    v = (x2 ** 0.5) * sina_abs
    if np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5) > 1e-5:
        v = - (x2 ** 0.5) * sina_abs
        if(np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5) > 1e-5):
            print(np.abs((u**2 + (v-1)**2)**0.5 - c2**0.5), ' is too large...')
            print(f"Found pixel {[hid, wid]} has uv: {(u, v)} But something wrong !!!")
            print(f"a: {a2**0.5}, b: {b2**0.5}, c: {c2**0.5}, idx: {idx[0, 0, 0]}, dist2: {dist2[0, 0, 0]}")
            os.exit(-1)
    return u, v


def spatial_average(in_tens, mask, keepdim=True):
    select = torch.masked_select(in_tens,mask.bool())
    if select.size()[0] <= 1:
        return torch.tensor(0).cuda()
    else:
        return select.mean()

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

import lpips.pretrained_networks as pn

# Learned perceptual metric
class LPIPS_M(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] tune the base/trunk network
            [True] keep base/trunk frozen
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS_M, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '/root/miniconda3/lib/python3.8/site-packages/lpips/', 'weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)          

        if(eval_mode):
            self.eval()

    def get_pool_mask(self, mask):
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), dilation = 1,ceil_mode=False)
        results = [mask]
        for i in range(4):
            result = self.maxpool(results[-1])
            results.append(result)
        return results

    def forward(self, in0, in1, mask, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        outsm = self.get_pool_mask(mask=mask)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]),outsm[kk], keepdim=True) for kk in range(self.L)]  # mean([2,3],keepdim=keepdim)
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    test(deform=True)
    test(deform=False)
