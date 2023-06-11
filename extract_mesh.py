"""
Generate video for as2
"""
import collections
import os.path as osp
import random
from argparse import ArgumentParser

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from kornia.utils import create_meshgrid3d
from mmcv import Config
from skimage.measure import marching_cubes

from dataset.co3d_dataset import FrameData
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from models import MODELS
from models.util import choose_views, mask_image
from tools import save_to_ply


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='The path of checkpoint file.'
    )
    parser.add_argument(
        'seq_name',
        type=str,
        help='Name of sequence.'
    )
    parser.add_argument(
        '--source_ids',
        type=int,
        nargs='+',
        default=None
    )
    parser.add_argument(
        '--volume_size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--thr',
        type=float,
        default=10.0
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/'
    )
    parser.add_argument(
        '--disable_deformation',
        action='store_true'
    )

    return parser.parse_args()


@torch.no_grad()
def prepare_input(src_img: torch.Tensor, src_fg: torch.Tensor, mask_thr: float):
    fg = src_fg.clone()
    fg[fg < mask_thr] = 0.0
    inputs = torch.cat([src_img, fg], dim=1)  # (n, 4, h, w)
    return inputs


@torch.no_grad()
def extract_mech():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model type
    model_type = cfg.model.type

    # model
    if model_type in ['hyper_nerf_model']:
        sequence_names = mmcv.list_from_file(osp.join(osp.dirname(cfg.model.best_transforms), 'sequence_names.txt'))
    else:
        sequence_names = None

    model = MODELS.build(
        {
            'type': cfg.model.type,
            'cfg': cfg,
            'sequences': sequence_names,
            'load_pretrained_weights': False
        }
    )
    state_dict = torch.load(arg.checkpoint)['state_dict']
    state_dict = collections.OrderedDict(
        [
            (k, state_dict[k])
            for k in filter(lambda x: '_grid_raysampler._xy_grid' not in x, state_dict.keys())
        ]
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.freeze()
    print(f'Weights are loaded from {arg.checkpoint}.')

    # dataset
    DATASET_CONFIGS['default']['image_width'] = cfg.width
    DATASET_CONFIGS['default']['image_height'] = cfg.height
    # depth data is not needed
    DATASET_CONFIGS['default']['load_depths'] = False
    DATASET_CONFIGS['default']['load_depth_masks'] = False

    dataset = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
        test_on_train=False
    )['val']

    # select sequence
    seq_to_idx = dataset.seq_to_idx
    sel_ids = seq_to_idx[arg.seq_name]
    frames = FrameData.collate([dataset[idx] for idx in sel_ids]).to(device)

    if arg.source_ids is None:
        img_ids = random.sample(list(range(len(sel_ids))), 7)
        sel_images, sel_fgs = choose_views(img_ids, frames.image_rgb, frames.fg_probability)
        # if model_type in ['pixel_nerf_model', 'grf_model']:
        #     source_cameras = choose_views(img_ids, frames.camera)
    else:
        sel_frames = FrameData.collate([dataset[idx] for idx in arg.source_ids]).to(device)
        sel_images, sel_fgs = sel_frames.image_rgb, sel_frames.fg_probability
        # if model_type in ['pixel_nerf_model', 'grf_model']:
        #     source_cameras = sel_frames.camera
    print(f'Recon with {len(sel_images)} images.')
    # resize
    sel_images = F.interpolate(sel_images, (128, 128), mode='bilinear', align_corners=False)
    sel_fgs = F.interpolate(sel_fgs, (128, 128), mode='bilinear', align_corners=False)
    source_images = mask_image(sel_images, sel_fgs, threshold=cfg.dataset.mask_thr)
    source_images = prepare_input(source_images, sel_fgs, cfg.dataset.mask_thr)

    # initialize volume
    grid_size = arg.volume_size
    grid3d = create_meshgrid3d(grid_size, grid_size, grid_size, normalized_coordinates=True, device=device,
                               dtype=torch.float) * 8.0  # 8.0 is the radius of co3d objects
    # compute density
    densities = model.model.compute_point_density(
        target_points=grid3d.reshape(1, grid_size ** 3, 3),
        source_image=source_images,
        enable_deformation=(not arg.disable_deformation),
        chunk_size=9600
    ).view(grid_size, grid_size, grid_size).cpu().numpy()  # (1, p, 1)

    verts, faces, _, _ = marching_cubes(densities, arg.thr)

    save_to_ply(osp.join(arg.output_dir, f'{arg.seq_name}_mesh.ply'), verts, faces)

    print('Done.')


if __name__ == '__main__':
    extract_mech()
