"""
Generate video for hypersim
"""
import collections
import datetime
import glob
import os
import os.path as osp
import random
from argparse import ArgumentParser

import imageio
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from mmcv import Config

from dataset.co3d_dataset import FrameData
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from evaluation.cam_util import generate_eval_video_cameras
from evaluation.util import save_image, tensors2images, tensors2depths, depth_to_mask
from models import MODELS
from models.util import choose_views, mask_image
from tools.utils import Timer


# render size, (w, h)
_RENDER_SIZE = (128, 128)
_TEMPLATES = {
    'hydrant': '427_60021_116224'
}


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
        '--output_dir',
        type=str,
        default='outputs/'
    )
    parser.add_argument(
        '--n_eval_cams',
        type=int,
        default=96
    )
    parser.add_argument(
        '--disable_deformation',
        action='store_true'
    )
    parser.add_argument(
        '--disable_specular',
        action='store_true'
    )

    return parser.parse_args()


def pixelnerf_grf_forward(model, tgt_cams, src_imgs, src_cams):
    out, _ = model(
            target_camera=tgt_cams,
            target_image=None,
            target_mask=None,
            source_image=src_imgs,
            source_camera=src_cams,
            n_sources=src_imgs.shape[0]
        )
    return out


def scenenerf_forward(model, tgt_cams):
    out, _ = model(
        target_camera=tgt_cams,
        target_image=None,
        target_mask=None
    )
    return out


@torch.no_grad()
def prepare_input(src_img: torch.Tensor, src_fg: torch.Tensor, mask_thr: float):
    fg = src_fg.clone()
    fg[fg < mask_thr] = 0.0
    inputs = torch.cat([src_img, fg], dim=1)  # (n, 4, h, w)
    return inputs


def hypersim_forward(model, tgt_cams, src_imgs, transform_code, enable_deformation, enable_specular):
    out, _ = model(
        target_camera=tgt_cams,
        target_image=None,
        target_fg_probability=None,
        target_mask=None,
        source_image=src_imgs,
        sequence_name=None,
        enable_deformation=enable_deformation,
        enable_specular=enable_specular,
        transform_code=transform_code
    )
    return out


def recon():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))
    # change size
    cfg.width, cfg.height = _RENDER_SIZE
    cfg.model.width, cfg.model.height = _RENDER_SIZE

    # change chunk_size_test
    cfg.model.chunk_size_test = arg.n_eval_cams * 80

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    random.seed(0)

    # model type
    model_type = cfg.model.type
    if model_type in ['pixel_nerf_model', 'grf_model']:
        assert _RENDER_SIZE == (128, 128)

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
        img_ids = random.sample(list(range(len(sel_ids))), 10)
        sel_images, sel_fgs = choose_views(img_ids, frames.image_rgb, frames.fg_probability)
        if model_type in ['pixel_nerf_model', 'grf_model']:
            source_cameras = choose_views(img_ids, frames.camera)
    else:
        sel_frames = FrameData.collate([dataset[idx] for idx in arg.source_ids]).to(device)
        sel_images, sel_fgs = sel_frames.image_rgb, sel_frames.fg_probability
        if model_type in ['pixel_nerf_model', 'grf_model']:
            source_cameras = sel_frames.camera
    print(f'Recon with {len(sel_images)} images.')
    # resize
    sel_images = F.interpolate(sel_images, (128, 128), mode='bilinear', align_corners=False)
    sel_fgs = F.interpolate(sel_fgs, (128, 128), mode='bilinear', align_corners=False)
    source_images = mask_image(sel_images, sel_fgs, threshold=cfg.dataset.mask_thr)

    # # eval cameras
    # cam_ids = random.sample(list(range(len(sel_ids))), min(len(sel_ids), arg.n_eval_cams))
    # test_cameras, test_fgs = choose_views(cam_ids, frames.camera, frames.fg_probability)
    #
    # # transform code
    # if model_type in ['sim_nerf_model', 'hyper_nerf_model']:
    #     transform_code = model.model.transforms.transforms[arg.seq_name]

    # load cameras from template
    temp_seq = _TEMPLATES['hydrant']
    temp_frames = FrameData.collate([dataset[idx] for idx in seq_to_idx[temp_seq]]).to(device)

    test_cameras = generate_eval_video_cameras(
        ref_cameras=temp_frames.camera,
        n_eval_cams=arg.n_eval_cams,
        device=device,
        up=None,
        scene_center=(-2.0, 0.0, 0.0),
        # focal_scale=0.95
    )
    transform_code = None

    # produce images
    with torch.no_grad():
        with Timer(quiet=False):
            if model_type in ['pixel_nerf_model', 'grf_model']:
                outputs = pixelnerf_grf_forward(model, test_cameras, source_images, source_cameras)
            # elif model_type == 'sim_nerf_model':
            #     outputs = simnerf_forward(model, test_cameras, source_images, transform_code,
            #                               (not arg.disable_deformation), (not arg.disable_specular))
            elif model_type == 'scene_model':
                outputs = scenenerf_forward(model, test_cameras)
            elif model_type == 'hyper_nerf_model':
                source_images = prepare_input(source_images, sel_fgs, cfg.dataset.mask_thr)
                outputs = hypersim_forward(model, test_cameras, source_images, transform_code,
                                           (not arg.disable_deformation), (not arg.disable_specular))
            else:
                raise ValueError(f'Unknown model type {cfg.model.type}.')

    # generate outputs
    keywords = ['depth_fine', 'rgb_fine']
    masks = None
    for keyword in keywords:
        data = torch.chunk(outputs[keyword], arg.n_eval_cams, dim=0)
        data = [d.squeeze(0) for d in data]
        # fgs = [fg.permute(1, 2, 0) for fg in test_fgs]
        # make directory
        out_dir = osp.join(arg.output_dir, arg.config_name, datetime.datetime.now().strftime('%y%m%d%H%M'), keyword)
        os.makedirs(out_dir, exist_ok=True)
        # save images
        print(f'Saving images to {out_dir}...')
        image_mode = keyword.split('_')[0]
        if image_mode == 'rgb':
            # mask background
            data = [(1.0 - m) * 1.0 + m * d for m, d in zip(masks, data)]
            data = tensors2images(data, 'HWC', False)
            # save images
            for idx, img in enumerate(data):
                save_image(img, osp.join(out_dir, '{:05}.png'.format(idx)))
        elif image_mode == 'depth':
            masks = depth_to_mask(data, min_depth=0.01, max_depth=32.0)
            # masks = [(mask + fg) / 2.0 for mask, fg in zip(masks, fgs)]
            # masks = fgs
            # to numpy
            data = [d.squeeze().cpu().numpy() for d in data]
            # # save numpy
            # for idx, img in enumerate(data):
            #     np.save(osp.join(out_dir, '{:05}.npy'.format(idx)), img)
            # save
            for idx, img in enumerate(data):
                plt.imsave(osp.join(out_dir, '{:05}.png'.format(idx)), img, cmap='gray')
        else:
            raise ValueError(f'Unsupported image mode: {image_mode}.')

    print('Done.')


if __name__ == '__main__':
    recon()
