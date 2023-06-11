"""
Generate video for simnerf
"""
import glob
import os
import os.path as osp
import numpy as np
import datetime
import mmcv

from PIL import Image
from argparse import ArgumentParser
from collections import OrderedDict

import imageio
import torch
from mmcv import Config

from evaluation.util import save_image, tensors2images, tensors2depths, \
    generate_inference_cameras
from models import SimNerfModel
from tools.utils import Timer


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
        'shape_dir',
        type=str,
        help='Directory of input files to encode shape.'
    )
    parser.add_argument(
        '--color_dir',
        type=str,
        default=None,
        help='Directory of input files to encode color.'
    )
    parser.add_argument(
        '--keywords',
        type=str,
        nargs='+',
        default=['rgb_fine'],
        choices=['rgb_fine', 'rgb_coarse', 'depth_fine', 'depth_coarse']
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
        '--save_images',
        action='store_true'
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


def _load_image(path):
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    # resize
    im = mmcv.imresize(im, (128, 128))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im


def _prepare_images(path):
    files = glob.glob(path)
    print(f'Found images: {",".join(files)}.')
    images = [_load_image(fn) for fn in files]
    images = [torch.from_numpy(img) for img in images]
    images = torch.stack(images, dim=0)
    return images


def gen_video():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))
    print(f'Keywords: {arg.keywords}.')

    # change chunk_size_test
    cfg.model.chunk_size_test = arg.n_eval_cams * 100

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = SimNerfModel(cfg, sequences=None, load_pretrained_weights=False)
    state_dict = torch.load(arg.checkpoint)['state_dict']
    state_dict = OrderedDict(
        [
            (k, state_dict[k])
            for k in filter(lambda x: '_grid_raysampler._xy_grid' not in x, state_dict.keys())
        ]
    )
    state_dict = OrderedDict(filter(lambda x: 'transforms.' not in x[0], state_dict.items()))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.freeze()
    print(f'Weights are loaded from {arg.checkpoint}.')

    # read source images
    shape_images = _prepare_images(osp.join(arg.shape_dir, '*.jpg')).to(device)
    if arg.color_dir is not None:
        color_images = _prepare_images(osp.join(arg.color_dir, '*.jpg')).to(device)

    # transform
    transform = torch.zeros(1, 6, device=device)
    if cfg.model.implicit_function.full_transform:
        transform = torch.cat([transform, torch.ones(1, 1, device=device)], dim=1)

    # transform = torch.tensor([1.9003, 0.0181, 2.7111, -0.2001,  -0.4544,  -0.5182,  0.9435], device=device).view(1, 7)

    # encode
    with torch.no_grad():
        shape_code, color_code = model.model.encode_codes(shape_images)
        if arg.color_dir is not None:
            _, color_code = model.model.encode_codes(color_images)

    # generate cameras
    test_cameras = generate_inference_cameras(
        focal_length=(5.0, 5.0),
        principal_point=(0.0, 0.0),
        n_eval_cams=arg.n_eval_cams,
        high=24.0,
        radius=12.0
    )
    test_cameras.to(device)

    # produce images
    with torch.no_grad():
        with Timer(quiet=False):
            outputs, _ = model(
                target_camera=test_cameras,
                target_image=None,
                target_fg_probability=None,
                target_mask=None,
                source_image=None,
                sequence_name=None,
                enable_deformation=(not arg.disable_deformation),
                enable_specular=(not arg.disable_specular),
                shape_code=shape_code,
                color_code=color_code,
                transform_code=transform
            )

    # generate outputs
    for keyword in arg.keywords:
        data = torch.chunk(outputs[keyword], arg.n_eval_cams, dim=0)
        data = [d.squeeze(0) for d in data]
        # make directory
        out_dir = osp.join(arg.output_dir, arg.config_name, datetime.datetime.now().strftime('%y%m%d%H%M'), keyword)
        os.makedirs(out_dir, exist_ok=True)
        # save images
        print(f'Saving images to {out_dir}...')
        image_mode = keyword.split('_')[0]
        if image_mode == 'rgb':
            data = tensors2images(data, 'HWC', False)
        elif image_mode == 'depth':
            data = tensors2depths(data, 'HWC')
        else:
            raise ValueError(f'Unsupported image mode: {image_mode}.')
        if arg.save_images:
            for idx, img in enumerate(data):
                save_image(img, osp.join(out_dir, '{:05}.png'.format(idx)))

        # save gif
        print(f'Saving video to {out_dir}...')
        imageio.mimwrite(osp.join(out_dir, f'video-{keyword}.gif'),
                         data, duration=0.05)  # 20 frames per second

    print('Done.')


if __name__ == '__main__':
    gen_video()
