"""
Generate video for hypersim
"""
import collections
import os
import os.path as osp
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mmcv
import torch
import torch.nn.functional as F
from mmcv import Config

from dataset.co3d_dataset import FrameData
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from evaluation.util import save_image, tensors2images, depth_to_mask
from evaluation.cam_util import generate_eval_video_cameras
from models import HyperNeRFModel
from models.util import choose_views, mask_image
from tools.utils import Timer


# render size, (w, h)
_RENDER_SIZE = (128, 128)
# factors
_SHAPE_FACTORS = [1.0, 0.75, 0.5, 0.25, 0.0]
_COLOR_FACTORS = [1.0, 1.0, 1.0, 1.0, 0.0]


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
        'seq_names',
        type=str,
        nargs=2,
        help='Name of sequences.'
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


@torch.no_grad()
def prepare_input(src_img: torch.Tensor, src_fg: torch.Tensor, mask_thr: float):
    fg = src_fg.clone()
    fg[fg < mask_thr] = 0.0
    inputs = torch.cat([src_img, fg], dim=1)  # (n, 4, h, w)
    return inputs


def interpolation():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))
    # change size
    cfg.width, cfg.height = _RENDER_SIZE
    cfg.model.width, cfg.model.height = _RENDER_SIZE

    # change chunk_size_test
    cfg.model.chunk_size_test = arg.n_eval_cams * 100

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    sequence_names = mmcv.list_from_file(osp.join(osp.dirname(cfg.model.best_transforms), 'sequence_names.txt'))
    model = HyperNeRFModel(cfg, sequences=sequence_names, load_pretrained_weights=False)
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
    )['test']

    # select sequence
    seq_to_idx = dataset.seq_to_idx
    shape_codes = []
    color_codes = []

    # ref_ids = None
    ref_frames = None
    with torch.no_grad():
        for i, seq_name in enumerate(arg.seq_names):
            sel_ids = seq_to_idx[seq_name]
            frames = FrameData.collate([dataset[idx] for idx in sel_ids]).to(device)
            if i == 0:
                # ref_ids = sel_ids
                ref_frames = frames

            img_ids = random.sample(list(range(len(sel_ids))), 5)
            sel_images, sel_fgs = choose_views(img_ids, frames.image_rgb, frames.fg_probability)
            # resize
            sel_images = F.interpolate(sel_images, (128, 128), mode='bilinear', align_corners=False)
            sel_fgs = F.interpolate(sel_fgs, (128, 128), mode='bilinear', align_corners=False)
            source_images = mask_image(sel_images, sel_fgs, threshold=cfg.dataset.mask_thr)
            # input
            src_input = prepare_input(source_images, sel_fgs, cfg.dataset.mask_thr)
            # encode
            shape_code_, color_code_ = model.model.encode_codes(src_input)
            shape_codes.append(shape_code_)
            color_codes.append(color_code_)
    # eval cameras
    test_cameras = generate_eval_video_cameras(
        ref_frames.camera,
        n_eval_cams=arg.n_eval_cams,
        up=None,
        focal_scale=0.95,
        device=device
    )

    # transform code
    transform_code = model.model.transforms.transforms[arg.seq_names[0]]

    # for each factors
    assert len(_SHAPE_FACTORS) == len(_COLOR_FACTORS)
    for shape_factor, color_factor in zip(_SHAPE_FACTORS, _COLOR_FACTORS):
        print(f'Processing with shape_factor={shape_factor}, color_factor={color_factor}...')
        assert 0.0 <= shape_factor <= 1.0
        assert 0.0 <= color_factor <= 1.0
        shape_code = shape_factor * shape_codes[0] + (1.0 - shape_factor) * shape_codes[1]
        color_code = color_factor * color_codes[0] + (1.0 - color_factor) * color_codes[1]

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
                    transform_code=transform_code,
                    shape_code=shape_code,
                    color_code=color_code
                )

        # generate outputs
        keywords = ['depth_fine', 'rgb_fine']
        masks = None
        for keyword in keywords:
            data = torch.chunk(outputs[keyword], arg.n_eval_cams, dim=0)
            data = [d.squeeze(0) for d in data]
            # make directory
            out_dir = osp.join(arg.output_dir, arg.config_name, f'{shape_factor}-{color_factor}/', keyword)
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
                masks = depth_to_mask(data, min_depth=0.01, max_depth=24.0)
                # to numpy
                data = [d.squeeze().cpu().numpy() for d in data]
                # # save numpy
                # for idx, img in enumerate(data):
                #     np.save(osp.join(out_dir, '{:05}.npy'.format(idx)), img)
                # save
                for idx, img in enumerate(data):
                    plt.imsave(osp.join(out_dir, '{:05}.png'.format(idx)), img, cmap='jet')
            else:
                raise ValueError(f'Unsupported image mode: {image_mode}.')

    print('Done.')


if __name__ == '__main__':
    interpolation()
