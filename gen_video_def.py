"""
Generate video for sepnerf
"""
import os
import os.path as osp
from argparse import ArgumentParser

import imageio
import torch
from mmcv import Config
from tqdm import tqdm

from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from evaluation.util import save_image, tensors2images, tensors2depths, \
    generate_circular_cameras
from models import DefNerfModel


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
        '--shape_ids',
        type=int,
        default=None,
        nargs='+'
    )
    parser.add_argument(
        '--color_ids',
        type=int,
        default=None,
        nargs='+'
    )
    parser.add_argument(
        '--shape_names',
        type=str,
        default=None,
        nargs='+'
    )
    parser.add_argument(
        '--color_names',
        type=str,
        default=None,
        nargs='+'
    )
    parser.add_argument(
        '--keywords',
        type=str,
        nargs='+',
        default=['rgb_fine'],
        choices=['rgb_fine', 'rgb_coarse', 'depth_fine', 'depth_coarse']
    )
    parser.add_argument(
        '--ref_camera_idx',
        type=int,
        default=0
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/'
    )
    parser.add_argument(
        '--n_eval_cams',
        type=int,
        default=100
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
        '--up_scale',
        type=float,
        default=1.0
    )

    return parser.parse_args()


def gen_video_def():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))
    print(f'Keywords: {arg.keywords}.')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    DATASET_CONFIGS['default']['image_width'] = cfg.width
    DATASET_CONFIGS['default']['image_height'] = cfg.height

    eval_set = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
        test_on_train=True
    )['test']

    # sequence data
    seq_to_idx = eval_set.seq_to_idx
    all_sequences = list(seq_to_idx.keys())
    print(f'Sequence Number: {len(all_sequences)}.')

    # model
    model = DefNerfModel(cfg, all_sequences)
    state_dict = torch.load(arg.checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ids to names
    if arg.shape_ids is not None:
        shape_seq_names = [all_sequences[idx] for idx in arg.shape_ids]
        color_seq_names = [all_sequences[idx] for idx in arg.color_ids]
        assert len(shape_seq_names) == len(color_seq_names), 'The length of shape_ids and color_ids is not same.'
    elif arg.shape_names is not None:
        shape_seq_names = arg.shape_names
        color_seq_names = arg.color_names
    else:
        raise ValueError('Neither idx nor name is specified.')
    assert len(shape_seq_names) == len(color_seq_names), 'The length of shape_ids and color_ids is not same.'

    for shape_seq_name, color_seq_name in zip(shape_seq_names, color_seq_names):
        # show processed sequence
        print(f'Shape: {shape_seq_name}, Color: {color_seq_name}.')

        # eval cameras
        eval_cameras = generate_circular_cameras(
            train_cameras=[eval_set[idx].camera for idx in
                           seq_to_idx[(shape_seq_name, color_seq_name)[arg.ref_camera_idx]]],
            n_eval_cams=arg.n_eval_cams,
            up_scale=arg.up_scale
        )

        # process
        outputs = []
        with torch.no_grad():
            for cam in tqdm(eval_cameras):
                cam.to(device)
                out, _ = model(
                    target_camera=cam,
                    target_image=None,
                    target_fg_probability=None,
                    target_mask=None,
                    shape_sequence_name=shape_seq_name,
                    color_sequence_name=color_seq_name,
                    enable_deformation=(not arg.disable_deformation)
                )
                outputs.append(out)

        # generate outputs
        for keyword in arg.keywords:
            data = [d[keyword][0] for d in outputs]
            # make directory
            out_dir = osp.join(arg.output_dir, arg.config_name, keyword)
            os.makedirs(out_dir, exist_ok=True)
            # save images
            tqdm.write(f'Saving images to {out_dir}...')
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
            tqdm.write(f'Saving video to {out_dir}...')
            imageio.mimwrite(osp.join(out_dir, f'{shape_seq_name}-{color_seq_name}-{keyword}.gif'),
                             data, duration=0.05)  # 20 frames per second

    tqdm.write('Done.')


if __name__ == '__main__':
    gen_video_def()
