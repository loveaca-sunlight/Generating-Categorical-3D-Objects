"""
Generate video for simnerf
"""
import datetime
import glob
import os
import os.path as osp
import random

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmcv import Config

from dataset.co3d_dataset import FrameData
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from evaluation.cam_util import generate_eval_video_cameras
from evaluation.util import generate_inference_cameras
from evaluation.util import save_image, tensors2images, depth_to_mask, tensors2depths
from models.util import choose_views, mask_image
from renderers import CompositionRenderer
from tools.utils import Timer

# configs
_RENDER_SIZE = (128, 128)

_SCENE_CONFIG = None  # 'hydrant_scene'
_SCENE_CHECKPOINT = None  # 'checkpoints/hydrant_scene/epoch=179-step=17999.ckpt'
_SCENE_SEQUENCE_NAME = '216_22805_47495'

_OUTPUT_DIR = 'outputs/'
_N_EVAL_CAMS = 36
_SAVE_IMAGES = True
_ENABLE_DEFORMATION = True
_ENABLE_SPECULAR = False

# _DATA_ROOT = "/opt/data/private/CO3D"
# _CATEGORY = 'mouse'
_CONFIG_NAMES = ['banana_simmlp_new', 'mouse_simmlp']  # ['banana_simmlp_new', 'mouse_simmlp']
_CHECKPOINTS = ['checkpoints/banana_simmlp_new/epoch=1089-step=53409.ckpt',
                'checkpoints/mouse_simmlp/epoch=1139-step=108299.ckpt']
# _OBJECT_SEQUENCE_NAMEs = ['30_1102_3037', '375_42399_84966']
# _OBJECT_TRANSFORMs = [[-0.0394,  0.2739,  0.2130, -0.1473, -0.1427,  0.2946,  0.9903],
#                       [0.0675 * 3.5, -0.9836, -0.6841, 0.0, 0.0, -4.0, 0.85]]

_OBJECT_TRANSFORMs = [[3.14, 0, 0, -2, 2, -0.8, 1], [3.14, 0, 0, 0, 0, 0, 1]]


def _load_image(path):
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im


def _load_stack_images(path):
    files = glob.glob(path)
    print(f'Found images: {",".join(files)}.')
    images = [_load_image(fn) for fn in files]
    images = [torch.from_numpy(img) for img in images]
    images = torch.stack(images, dim=0)
    return images


def _load_from_dir(dir_name, device):
    source_images = []
    for dm in os.scandir(dir_name):
        if dm.is_dir():
            images = []
            for di in os.scandir(dm.path):
                if di.is_dir():
                    images.append(_load_stack_images(osp.join(di.path, '*.jpg')).to(device))
            source_images.append(images)
    return source_images


def composition():
    # configurations
    if _SCENE_CONFIG is not None:
        scene_cfg = Config.fromfile(osp.join('configs/', f'{_SCENE_CONFIG}.yaml'))
    else:
        scene_cfg = None

    cfgs = [
        Config.fromfile(osp.join('configs/', f'{name}.yaml'))
        for name in _CONFIG_NAMES
    ]

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = CompositionRenderer(
        image_size=_RENDER_SIZE,
        n_pts_per_ray=96,
        n_rays_per_image=32,  # ignored
        min_depth=8.0,
        max_depth=8.0,
        chunk_size_test=_N_EVAL_CAMS * 100,
        scene_function_config=(scene_cfg.model.implicit_function if scene_cfg is not None else None),
        implicit_function_configs=[cfg.model.implicit_function for cfg in cfgs],
        deformable_configs=[cfg.model.deformable_config for cfg in cfgs],
        encoder_configs=[cfg.model.encoder_config for cfg in cfgs]
    )
    scene_weights = (torch.load(_SCENE_CHECKPOINT)['state_dict'] if _SCENE_CONFIG is not None else None)
    state_dicts = [
        torch.load(name)['state_dict']
        for name in _CHECKPOINTS
    ]
    model.load_weights(scene_weights, state_dicts)
    model.to(device)
    model.eval()
    print(f'Weights are loaded.')

    # # dataset
    # DATASET_CONFIGS['default']['image_width'] = _RENDER_SIZE[0]
    # DATASET_CONFIGS['default']['image_height'] = _RENDER_SIZE[1]
    # # depth data is not needed
    # DATASET_CONFIGS['default']['load_depths'] = False
    # DATASET_CONFIGS['default']['load_depth_masks'] = False
    # # DATASET_CONFIGS['default']['box_crop'] = False
    #
    # dataset = dataset_zoo(
    #     dataset_name='co3d_multisequence',
    #     dataset_root=_DATA_ROOT,
    #     category=_CATEGORY,
    #     mask_images=False,
    #     test_on_train=True
    # )['val']
    # seq_to_idx = dataset.seq_to_idx

    # # select scene frames
    # scene_ids = seq_to_idx[_SCENE_SEQUENCE_NAME]
    # scene_frames = FrameData.collate([dataset[idx] for idx in scene_ids]).to(device)

    # # process objects
    # object_source_images = []
    # frames_1st = None
    # for idx, obj_seq_name in enumerate(_OBJECT_SEQUENCE_NAMEs):
    #     # select object frames
    #     sel_ids = seq_to_idx[obj_seq_name]
    #     frames = FrameData.collate([dataset[idx] for idx in sel_ids]).to(device)
    #     if idx == 0:
    #         frames_1st = frames
    #
    #     img_ids = random.sample(list(range(len(sel_ids))), 7)
    #     sel_images, sel_fgs = choose_views(img_ids, frames.image_rgb, frames.fg_probability)
    #     # resize
    #     sel_images = F.interpolate(sel_images, (128, 128), mode='bilinear', align_corners=False)
    #     sel_fgs = F.interpolate(sel_fgs, (128, 128), mode='bilinear', align_corners=False)
    #     source_images = mask_image(sel_images, sel_fgs, threshold=0.05)
    #     object_source_images.append(source_images)
    # source_images = [object_source_images]

    # read images from file
    source_images = _load_from_dir('test_images/', device)

    # object transform
    object_transforms = [
        [
            torch.tensor(sub_func, dtype=torch.float32, device=device).view(1, 7)
        ]
        for sub_func in _OBJECT_TRANSFORMs
    ]

    # generate cameras
    # test_camera_ids = random.sample(list(range(len(scene_ids))), min(_N_EVAL_CAMS, len(scene_ids)))
    # test_cameras = choose_views(test_camera_ids, scene_frames.camera)
    # test_cameras = generate_eval_video_cameras(
    #     ref_cameras=frames_1st.camera,
    #     n_eval_cams=_N_EVAL_CAMS,
    #     up=None,
    #     device=device
    # )
    # generate cameras
    test_cameras = generate_inference_cameras(
        focal_length=(5.0, 5.0),
        principal_point=(0.0, 0.0),
        n_eval_cams=_N_EVAL_CAMS,
        high=20.0,
        radius=10.0
    )
    test_cameras.to(device)

    # produce images
    with torch.no_grad():
        with Timer(quiet=False):
            outputs = model(
                target_camera=test_cameras,
                object_images=source_images,
                object_transforms=object_transforms,
                enable_deformation=_ENABLE_DEFORMATION,
                enable_specular=_ENABLE_SPECULAR,
            )

    # generate outputs
    masks = None
    for keyword in ('depth_fine', 'rgb_fine'):
        data = torch.chunk(outputs[keyword], _N_EVAL_CAMS, dim=0)
        data = [d.squeeze(0) for d in data]
        # make directory
        out_dir = osp.join(_OUTPUT_DIR, 'composition/', datetime.datetime.now().strftime('%y%m%d%H%M'), keyword)
        os.makedirs(out_dir, exist_ok=True)
        # save images
        print(f'Saving images to {out_dir}...')
        image_mode = keyword.split('_')[0]
        if image_mode == 'rgb':
            # mask background
            data = [(1.0 - m) * 1.0 + m * d for m, d in zip(masks, data)]
            data = tensors2images(data, 'HWC', False)
            # data = [
            #     d[:, : int(_RENDER_SIZE[1] * 0.7), :]  # 0.567
            #     for d in data
            # ]
        elif image_mode == 'depth':
            masks = depth_to_mask(data, min_depth=0.01, max_depth=100.0)
            data = tensors2depths(data, 'HWC')
        else:
            raise ValueError(f'Unsupported image mode: {image_mode}.')
        if _SAVE_IMAGES:
            for idx, img in enumerate(data):
                save_image(img, osp.join(out_dir, '{:05}.png'.format(idx)))

        # save gif
        print(f'Saving video to {out_dir}...')
        imageio.mimwrite(osp.join(out_dir, f'video-{keyword}.gif'),
                         data, duration=0.05)  # 20 frames per second

    print('Done.')


if __name__ == '__main__':
    composition()
