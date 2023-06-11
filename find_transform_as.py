"""
Find transforms
"""
import os.path as osp
import os
import random
import numpy as np
import mmcv
from argparse import ArgumentParser

import torch
import collections
from mmcv import Config
from pytorch3d.renderer import PerspectiveCameras
from tqdm import tqdm, trange

from dataset.co3d_dataset import FrameData
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from models import MODELS
from models.util import mask_image, choose_views
from modules.multiplex_new import TransformMultiplex
from modules.util import freeze


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
        '--n_iters',
        type=int,
        default=300
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=24
    )
    parser.add_argument(
        '--find_all',
        action='store_true'
    )

    return parser.parse_args()


@torch.no_grad()
def prepare_input(src_img: torch.Tensor, src_fg: torch.Tensor, mask_thr: float):
    fg = src_fg.clone()
    fg[fg < mask_thr] = 0.0
    inputs = torch.cat([src_img, fg], dim=1)  # (n, 4, h, w)
    return inputs


def adapt_transform_multiplex(cfg: dict, model, src_images: torch.Tensor, tgt_images: torch.Tensor,
                              tgt_cameras: PerspectiveCameras, tgt_fg: torch.Tensor, n_iters: int = 200):
    # initialize a multiplex
    n_ways = 4
    multiplex = TransformMultiplex(n_ways, beta=0.9)
    multiplex.to(src_images.device)

    # optimizer
    optimizer = torch.optim.SGD(multiplex.parameters(), lr=0.05, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[int(n_iters * 0.7), int(n_iters * 0.85), int(n_iters * 0.95)],
    #     gamma=0.5
    # )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.994
    )

    # adapt
    process = trange(n_iters, desc='Adapting transforms...', leave=False)
    transforms = multiplex()

    # encoder shape and color code first
    with torch.no_grad():
        shape_code, color_code = model.model.encode_codes(src_images)

    for _ in process:
        losses = []

        # for each transform
        for idx in range(n_ways):
            transform_code = transforms[(idx, ), :]

            # forward and get loss
            _, metrics = model(
                target_camera=tgt_cameras,
                target_image=tgt_images,
                target_fg_probability=tgt_fg,
                target_mask=tgt_fg,
                source_image=None,
                sequence_name=None,
                enable_deformation=True,
                enable_specular=True,
                shape_code=shape_code,
                color_code=color_code,
                transform_code=transform_code
            )

            mse_fine = metrics['mse_fine']
            weights_reg_fine = metrics['weights_reg_fine']
            curr_loss = mse_fine + 0.5 * weights_reg_fine
            losses.append(curr_loss)

        loss = torch.stack(losses)  # (m,)

        # compute normalized scores to each transform
        mp_losses = loss.detach()
        scores = TransformMultiplex.losses_to_scores(mp_losses)
        multiplex.update_scores(scores)

        # mean loss
        loss = loss.mean()

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        # show message
        process.set_postfix_str(f'loss={loss.detach().item():.05f}.')

    tqdm.write(f'Scores: {multiplex.scores}.')
    return multiplex.best_transform().detach()


def find_transform():
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

    datasets = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
        test_on_train=False
    )
    val_set, test_set = datasets['val'], datasets['test']

    # all sequences in test split
    eval_batches = test_set.eval_batches
    all_sequences = [
        test_set[batch[0]].sequence_name
        for batch in eval_batches
    ]
    all_sequences = list(set(all_sequences))
    print(f'Required transforms: {len(all_sequences)}.')

    # sequence to ids
    seq_to_ids = val_set.seq_to_idx

    # model
    cfg.model.weighted_mse = True
    cfg.model.n_rays_per_image = 32  # reduce computation cost
    model = MODELS.build(
        {
            'type': cfg.model.type,
            'cfg': cfg,
            'sequences': None,
            'load_pretrained_weights': True,
            'calibration_mode': True,
        }
    )
    state_dict = torch.load(arg.checkpoint, map_location='cpu')['state_dict']
    state_dict = collections.OrderedDict(
        [
            (k, state_dict[k])
            for k in filter(lambda x: '_grid_raysampler._xy_grid' not in x, state_dict.keys())
        ]
    )  # for checkpoint of old version
    model.load_state_dict(state_dict, strict=True)
    print(f'Model weights are loaded from {arg.checkpoint}.')
    model.to(device)
    freeze(model)  # still train mode

    # set dropout to eval mode
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    print('Model weights are frozen.')

    # load training list
    seq_fp = os.path.join(os.path.dirname(cfg.model.best_transforms), 'sequence_names.txt')
    sequences = mmcv.list_from_file(seq_fp)
    print(f'Sequence data is found at {seq_fp}.')

    # find transform
    transforms = {}
    batch_size = arg.batch_size
    n_sources = 16
    for seq_name in tqdm(all_sequences):

        # check if transform has been computed
        if (not arg.find_all) and (seq_name in sequences):
            transforms[seq_name] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float).view(1, 10)
            tqdm.write(f'Transform of {seq_name} exists, skipping...')
            continue

        tqdm.write(f'Processing transform of sequence {seq_name}...')

        # get batch data
        seq_ids = seq_to_ids[seq_name]
        batch_ids = np.random.choice(seq_ids, batch_size, replace=(batch_size > len(seq_ids))).tolist()
        frames = [
            val_set[idx]
            for idx in batch_ids
        ]
        batch_data = FrameData.collate(frames)

        # to device
        batch_data = batch_data.to(device)

        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        fg_probability = batch_data.fg_probability

        # mask image
        _mask_image = cfg.model.mask_image
        if _mask_image:
            images = mask_image(images, fg_probability, cfg.dataset.mask_thr)

        # number of images
        n_images = images.shape[0]

        # source ids
        source_ids = random.sample(list(range(n_images)), n_sources)

        # choose source data
        src_images, src_fgs = choose_views(source_ids, images, fg_probability)

        # prepare input
        src_inputs = prepare_input(src_images, src_fgs, cfg.dataset.mask_thr)

        # get proper transform
        transform_code = adapt_transform_multiplex(
            cfg=cfg,
            model=model,
            src_images=src_inputs,
            tgt_images=images,
            tgt_cameras=cameras,
            tgt_fg=fg_probability,
            n_iters=arg.n_iters
        )

        transforms[seq_name] = transform_code.cpu()

    # save transforms
    fn = osp.join('checkpoints/', arg.config_name, 'sequence_transform.pt')
    torch.save(transforms, fn)


if __name__ == '__main__':
    find_transform()
