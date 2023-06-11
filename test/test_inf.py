"""
Test InfNerf
"""
import os.path as osp
import random
from argparse import ArgumentParser

import torch
from mmcv import Config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataset.dataloader_zoo import dataloader_zoo
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from dataset.utils import DATASET_TYPE_TRAIN, DATASET_TYPE_KNOWN
from models import InfNerfModel
from models.util import mask_image, choose_views
from modules.multiplex import TransformMultiplex
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
        '--output_dir',
        type=str,
        default='outputs/'
    )
    parser.add_argument(
        '--save_params',
        action='so'
    )

    return parser.parse_args()


def adapt_transform_multiplex(cfg, model: InfNerfModel, src_images, src_cameras, src_fg, src_mask,
                              n_iters: int = 100):
    max_source_views = 5

    # initialize a multiplex
    n_ways = 4
    multiplex = TransformMultiplex(n_ways, beta=0.9)
    multiplex.to(src_images.device)

    # optimizer
    optimizer = torch.optim.SGD(multiplex.parameters(), lr=0.02, momentum=0.9)

    # adapt
    process = trange(n_iters, desc='Adapting transforms...')
    transforms = multiplex()

    # encoder shape and color code first
    with torch.no_grad():
        shape_code, color_code = model.model.encode_codes(src_images)

    for _ in process:
        losses = []

        # choose source views to reduce time consume
        n_sources = src_images.shape[0]
        if n_sources > max_source_views:
            sel_ids = random.sample(list(range(n_sources)), max_source_views)
            sel_imgs, sel_cams, sel_fg, sel_mask = choose_views(sel_ids, src_images, src_cameras, src_fg, src_mask)
        else:
            sel_imgs, sel_cams, sel_fg, sel_mask = src_images, src_cameras, src_fg, src_mask

        # for each transform
        for idx in range(n_ways):
            transform_code = transforms[(idx, ), :]

            # forward and get loss
            _, metrics = model(
                target_camera=sel_cams,
                target_image=sel_imgs,
                target_fg_probability=sel_fg,
                target_mask=(sel_fg if cfg.model.mask_image else sel_mask),
                source_image=src_images,
                sequence_name=None,
                enable_deformation=True,
                transform_code=transform_code,
                shape_code=shape_code,
                color_code=color_code
            )

            # mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
            weights_reg_coarse, weights_reg_fine = metrics['weights_reg_coarse'], metrics['weights_reg_fine']
            # use same weight
            curr_loss = (weights_reg_coarse + weights_reg_fine)  #  + (mse_coarse + mse_fine)
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

        # show message
        process.set_postfix_str(f'loss={loss.detach().item():.05f}.')

    tqdm.write(f'Scores: {multiplex.scores}.')
    return multiplex.best_transform().detach(), shape_code, color_code


def adapt_all(cfg, model: InfNerfModel, src_images, src_cameras, src_fg, src_mask,
              n_iters_transform: int = 100, n_iters_code: int = 100):
    # adapt transform first
    t_code, s_code, c_code = adapt_transform_multiplex(
        cfg,
        model,
        src_images,
        src_cameras,
        src_fg,
        src_mask,
        n_iters_transform
    )

    if n_iters_code <= 0:
        return t_code, s_code, c_code

    # create parameters
    transform_code = torch.nn.Parameter(torch.zeros_like(t_code), requires_grad=True)
    shape_code = torch.nn.Parameter(torch.zeros_like(s_code), requires_grad=True)
    color_code = torch.nn.Parameter(torch.zeros_like(c_code), requires_grad=True)

    with torch.no_grad():
        transform_code.data.copy_(t_code)
        shape_code.data.copy_(s_code)
        color_code.data.copy_(c_code)

    # optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': [shape_code, color_code], 'lr': 1.0e-5},
            {'params': [transform_code], 'lr': 1.0e-4}
        ],
        lr=0.01
    )

    # adapt
    process = trange(n_iters_code, desc='Adapting codes...')

    for _ in process:
        # forward and get loss
        _, metrics = model(
            target_camera=src_cameras,
            target_image=src_images,
            target_fg_probability=src_fg,
            target_mask=(src_fg if cfg.model.mask_image else src_mask),
            source_image=src_images,
            sequence_name=None,
            enable_deformation=True,
            transform_code=transform_code,
            shape_code=shape_code,
            color_code=color_code
        )

        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
        weights_reg_coarse, weights_reg_fine = metrics['weights_reg_coarse'], metrics['weights_reg_fine']
        deformation_reg_coarse, deformation_reg_fine = metrics['deformation_reg_coarse'], \
                                                       metrics['deformation_reg_fine']
        deformation_loss_coarse, deformation_loss_fine = metrics['deformation_loss_coarse'], \
                                                         metrics['deformation_loss_fine']

        # loss
        loss = mse_coarse + mse_fine \
                + cfg.loss.deformation_reg_coef * (deformation_reg_coarse + deformation_reg_fine) \
                + cfg.loss.deformation_loss_coef * (deformation_loss_coarse + deformation_loss_fine) \
                + cfg.loss.weights_reg_coef * (weights_reg_coarse + weights_reg_fine)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show message
        process.set_postfix_str(f'loss={loss.detach().item():.05f}.')

    return transform_code.detach(), shape_code.detach(), color_code.detach()


def test_inf():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    DATASET_CONFIGS['default']['image_width'] = cfg.width
    DATASET_CONFIGS['default']['image_height'] = cfg.height

    datasets = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
        test_on_train=False
    )
    train_set, eval_set = datasets['train'], datasets['test']

    test_loader = dataloader_zoo(
        datasets=datasets,
        dataset_name='co3d_multisequence',
        batch_size=1
    )['test']

    # sequence data
    seq_to_idx = train_set.seq_to_idx
    all_sequences = list(seq_to_idx.keys())
    print(f'Sequence Number: {len(all_sequences)}.')

    # model
    model = InfNerfModel(cfg, all_sequences)
    state_dict = torch.load(arg.checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    print(f'Model weights are loaded from {arg.checkpoint}.')
    model.to(device)
    freeze(model)
    print('Model weights are frozen.')

    # writer
    writer = SummaryWriter(log_dir=osp.join(arg.output_dir, arg.config_name))

    # test
    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        # to device
        batch_data = batch_data.to(device)

        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        fg_probability = batch_data.fg_probability
        mask_crop = batch_data.mask_crop

        # mask image
        _mask_image = cfg.model.mask_image
        if _mask_image:
            images = mask_image(images, fg_probability, cfg.dataset.mask_thr)

        # check if test image
        assert batch_data.frame_type[0] != f'{DATASET_TYPE_TRAIN}_{DATASET_TYPE_KNOWN}', \
            f'Excepted test sample, but given {batch_data.frame_type[0]}.'

        # number of images
        n_images = images.shape[0]

        # source ids
        source_ids = list(range(1, n_images))

        # choose source data
        src_images, src_cameras, src_fg, src_mask = choose_views(source_ids, images, cameras, fg_probability, mask_crop)

        # get proper transform
        model.train()
        transform_code, shape_code, color_code = adapt_all(
            cfg=cfg,
            model=model,
            src_images=src_images,
            src_cameras=src_cameras,
            src_fg=src_fg,
            src_mask=src_mask,
            n_iters_transform=150,
            n_iters_code=0
        )

        # get predicted outputs and compute losses
        model.eval()
        with torch.no_grad():
            test_out, test_metrics = model(
                target_camera=cameras,
                target_image=images,
                target_fg_probability=fg_probability,
                target_mask=fg_probability if _mask_image else mask_crop,
                source_image=src_images,
                sequence_name=None,
                enable_deformation=True,
                transform_code=transform_code,
                shape_code=shape_code,
                color_code=color_code
            )

        # write result
        for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
            img = test_out[k]
            writer.add_images(f'test_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0), dataformats='NHWC')
        for k in ['depth_coarse', 'depth_fine']:
            depth = test_out[k]
            depth = depth / (depth.max() + 1.0e-8)
            writer.add_images(f'test_{batch_idx}/{k}', depth, dataformats='NHWC')

        val_acc = (test_metrics['psnr_fine'] + test_metrics['psnr_coarse']) / 2.0
        writer.add_scalar('accurate', val_acc, global_step=batch_idx)


if __name__ == '__main__':
    test_inf()
