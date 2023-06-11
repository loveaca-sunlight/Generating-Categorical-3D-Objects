import os
import os.path as osp
import typing
from argparse import ArgumentParser

import imageio
import torch
from mmcv import Config
from tqdm import tqdm, trange

from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS
from dataset.dataloader_zoo import dataloader_zoo
from evaluation.util import save_image, tensors2images, tensors2depths, \
    generate_circular_cameras
from models import DefNerfModel
from models.util import mask_image
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.renderer.cameras import CamerasBase
from tools.camera_utils import select_cameras
from torch.optim import Adam
from renderers.util import zero_param


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
        '--max_iters',
        type=int,
        default=200
    )

    return parser.parse_args()


def split_views(images: torch.Tensor, fg_probabilities: torch.Tensor, cameras: CamerasBase):
    source_images = images[1:, :, :, :]
    source_probabilities = fg_probabilities[1:, :, :, :]
    source_cameras = select_cameras(cameras, list(range(1, cameras.R.shape[0])))

    target_images = images[: 1, :, :, :]
    target_probabilities = fg_probabilities[: 1, :, :, :]
    target_cameras = select_cameras(cameras, 0)

    return source_images, source_probabilities, source_cameras, target_images, target_probabilities, target_cameras


def reset_embeddings(parameter: torch.nn.Parameter, distribution: torch.distributions.Distribution, sample_size):
    parameter.data.copy_(distribution.sample(sample_size))


def reset_parameters(model: DefNerfModel, shape_code_dim: int, color_code_dim: int):
    normal = torch.distributions.Normal(0, 1)
    model.model.latent_embedding.reset_parameters(
        funcs={'shape': reset_embeddings},
        distribution=normal,
        sample_size=(1, shape_code_dim)
    )
    model.model.latent_embedding.reset_parameters(
        funcs={'color': reset_embeddings},
        distribution=normal,
        sample_size=(1, color_code_dim)
    )
    model.model.rigid_embedding.reset_parameters(
        funcs=zero_param
    )


def adapt(model: DefNerfModel, source_images: torch.Tensor, source_probabilities: torch.Tensor,
          source_cameras: torch.Tensor, max_iters: int, shape_code_dim: int, color_code_dim: int,
          latent_code_reg_coef: float, rigid_reg_coef: float, deformation_reg_coef: float,
          deformation_loss_coef: float, weights_reg_coef: float):
    # initialize the parameters of embeddings
    reset_parameters(model, shape_code_dim, color_code_dim)

    # unfreeze
    model.model.latent_embedding.unfreeze()
    model.model.rigid_embedding.unfreeze()
    model.train()

    # optimizers
    optimizer = Adam(model.model.embeddings_parameters(), lr=0.001)

    # begin iter
    pbar = trange(max_iters)
    for idx in pbar:
        pbar.set_description(f'Iter {idx + 1}')

        _, metrics = model(
            target_camera=source_cameras,
            target_image=source_images,
            target_fg_probability=source_probabilities,
            target_mask=source_probabilities,
            shape_sequence_name='seq',
            color_sequence_name=None,
            enable_deformation=True
        )

        # get metrics
        mse_coarse, mse_fine = metrics['mse_coarse'], metrics['mse_fine']
        shape_code_reg = metrics['shape_code_reg']
        color_code_reg = metrics['color_code_reg']
        rigid_code_reg = metrics['rigid_code_reg']
        deformation_reg_coarse, deformation_reg_fine = metrics['deformation_reg_coarse'], \
                                                       metrics['deformation_reg_fine']
        deformation_loss_coarse, deformation_loss_fine = metrics['deformation_loss_coarse'], \
                                                         metrics['deformation_loss_fine']
        weights_reg_coarse, weights_reg_fine = metrics['weights_reg_coarse'], metrics['weights_reg_fine']

        # loss
        loss = mse_coarse + mse_fine \
               + latent_code_reg_coef * (shape_code_reg + color_code_reg) \
               + rigid_reg_coef * rigid_code_reg \
               + deformation_reg_coef * (deformation_reg_coarse + deformation_reg_fine) \
               + deformation_loss_coef * (deformation_loss_coarse + deformation_loss_fine) \
               + weights_reg_coef * (weights_reg_coarse + weights_reg_fine)

        pbar.set_postfix_str("loss={:05f}".format(loss.item()))

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test():
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
    train_set, test_set = datasets['train'], datasets['test']

    test_loader = dataloader_zoo(
        datasets=datasets,
        dataset_name='co3d_multisequence',
        batch_size=1,
        num_workers=1
    )['test']

    # sequence data
    seq_to_idx = train_set.seq_to_idx
    all_sequences = list(seq_to_idx.keys())
    print(f'Sequence Number: {len(all_sequences)}.')

    # model
    model = DefNerfModel(cfg, all_sequences)
    state_dict = torch.load(arg.checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.freeze()

    # reset embeddings
    shape_code_dim = cfg.model.implicit_function.shape_code_dim
    color_code_dim = cfg.model.implicit_function.color_code_dim
    rigid_code_dim = (6 if cfg.model.implicit_function.with_translation else 3)
    model.model.reset_embeddings(
        name_idx_mapping={'seq': 0},
        shape_code_dim=shape_code_dim,
        color_code_dim=color_code_dim,
        rigid_code_dim=rigid_code_dim,
        device=device
    )

    # writer
    writer = SummaryWriter(osp.join(arg.output_dir, arg.config_name))

    # begin testing
    mask_thr = cfg.dataset.mask_thr
    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        # get data
        images = batch_data.image_rgb
        cameras = batch_data.camera
        fg_probability = batch_data.fg_probability

        # write batch size
        writer.add_scalar('val/samples', images.shape[0], global_step=batch_idx)

        # to device
        images = images.to(device)
        cameras.to(device)
        fg_probability = fg_probability.to(device)

        # mask image
        images = mask_image(images, fg_probability, mask_thr)

        # split views
        source_images, source_probabilities, source_cameras, target_images, target_probabilities, target_cameras = \
            split_views(images, fg_probability, cameras)

        # adapt model
        adapt(model, source_images, source_probabilities, source_cameras, arg.max_iters, shape_code_dim, color_code_dim,
              cfg.loss.latent_code_reg_coef, cfg.loss.rigid_reg_coef, cfg.loss.deformation_reg_coef,
              cfg.loss.deformation_loss_coef, cfg.loss.weights_reg_coef)

        # freeze
        model.freeze()

        # produce outputs
        out, metrics = model(
            target_camera=target_cameras,
            target_image=target_images,
            target_fg_probability=target_probabilities,
            target_mask=target_probabilities,
            shape_sequence_name='seq',
            color_sequence_name=None,
            enable_deformation=True
        )

        # write to log
        for k in ['rgb_coarse', 'rgb_fine', 'rgb_gt']:
            img = out[k]
            writer.add_images(f'val_{batch_idx}/{k}', torch.clamp(img, min=0.0, max=1.0), dataformats='NHWC')
        for k in ['depth_coarse', 'depth_fine']:
            depth = out[k]
            depth = depth / (depth.max() + 1.0e-8)
            writer.add_images(f'val_{batch_idx}/{k}', depth, dataformats='NHWC')
        for k in ['mse_coarse', 'mse_fine', 'psnr_coarse', 'psnr_fine']:
            writer.add_scalar(f'val/{k}', metrics[k], global_step=batch_idx)

        # template
        out, _ = model(
            target_camera=target_cameras,
            target_image=None,
            target_fg_probability=None,
            target_mask=None,
            shape_sequence_name='seq',
            color_sequence_name=None,
            enable_deformation=False
        )

        for k in ['depth_coarse', 'depth_fine']:
            depth = out[k]
            depth = depth / (depth.max() + 1.0e-8)
            writer.add_images(f'val_{batch_idx}/template_{k}', depth, dataformats='NHWC')


if __name__ == '__main__':
    test()
