import os.path as osp
from argparse import ArgumentParser

import torch
from mmcv import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import dataset_zoo, dataloader_zoo
from dataset.dataset_zoo import DATASET_CONFIGS
from models import SltNerfModel,SltbNerfModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )

    return parser.parse_args()


def main():
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yaml'))

    # dataset
    DATASET_CONFIGS['default']['image_width'] = cfg.width
    DATASET_CONFIGS['default']['image_height'] = cfg.height
    # depth data is not needed
    DATASET_CONFIGS['default']['load_depths'] = False
    DATASET_CONFIGS['default']['load_depth_masks'] = False

    data = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
        test_on_train=True,
        load_key_points=True
    )

    # dataloader
    loader = dataloader_zoo(
        datasets=data,
        dataset_name='co3d_multisequence',
        batch_size=cfg.batch_size,
        batch_size_val=cfg.batch_size_val,
        num_workers=cfg.num_workers,
        dataset_len=cfg.dataset.dataset_len,
        dataset_len_val=cfg.dataset.dataset_len_val,
        curriculum_config=cfg.dataset.get('curriculum_config', None),
        val_follows_train=cfg.dataset.get('val_follows_train', False)
    )
    train_loader, val_loader = [loader[key] for key in ['train', 'val']]

    # model
    sequences = list(data['train'].seq_annots.keys())
    resume_path = cfg.get('resume_path', None)
    if cfg.model.implicit_function.type == 'slt_mlp':
        model = SltNerfModel(cfg, sequences, resume=(resume_path is not None))
    elif cfg.model.implicit_function.type == 'slt_mlp_b':
        model = SltbNerfModel(cfg, sequences, resume=(resume_path is not None))
    else:
        print("Wrong function please check config.yaml")
        return
    if resume_path is not None:
        model.load_state_dict(torch.load(resume_path, map_location='cpu')['state_dict'])
    print(f'Resume path: {resume_path}.')

    # define trainer
    work_dir = osp.join(cfg.work_dir, arg.config_name)
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          monitor='val/val_loss',
                                          mode='min',
                                          every_n_epochs=cfg.epochs_per_checkpoint,
                                          save_weights_only=False,  # save all data
                                          save_top_k=5,
                                          save_last=True)
    trainer = Trainer(
        default_root_dir=work_dir,
        gpus=1,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.epochs_per_validation,
        callbacks=[checkpoint_callback],

        track_grad_norm=cfg.get('track_grad_norm', -1)
    )

    # training
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
