import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import dataset_zoo, dataloader_zoo
from dataset.dataset_zoo import DATASET_CONFIGS
from models import PixelNerfModel


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
    print(f'Config name: {arg.config_name}.')

    # dataset
    DATASET_CONFIGS['default']['image_width'] = cfg.width
    DATASET_CONFIGS['default']['image_height'] = cfg.height

    data = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=cfg.dataset.root,
        category=cfg.dataset.category,
        mask_images=False,
    )

    # dataloader
    loader = dataloader_zoo(
        datasets=data,
        dataset_name='co3d_multisequence',
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        images_per_seq_options=[cfg.batch_size],
        dataset_len_val=cfg.dataset.dataset_len_val,
        dataset_len=cfg.dataset.dataset_len
    )
    train_loader, val_loader = [loader[key] for key in ['train', 'val']]

    # model
    model = PixelNerfModel(cfg)

    # define trainer
    work_dir = osp.join(cfg.work_dir, arg.config_name)
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          monitor='val/val_acc',
                                          mode='max',
                                          every_n_epochs=cfg.epochs_per_checkpoint,
                                          save_weights_only=True,
                                          save_top_k=3)
    trainer = Trainer(
        default_root_dir=work_dir,
        gpus=1,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.epochs_per_validation,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=cfg.optimizer.accumulate_grad_batches,
    )

    # training
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
