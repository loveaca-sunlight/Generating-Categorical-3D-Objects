import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import dataset_zoo, dataloader_zoo
from models import ReconstructionModel


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
    args = parse_args()
    configs = Config.fromfile(osp.join('configs/', f'{args.config_name}.yaml'))

    # dataset
    data = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=configs.dataset.root,
        category=configs.dataset.category,
        mask_images=True,
    )

    # dataloader
    loader = dataloader_zoo(
        datasets=data,
        dataset_name='co3d_multisequence',
        batch_size=configs.sequences_per_batch * configs.frames_per_sequence,
        num_workers=configs.num_workers,
        dataset_len=configs.dataset.batches_per_epoch,
        images_per_seq_options=[configs.frames_per_sequence]
    )
    train_loader, val_loader, test_loader = [loader[label] for label in ['train', 'val', 'test']]

    # model
    model = ReconstructionModel(configs, list(data['train'].seq_to_idx.keys()))

    # define trainer
    work_dir = osp.join(configs.work_dir, args.config_name)
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          save_weights_only=True,
                                          save_top_k=-1,
                                          filename='checkpoint_{epoch}',
                                          every_n_epochs=configs.epochs_per_checkpoint)
    trainer = Trainer(
        default_root_dir=work_dir,
        gpus=1,
        max_epochs=configs.max_epochs,
        callbacks=[checkpoint_callback]
    )

    # training
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
