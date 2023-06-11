import os.path as osp
import os
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import dataset_zoo, dataloader_zoo
from dataset.dataset_zoo import DATASET_CONFIGS
from models import MODELS
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )

    return parser.parse_args()


def main():
    # set seed
    seed_everything(1024)

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
        load_key_points=True,  # to filter out sequences without key point
        limit_to=-1,
        limit_sequences_to=-1,
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
    model = MODELS.build({
        'type': cfg.model.type,
        'cfg': cfg,
        'sequences': sequences,
        'style_data': 'test'
    })
    print(f'Training with model {type(model).__name__}...')

    # get resume path
    resume_path = cfg.get('resume_from', None)
    if resume_path is not None:
        print(f'The training is resumed from {resume_path}.')
    else:
        print(f'The training is from scratch.')

    # define trainer
    work_dir = osp.join(cfg.work_dir, arg.config_name)
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          monitor='train/loss',
                                          mode='max',
                                          every_n_epochs=cfg.epochs_per_checkpoint,
                                          save_weights_only=False,  # save all data
                                          save_top_k=4,
                                          save_last=True
                                          )
    trainer = Trainer(
        default_root_dir=work_dir,
        gpus=1,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.epochs_per_validation,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=cfg.optimizer.accumulate_grad_batches,
        gradient_clip_val=cfg.get('gradient_clip_val', 0),

        track_grad_norm=cfg.get('track_grad_norm', -1),
        detect_anomaly=False,
        log_every_n_steps=1,
    )

    # training
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)


if __name__ == '__main__':
    main()
