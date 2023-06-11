import torch
import os
import os.path as osp

from analysis.tools import embeddings_distribution
from models import DefNerfModel
from mmcv import Config
from dataset.dataset_zoo import dataset_zoo, DATASET_CONFIGS


# checkpoint path
_CHECKPOINT_PATH = 'checkpoints/banana_defmlp_sel/epoch=123-step=24799.ckpt'
# config name
_CONFIG_NAME = 'banana_defmlp_sel'


def main():
    cfg = Config.fromfile(osp.join('configs/', f'{_CONFIG_NAME}.yaml'))

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
    state_dict = torch.load(_CHECKPOINT_PATH)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.freeze()

    # draw distribution hist
    for idx in ['color', 'shape']:
        print(f'Index: {idx}:')
        output_dir = osp.join('outputs/', idx)
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        embeddings_distribution(model.model.latent_embedding.embeddings[idx], contents=['mean', 'var'],
                                sample_dims=16, output_dir=output_dir, n_bins=20)

    for idx in ['rigid']:
        print(f'Index: {idx}:')
        output_dir = osp.join('outputs/', idx)
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        embeddings_distribution(model.model.rigid_embedding.embeddings[idx], contents=['min', 'max'],
                                sample_dims=6, output_dir=output_dir, n_bins=None)


if __name__ == '__main__':
    main()
