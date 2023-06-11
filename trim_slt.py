import os.path as osp
import mmcv
import torch

from models import SltNerfModel, SltbNerfModel
from argparse import ArgumentParser


# name_idx_mapping path
_N2I_MAPPING_PATH = ''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )

    return parser.parse_args()

def main():
    arg = parse_args()
    # configurations
    _CONFIG_NAME = arg.config_name
    # config name

    # checkpoint path
    _CHECKPOINT_PATH = f'checkpoints/{_CONFIG_NAME}/last.ckpt'
    cfg = mmcv.Config.fromfile(osp.join('configs/', f'{_CONFIG_NAME}.yaml'))

    # search for sequence name data
    name_file = osp.join(osp.dirname(_CHECKPOINT_PATH), 'sequence_names.txt')
    if osp.exists(name_file):
        sequence_names = mmcv.list_from_file(name_file)
        print(f'Load sequence names from {name_file}.')
    else:
        sequence_names = list(torch.load(_N2I_MAPPING_PATH).keys())
        print(f'Load sequence names from {_N2I_MAPPING_PATH}.')

    # create model
    if cfg.model.implicit_function.type == 'slt_mlp':
        model = SltNerfModel(cfg, sequence_names)
    elif cfg.model.implicit_function.type == 'slt_mlp_b':
        model = SltbNerfModel(cfg, sequence_names)
    else:
        print("Wrong function please check config.yaml")
        return
    state_dict = torch.load(_CHECKPOINT_PATH, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
    transform_dict = model.model.transforms

    # print scores
    for name, multiplexes in transform_dict.multiplexes.items():
        print(f'{name}: {multiplexes.scores}.')

    # get best transforms
    best_transforms = {
        name: multiplexes.best_transform()
        for name, multiplexes in transform_dict.multiplexes.items()
    }

    # save
    torch.save(best_transforms, osp.join(f'checkpoints/{_CONFIG_NAME}', 'best.pth'))
    print('The best transforms are saved.')


if __name__ == '__main__':
    main()
