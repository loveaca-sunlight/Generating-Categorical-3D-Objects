"""
Calculate metrics
"""
import os.path as osp

from dataset.dataset_zoo import dataset_zoo

_DATA_ROOT = r'F:\Dataset\CO3D'
_CATEGORY = 'mouse'
_OUTPUT_DIR = './files/'


def gen_list():

    # dataset
    datasets = dataset_zoo(
        dataset_name='co3d_multisequence',
        dataset_root=_DATA_ROOT,
        category='banana',  # arg.category,
        mask_images=True,
        test_on_train=False
    )

    # get eval batches
    eval_batches = datasets['test'].eval_batches

    # filter
    samples = set()
    result = []

    for idx, batch in enumerate(eval_batches):
        sample = tuple(batch[: 2])

        if sample not in samples:
            result.append(idx)
            samples.add(sample)

    print(f'Length after filtering: {len(result)}.')

    # write to file
    to_write = [f'{line}\n' for line in result]
    with open(osp.join(_OUTPUT_DIR, f'{_CATEGORY}_list.txt'), 'w') as f:
        f.writelines(to_write)


if __name__ == '__main__':
    gen_list()
