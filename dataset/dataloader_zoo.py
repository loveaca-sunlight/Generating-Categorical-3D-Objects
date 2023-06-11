# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing

import torch

from modules.curriculum.curriculum_batch_sampler import CurriculumBatchSampler
from .co3d_dataset import FrameData
from modules.scene_batch_sampler import TrainBatchSampler, ValBatchSampler


def dataloader_zoo(
    datasets: typing.Dict[str, torch.utils.data.Dataset],
    dataset_name="co3d_singlesequence",
    batch_size: int = 1,
    batch_size_val: int = None,
    num_workers: int = 0,
    dataset_len: int = 1000,
    dataset_len_val: int = 1,
    curriculum_config: dict = None,
    val_follows_train: bool = False
):
    """
    Returns a set of dataloaders for a given set of datasets.

    Args:
        datasets: A dictionary containing the
            `"dataset_subset_name": torch_dataset_object` key, value pairs.
        dataset_name: The name of the returned dataset.
        batch_size: The size of the batch of the dataloader.
        batch_size_val: The size of the batch of the dataloader when eval.
        num_workers: Number data-loading threads.
        dataset_len: The number of batches in a training epoch.
        dataset_len_val: The number of batches in a validation epoch.
        images_per_seq_options: Possible numbers of images sampled per sequence.
        curriculum_config: config for curriculum sampling
        val_follows_train:
    Returns:
        dataloaders: A dictionary containing the
            `"dataset_subset_name": torch_dataloader_object` key, value pairs.
    """

    if dataset_name not in ["co3d_singlesequence", "co3d_multisequence"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if batch_size_val is None:
        batch_size_val = batch_size

    dataloaders = {}

    train_batch_sampler = None

    if dataset_name in ["co3d_singlesequence", "co3d_multisequence"]:
        for dataset_set, dataset in datasets.items():
            num_samples = {
                "train": dataset_len,
                "val": dataset_len_val,
                "test": None,
            }[dataset_set]

            if dataset_set == "test":
                batch_sampler = dataset.eval_batches
            elif dataset_set == 'train':
                num_samples = len(dataset) if num_samples <= 0 else num_samples
                batch_sampler = CurriculumBatchSampler(
                    dataset,
                    batch_size,
                    num_batches=num_samples,
                    curriculum_config=curriculum_config
                ) if curriculum_config is not None else TrainBatchSampler(
                    dataset,
                    batch_size
                )
                train_batch_sampler = batch_sampler
            elif dataset_set == 'val':
                num_samples = len(dataset) if num_samples <= 0 else num_samples
                batch_sampler = CurriculumBatchSampler(
                    dataset,
                    batch_size_val,
                    num_batches=num_samples,
                    curriculum_config={
                        'type': 'joint',
                        'scheduler': train_batch_sampler._curm_sch
                    }
                ) if val_follows_train else ValBatchSampler(
                    dataset,
                    batch_size_val,
                    num_batches=num_samples
                )
            else:
                raise ValueError(f'Unknown dataset type: {dataset_set}.')

            dataloaders[dataset_set] = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=FrameData.collate,
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataloaders
