# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.data.sampler import Sampler

from dataset.co3d_dataset import Co3dDataset


@dataclass(eq=False)
class TrainBatchSampler(Sampler[List[int]]):
    """
    A class for sampling training batches with a controlled composition
    of sequences.
    """

    dataset: Co3dDataset
    batch_size: int

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integral value, "
                f"but got batch_size={self.batch_size}"
            )

        self.seq_names = list(self.dataset.seq_annots.keys())

    def _shuffle_sequences(self):
        random.shuffle(self.seq_names)

    def __len__(self):
        return len(self.seq_names)

    def __iter__(self):
        self._shuffle_sequences()
        for chosen_seq in self.seq_names:
            batch = self._sample_batch(chosen_seq)
            yield batch

    def _sample_batch(self, chosen_seq):
        # extend the number of samples to batch size for single-seq data
        # DN: turning this off as we should not assume users want to do this automatically
        # n_per_seq = max(n_per_seq, self.batch_size // len(chosen_seq))
        frame_idx = _capped_random_choice(
                        self.dataset.seq_to_idx[chosen_seq], self.batch_size, replace=True  # replace=False, adapted
                    ).tolist()
        # if len(frame_idx) < self.batch_size:
        #     warnings.warn(
        #         "Batch size smaller than self.batch_size!"
        #         + " (This is fine for experiments with a single scene and viewpooling)"
        #     )
        return frame_idx


@dataclass(eq=False)
class ValBatchSampler(Sampler[List[int]]):
    """
    A class for sampling training batches with a controlled composition
    of sequences.
    """

    dataset: Co3dDataset
    batch_size: int
    num_batches: int

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integral value, "
                f"but got batch_size={self.batch_size}"
            )

        self.seq_names = list(self.dataset.seq_annots.keys())

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for batch_idx in range(len(self)):
            batch = self._sample_batch(batch_idx)
            yield batch

    def _sample_batch(self, batch_idx):
        n_per_seq = self.batch_size
        n_seqs = 1
        chosen_seq = _capped_random_choice(self.seq_names, n_seqs, replace=False)
        # extend the number of samples to batch size for single-seq data
        # DN: turning this off as we should not assume users want to do this automatically
        # n_per_seq = max(n_per_seq, self.batch_size // len(chosen_seq))
        frame_idx = np.concatenate(
            [
                _capped_random_choice(
                    self.dataset.seq_to_idx[seq], n_per_seq, replace=True  # replace=False, adapted
                )
                for seq in chosen_seq
            ]
        )[: self.batch_size].tolist()

        return frame_idx


def _capped_random_choice(x, size, replace=True):
    """
    if replace==True
        randomly chooses from x `size` elements without replacement if len(x)>size
        else allows replacement and selects `size` elements again.
    replace==False
        randomly chooses from x `min(len(x), size)` elements without replacement
    """
    len_x = x if isinstance(x, int) else len(x)
    if replace:
        return np.random.choice(x, size=size, replace=len_x < size)
    else:
        return np.random.choice(x, size=min(size, len_x), replace=False)
