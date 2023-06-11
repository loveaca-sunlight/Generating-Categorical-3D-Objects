# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List
import warnings

import numpy as np
from torch.utils.data.sampler import Sampler

from dataset.co3d_dataset import Co3dDataset
from dataset.scene_batch_sampler import _capped_random_choice
from .curriculum_scheduler import CURRICULUM_SCHEDULERS


@dataclass(eq=False)  # TODO: do we need this if not init from config?
class CurriculumBatchSampler(Sampler[List[int]]):
    """
    A batch sampler for curriculum training
    """

    dataset: Co3dDataset
    batch_size: int
    num_batches: int
    # configs for curriculum learning
    curriculum_config: dict

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integral value, "
                f"but got batch_size={self.batch_size}"
            )

        self.seq_names = list(self.dataset.seq_annots.keys())

        # log iter number
        self._curr_iters = 0

        # create curriculum scheduler
        assert 'seq_names' not in self.curriculum_config, 'seq_names can not be specified.'
        self.curriculum_config['seq_names'] = self.seq_names
        self._curm_sch = CURRICULUM_SCHEDULERS.build(self.curriculum_config)
        print(self._curm_sch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for batch_idx in range(len(self)):
            batch = self._sample_batch(batch_idx)
            # increase counter
            self._curr_iters += 1
            yield batch

    def _sample_batch(self, batch_idx):
        n_per_seq = self.batch_size
        n_seqs = 1

        # select sequence from limited range
        sel_seqs = self._curm_sch(self._curr_iters)
        assert len(sel_seqs) > 0, 'Selected empty list.'
        chosen_seq = _capped_random_choice(sel_seqs, n_seqs, replace=False)

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
        if len(frame_idx) < self.batch_size:
            warnings.warn(
                "Batch size smaller than self.batch_size!"
                + " (This is fine for experiments with a single scene and viewpooling)"
            )
        return frame_idx
