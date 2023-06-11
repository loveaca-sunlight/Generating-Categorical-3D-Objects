import random

from typing import List, Iterator
from torch.utils.data.sampler import Sampler, T_co
from .llff import LLFFDataset


class LLFFSampler(Sampler[List[int]]):
    def __init__(self, dataset: LLFFDataset, batch_size: int, num_batches: int):
        super(LLFFSampler, self).__init__(dataset)

        self._data_len = len(dataset)
        self._batch_size = batch_size
        self._num_batches = num_batches

        print(f'Data Length: {self._data_len}, Num Batches: {self._num_batches}.')

    def __len__(self):
        return self._num_batches

    def __iter__(self) -> Iterator[T_co]:
        for i in range(len(self)):
            ids = random.sample(list(range(self._data_len)), self._batch_size)
            yield ids
