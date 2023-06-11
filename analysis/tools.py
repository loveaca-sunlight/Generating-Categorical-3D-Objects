import typing

import torch
import matplotlib.pyplot as plt
import random
import os.path as osp
import numpy as np

from tqdm import tqdm


_FUNCTIONS_MAPPING = {
    'mean': np.mean,
    'var': np.var,
    'min': np.min,
    'max': np.max,
    'median': np.median
}


def embeddings_distribution(embedding: torch.nn.Embedding, sample_dims: int, output_dir: str, n_bins: int = None,
                            contents: typing.List[str] = None):
    # index all elements
    ids = torch.arange(embedding.num_embeddings)
    vectors = embedding(ids)  # (n, d)

    # sample dims
    n, d = vectors.shape
    dims = random.sample(list(range(d)), sample_dims)

    # output distribution hist
    for i in tqdm(dims):
        vec = vectors[:, i]
        arr = vec.numpy()

        plt.hist(arr, bins=n_bins)

        cnt = [f'{idx}={_FUNCTIONS_MAPPING[idx](arr)}' for idx in contents]
        plt.title(', '.join(cnt))

        plt.savefig(osp.join(output_dir, f'dim_{i}.png'))
        plt.close()
