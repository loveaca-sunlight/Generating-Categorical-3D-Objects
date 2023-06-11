import torch


def repeat_first_dim(tensor: torch.Tensor, dim: int, n_repeats: int):
    assert dim in [0, 1], f'dim can only be 0 or 1.'

    raw_shape = tensor.shape
    raw_dims = tensor.dim()

    expands = [-1] * raw_dims
    expands.insert(dim, n_repeats)

    out = tensor.unsqueeze(dim).expand(*expands)
    out = out.reshape(-1, *raw_shape[1:])

    return out
