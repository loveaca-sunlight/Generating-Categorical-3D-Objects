import torch
import torch.nn as nn
import warnings


def get_bounding_points(fg_prob: torch.Tensor, thr: float):
    """
    Return bounding points ((min_x, max_x), (min_y, max_y))
    :param fg_prob: foreground probability
    :param thr: threshold
    """
    assert fg_prob.dim() == 4

    # (n, h, w)
    prob = (fg_prob.squeeze(1) >= thr).float()
    n, _, _ = prob.shape

    rows = prob.sum(dim=-2)  # (n, w)
    bx = [
        torch.nonzero(rows[i, :])[[0, -1]]
        for i in range(n)
    ]  # ([2] * n)
    bx = torch.stack(bx, dim=0).squeeze(-1)  # (n, 2)

    cols = prob.sum(dim=-1)  # (n, h)
    by = [
        torch.nonzero(cols[i, :])[[0, -1]]
        for i in range(n)
    ]
    by = torch.stack(by, dim=0).squeeze(-1)  # (n, 2)

    return bx, by


def _set_requires_grad(module: nn.Module, state: bool):
    for param in module.parameters():
        param.requires_grad = state


def freeze(module: nn.Module):
    _set_requires_grad(module, False)


def unfreeze(module: nn.Module):
    _set_requires_grad(module, True)


def deprecate_warning(module: object, postfix: str = None):
    msg = f'The module {type(module).__name__} is deprecated.'
    if postfix is not None:
        msg = f'{msg} {postfix}'
    warnings.warn(msg)
