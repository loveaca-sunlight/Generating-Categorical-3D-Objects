import collections
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from ..transform_new.utils import euler_to_vec6d

_EULER_CONVENTION = 'XYZ'


@torch.no_grad()
def init_transforms(n_ways: int):
    """
    Initialize multiple transform codes
    :param n_ways:
    :return:
    """
    pi = np.pi
    if n_ways == 4:
        euler_angles = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [pi, 0.0, 0.0],
                [0.0, pi, 0.0],
                [0.0, 0.0, pi]
            ], dtype=torch.float
        )  # (4, 3)
    elif n_ways == 8:
        euler_angles = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [pi, 0.0, 0.0],
                [0.0, pi, 0.0],
                [0.0, 0.0, pi],
                [pi, pi, 0.0],
                [0.0, pi, pi],
                [pi, 0.0, pi],
                [pi, pi, pi],
            ], dtype=torch.float
        )  # (8， 3)
    else:
        raise ValueError('n_ways can only be 4 or 8.')
    # rotation vector
    vec_r = euler_to_vec6d(euler_angles, convention=_EULER_CONVENTION)
    # translation
    vec_t = torch.zeros(n_ways, 3)
    # scale
    vec_s = torch.ones(n_ways, 1)
    # cat
    vectors = torch.cat([vec_r, vec_t, vec_s], dim=-1)  # (n_ways, 10)
    # parameters
    parameters = torch.nn.Parameter(torch.randn(n_ways, 10), requires_grad=True) #将trans转化为可训练参数
    parameters.data.copy_(vectors)

    return parameters


class TransformMultiplex(nn.Module):
    def __init__(self, n_ways: int, beta: float = 0.9):
        """
        Initialize
        :param n_ways: number of initialized transforms
        """
        super().__init__()

        self.transforms = init_transforms(n_ways)

        self._n_ways = n_ways
        self._beta = beta

        # save scores, bigger is better
        self.register_buffer('scores', torch.zeros(n_ways, dtype=torch.float))

    @staticmethod
    def losses_to_scores(losses):
        with torch.no_grad():
            _scores = (losses - losses.min()) / (losses.max() - losses.min() + 1.0e-6)  # [0, 1]
            scores = 1.0 - _scores
        return scores

    def update_scores(self, scores: torch.Tensor):
        assert scores.numel() == self._n_ways, f'The excepted number of scores is {self._n_ways}, ' \
                                               f'but given {scores.numel()}.'
        # only can update scores when training
        assert self.training

        # update scores with EWMA
        with torch.no_grad():
            self.scores = self._beta * self.scores + (1. - self._beta) * scores

    def best_transform(self):
        """
        Return the transform with highest score
        :return:
        """
        idx = torch.argmax(self.scores)[None]
        return self.transforms[idx, :]

    def forward(self):
        """
        Return the stored transforms
        :return:
        """
        if self.training:
            return self.transforms
        else:
            # only return the best transform
            return self.best_transform()


class TransformMultiplexDict(nn.Module):
    def __init__(self, sequence_names: List[str], n_ways: int, **kwargs):
        super(TransformMultiplexDict, self).__init__()

        self._n_ways = n_ways

        self.multiplexes = nn.ModuleDict(
            {
                name: TransformMultiplex(n_ways, **kwargs)
                for name in sequence_names
            }
        )

    @property
    def sequence_names(self):
        return list(self.multiplexes.keys())

    @property
    def num_multiplexes(self):
        return self._n_ways if self.training else 1

    def forward(self, sequence_name: str):
        return self.multiplexes[sequence_name]()

    def update_scores(self, sequence_name: str, scores: torch.Tensor):
        self.multiplexes[sequence_name].update_scores(scores)


class TransformDict(nn.Module):
    """
    A parameter dict that only stores the best transform
    """

    def __init__(self, sequence_names: List[str]):
        super(TransformDict, self).__init__()

        init_parameter = torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float, requires_grad=False
        ).view(1, 10)

        self.transforms = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(*init_parameter.shape), requires_grad=True)
                for name in sequence_names
            }
        )

        with torch.no_grad():
            for param in self.transforms.parameters():
                param.data.copy_(init_parameter)

    def forward(self, sequence_name: str):
        return self.transforms[sequence_name]

    @property
    def keys(self):
        return self.transforms.keys()

    def load_pretrained_transforms(self, state_dict: collections.OrderedDict):
        """
        Load pretrained transforms
        :param state_dict:
        :return:
        """
        self.load_state_dict(state_dict, strict=True)
