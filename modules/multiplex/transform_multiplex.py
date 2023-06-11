import collections
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from modules.transform.utils import EULER_CONVENTION


@torch.no_grad()
def init_from_base(n_ways: int, base_transform: Dict[str, torch.Tensor] = None):
    """
    Init transform from base transforms
    :param n_ways:
    :param base_transform: R - (3, 3), T - (3, 1), S - (1,)
    :return:
    """
    if base_transform is None:
        base_transform = {
            'R': torch.eye(3, dtype=torch.float),
            'T': torch.zeros(3, 1, dtype=torch.float),
            'S': torch.ones(1, dtype=torch.float)
        }

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
        )
    else:
        raise ValueError('n_ways can only be 4 or 8.')
    # Transform matrix
    Ri = euler_angles_to_matrix(euler_angles, EULER_CONVENTION)  # (n_ways, 3, 3)
    Rb = base_transform['R'].view(1, 3, 3)
    R = Ri @ Rb  # (n_ways, 3, 3)
    T = Ri @ base_transform['T']  # (n_ways, 3, 1)
    S = base_transform['S']  # (1,)
    # Convert to transform code
    rotation = matrix_to_euler_angles(R, EULER_CONVENTION)  # (n_ways, 3)
    translation = T.view(n_ways, 3)  # (n_ways, 3)
    scale = S[None].expand(n_ways, 1)  # (n_ways, 1)
    transform_codes = torch.cat([rotation, translation, scale], dim=-1)  # (n_ways, 7)
    # create and init parameters
    parameters = nn.Parameter(torch.zeros(n_ways, 7, dtype=torch.float), requires_grad=True)
    parameters.data.copy_(transform_codes)

    return parameters


class TransformMultiplex(nn.Module):
    def __init__(self, n_ways: int, base_transform: Dict[str, torch.Tensor], full_transform: bool = True,
                 beta: float = 0.9):
        """
        Initialize
        :param n_ways: number of initialized transforms
        """
        super().__init__()

        assert full_transform, 'Deprecated parameter, full_transform must be True.'

        self.transforms = init_from_base(
            n_ways=n_ways,
            base_transform=base_transform
        )

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
    def __init__(self, sequence_names: List[str], n_ways: int,
                 base_transforms: Dict[str, Dict[str, torch.Tensor]] = None):
        super(TransformMultiplexDict, self).__init__()

        self._n_ways = n_ways

        self.multiplexes = nn.ModuleDict(
            {
                name: TransformMultiplex(
                    n_ways,
                    base_transform=(None if base_transforms is None else base_transforms[name])
                )
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
    def __init__(self, sequence_names: List[str], full_transform: bool):
        super(TransformDict, self).__init__()

        parameter_length = (7 if full_transform else 6)
        self.transforms = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, parameter_length), requires_grad=True)
                for name in sequence_names
            }
        )
        # set scale to 1
        if full_transform:
            for param in self.transforms.values():
                param.data[:, -1] = 1.0

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
        # try to directly load
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            print('Trying to append scale parameters...')
            # try to append scale parameter
            with torch.no_grad():
                state_dict_ = {
                    name: torch.cat([param, torch.ones(1, 1, dtype=param.dtype)], dim=1)
                    for name, param in state_dict.items()
                }
            self.load_state_dict(state_dict_, strict=True)
