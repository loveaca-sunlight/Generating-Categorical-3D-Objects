import os.path
import sys
import warnings

import torch
import torch.nn as nn


_LEAKY_RELU_ALPHA = 1 / 5.5


def _init_linear(linear, nonlinearity: str):
    """
    Performs the weight initialization of the linear layer `linear`.
    """
    if nonlinearity == 'elu':
        nonlinearity = 'selu'
        warnings.warn('Using elu activation.')
    nn.init.kaiming_normal_(linear.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in', nonlinearity=nonlinearity)


def _get_activation(name: str):
    if name == 'leaky_relu':
        return nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'Soft':
        return nn.Softplus(beta=100)
    else:
        raise ValueError(f'Unknown activation: {name}.')

import numpy as np
class ResidualBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    """

    def __init__(self, dim_in: int, dim_out: int, norm: bool = False, weight_norm: bool = True, act: str = 'leaky_relu'):
        super().__init__()

        # Submodules
        self.fc_0 = nn.Linear(dim_in, dim_out, bias=True)
        self.fc_1 = nn.Linear(dim_out, dim_out, bias=True)
        self.norm0 = nn.LayerNorm(dim_out) if norm else nn.Identity()
        self.norm1 = nn.LayerNorm(dim_out) if norm else nn.Identity()
        if dim_in != dim_out:
            self.downsample = nn.Sequential(
                nn.Linear(dim_in, dim_out, bias=False),
                nn.LayerNorm(dim_out) if norm else nn.Identity()
            )
        else:
            self.downsample = nn.Identity()

        self.act = _get_activation(act)

        # init
        '''for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, act)'''

        torch.nn.init.constant_(self.fc_0.bias, 0.0)
        torch.nn.init.normal_(self.fc_0.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))

        torch.nn.init.constant_(self.fc_1.bias, 0.0)
        torch.nn.init.normal_(self.fc_1.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))

        if weight_norm == True:
            self.fc_1 = nn.utils.weight_norm(self.fc_1)
            self.fc_0 = nn.utils.weight_norm(self.fc_0)
            self.act = nn.Softplus(beta=100)

    def forward(self, x):
        out = self.act(self.norm0(self.fc_0(x)))
        dx = self.norm1(self.fc_1(out))
        x_s = self.downsample(x)

        return self.act(x_s + dx)


def _get_invoker_name(n_back: int = 1):
    frame = sys._getframe().f_back
    for _ in range(n_back):
        frame = frame.f_back
    return os.path.basename(frame.f_code.co_filename)


class ResBlocks(nn.Module):
    """
    Mlp with residual connections
    """
    def __init__(
            self,
            dim_in: int,
            dim_hidden: int,
            n_blocks: int,
            norm: bool = False,
            act: str = 'leaky_relu'
    ):
        """
        Initialize
        :param dim_in: input dimensions
        :param dim_hidden: hidden dimensions
        :param n_blocks: number of blocks
        """
        super(ResBlocks, self).__init__()

        blocks = [
            ResidualBlockFC(
                dim_in=dim_in,
                dim_out=dim_hidden,
                norm=norm,
                act=act
            )
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                ResidualBlockFC(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    norm=norm
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)

        return out


class LinearBlock(nn.Module):
    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            norm: bool,
            act: str = 'leaky_relu'
    ):
        super(LinearBlock, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out) if norm else nn.Identity()
        self.act = _get_activation(act)

        _init_linear(self.fc, nonlinearity=act)

    def forward(self, x: torch.Tensor):
        return self.act(
            self.norm(
                self.fc(x)
            )
        )


class ResMlp(nn.Module):
    """
    Mlp with residual connections
    """
    def __init__(
            self,
            dim_in: int,
            dim_hidden: int,
            dim_out: int,
            n_blocks: int,
            norm: bool = False,
            weight_norm: bool = True,
            act: str = 'leaky_relu'
    ):
        """
        Initialize
        :param dim_in: input dimensions
        :param dim_hidden: hidden dimensions
        :param dim_out: output dimensions
        :param n_blocks: number of blocks
        """
        super(ResMlp, self).__init__()

        # input layer
        self.linear_in = nn.Linear(dim_in, dim_hidden, bias=True)
        #_init_linear(self.linear_in, act)
        torch.nn.init.constant_(self.linear_in.bias, 0.0)
        torch.nn.init.constant_(self.linear_in.weight[:, 3:], 0.0)
        torch.nn.init.normal_(self.linear_in.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
        

        self.norm = nn.LayerNorm(dim_hidden) if norm else nn.Identity()
        self.act = _get_activation(act)
        if weight_norm == True:
            self.linear_in = nn.utils.weight_norm(self.linear_in)
            self.act = nn.Softplus(beta=100)

        # intermediate blocks
        self.blocks = nn.Sequential(
            *[ResidualBlockFC(
                dim_in=dim_hidden,
                dim_out=dim_hidden,
                norm=norm,
                weight_norm=weight_norm,
                act=act
            ) for _ in range(n_blocks)]
        )

        print(f'Name: {_get_invoker_name(1)}, Norm Layer: {type(self.norm).__name__}.')

    def forward(self, x):
        out = self.act(self.norm(self.linear_in(x)))
        out = self.blocks(out)

        return out
