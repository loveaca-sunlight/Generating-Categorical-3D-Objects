import torch
import torch.nn as nn

from .hyper import HyperModule
from .hyper_linear import HyperLinear
from .hyper_layer_norm import HyperLayerNorm


class HyperResidualBlock(nn.Module):
    """
    Hyper linear block with residual connection
    """
    def __init__(self, dim_in: int, dim_out: int, dim_middle: int, norm: bool, hyper_dim_in: int, hyper_dim_hidden: int,
                 hyper_norm: bool = False):
        super(HyperResidualBlock, self).__init__()

        self.fc0 = HyperLinear(dim_in, dim_out, dim_middle, hyper_dim_in, hyper_dim_hidden, hyper_norm)
        self.fc1 = HyperLinear(dim_out, dim_out, dim_middle, hyper_dim_in, hyper_dim_hidden, hyper_norm)

        if norm:
            self.norm0 = HyperLayerNorm(dim_out, hyper_dim_in, hyper_dim_hidden, hyper_norm)
            self.norm1 = HyperLayerNorm(dim_out, hyper_dim_in, hyper_dim_hidden, hyper_norm)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()

        if dim_in != dim_out:
            self.downsample = nn.Sequential(
                HyperLinear(dim_in, dim_out, dim_middle, hyper_dim_in, hyper_dim_hidden, hyper_norm, bias=norm),
                HyperLayerNorm(dim_out, hyper_dim_in, hyper_dim_hidden, hyper_norm) if norm else nn.Identity()
            )
        else:
            self.downsample = nn.Identity()

        self.act = nn.ELU(inplace=True)

    def produce_parameters(self, latent_code: torch.Tensor):
        parameters = []
        for module in self.modules():
            if isinstance(module, HyperModule): #生成两个fc0
                parameters.extend(module.produce_parameters(latent_code))
        return parameters

    def forward(self, x: torch.Tensor):
        out = self.act(
            self.norm0(
                self.fc0(x)
            )
        )

        out = self.norm1(
            self.fc1(out)
        )

        out += self.downsample(x)

        return self.act(out)


class HyperBlocks(nn.Module):
    """
    Multiple blocks
    """
    def __init__(self, dim_in: int, dim_hidden: int, dim_middle: int, n_blocks: int, norm: bool, hyper_dim_in: int,
                 hyper_dim_hidden: int, hyper_norm: bool = True):
        super(HyperBlocks, self).__init__()

        assert n_blocks >= 1

        blocks = [
            HyperResidualBlock(
                dim_in=dim_in,
                dim_out=dim_hidden,
                dim_middle=dim_middle,
                norm=norm,
                hyper_dim_in=hyper_dim_in,
                hyper_dim_hidden=hyper_dim_hidden,
                hyper_norm=hyper_norm
            )
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                HyperResidualBlock(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    dim_middle=dim_middle,
                    norm=norm,
                    hyper_dim_in=hyper_dim_in,
                    hyper_dim_hidden=hyper_dim_hidden,
                    hyper_norm=hyper_norm
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def produce_parameters(self, latent_code: torch.Tensor):
        parameters = []
        for module in self.blocks:
            parameters.extend(module.produce_parameters(latent_code))
        return parameters

    def forward(self, x: torch.Tensor):
        out = self.blocks(x)

        return out


class HyperMLP(nn.Module):
    """
    Hyper MLP network
    """
    def __init__(self, dim_in: int, dim_hidden: int, dim_middle: int, n_blocks: int, norm: bool, hyper_dim_in: int,
                 hyper_dim_hidden: int, hyper_norm: bool = True):
        super(HyperMLP, self).__init__()

        self.linear_in = HyperLinear(dim_in, dim_hidden, dim_middle, hyper_dim_in, hyper_dim_hidden, hyper_norm)
        self.norm = HyperLayerNorm(dim_hidden, hyper_dim_in, hyper_dim_hidden, hyper_norm) if norm else nn.Identity()
        self.act = nn.ELU(inplace=True)

        self.blocks = nn.Sequential(
            *[
                HyperResidualBlock(dim_hidden, dim_hidden, dim_middle, norm, hyper_dim_in, hyper_dim_hidden, hyper_norm)
                for _ in range(n_blocks)
            ]
        )

    def produce_parameters(self, latent_code: torch.Tensor):
        parameters = []
        for module in (self.linear_in, self.norm):
            if isinstance(module, HyperModule):
                parameters.extend(module.produce_parameters(latent_code))
        for module in self.blocks:
            parameters.extend(module.produce_parameters(latent_code))
        return parameters

    def forward(self, x: torch.Tensor):
        out = self.act(
            self.norm(
                self.linear_in(x)
            )
        )
        out = self.blocks(out)

        return out
