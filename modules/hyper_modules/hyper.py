import torch
import torch.nn as nn

from modules.component import ResidualBlockFC


# _LEAKY_RELU_ALPHA = 1 / 5.5
#
#
# class _ResidualBlock(nn.Module):
#     def __init__(self, dim_in: int, dim_hidden: int):
#         super(_ResidualBlock, self).__init__()
#
#         self.fc0 = nn.Linear(dim_in, dim_hidden)
#         self.fc1 = nn.Linear(dim_hidden, dim_hidden)
#         if dim_in != dim_hidden:
#             self.downsample = nn.Linear(dim_in, dim_hidden, bias=False)
#         else:
#             self.downsample = nn.Identity()
#         self.act = nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x: torch.Tensor):
#         out = self.act(
#             self.fc0(x)
#         )
#         out = self.act(
#             self.downsample(x) + self.fc1(out)
#         )
#
#         return out


class HyperModule(nn.Module):
    """
    Base class for hyper modules
    """
    def __init__(self, hyper_dim_in: int, hyper_dim_hidden: int, hyper_norm: bool = False):
        super(HyperModule, self).__init__()

        # a 2-layer mlp
        self.mlp = ResidualBlockFC(
            dim_in=hyper_dim_in,
            dim_out=hyper_dim_hidden,
            norm=hyper_norm,
            act='leaky_relu'
        )

    def produce_parameters(self, latent_code: torch.Tensor):
        """
        Produce parameters for this module
        :param latent_code:
        :return:
        """
        raise NotImplementedError
