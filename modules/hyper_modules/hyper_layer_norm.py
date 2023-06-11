import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyper import HyperModule


class HyperLayerNormV1(HyperModule):
    """
    A hyper layer norm module
    """
    def __init__(self, dim_hidden: int, hyper_dim_in: int, hyper_dim_hidden: int, hyper_norm: bool = False):
        super(HyperLayerNormV1, self).__init__(
            hyper_dim_in,
            hyper_dim_hidden,
            hyper_norm
        )

        self._dim_hidden = dim_hidden
        self._out_scale = 0.5

        # heads
        self.w = nn.Linear(hyper_dim_hidden, dim_hidden)
        self.b = nn.Linear(hyper_dim_hidden, dim_hidden)

        # store weights
        self.weight = None
        self.bias = None

        # reset parameters
        self.reset_params()

    @torch.no_grad()
    def reset_params(self):
        for m in (self.w, self.b):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='linear')
            m.weight.data *= 0.1
            m.bias.data *= 0.1

    def produce_parameters(self, latent_code: torch.Tensor):
        """
        Produce parameters
        :param latent_code: (1, d)
        :return:
        """
        assert latent_code.shape[0] == 1
        
        feature = self.mlp(latent_code)

        self.weight = self.w(feature).view(self._dim_hidden) * self._out_scale
        self.bias = self.b(feature).view(self._dim_hidden) * self._out_scale

        return self.weight, self.bias

    def forward(self, x: torch.Tensor):
        weight = self.weight + 1.0
        bias = self.bias
        return F.layer_norm(x, [self._dim_hidden], weight, bias)


def HyperLayerNormV2(dim_hidden: int, hyper_dim_in: int, hyper_dim_hidden: int, hyper_norm: bool = False):
    return nn.LayerNorm(dim_hidden)


HyperLayerNorm = HyperLayerNormV1
print(f'HyperLayerNorm Type: {HyperLayerNorm.__name__}.')
