import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyper import HyperModule


class HyperLinearV1(HyperModule):
    """
    A linear hyper module
    """
    def __init__(self, dim_in: int, dim_out: int, dim_middle: int, hyper_dim_in: int, hyper_dim_hidden: int,
                 hyper_norm: bool = True, bias: bool = True):
        super(HyperLinearV1, self).__init__(
            hyper_dim_in,
            hyper_dim_hidden,
            hyper_norm
        )

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._dim_middle = dim_middle

        # heads
        self.w0 = nn.Linear(hyper_dim_hidden, dim_in * dim_middle)
        self.w1 = nn.Linear(hyper_dim_hidden, dim_out * dim_middle)
        self.b = nn.Linear(hyper_dim_hidden, dim_out) if bias else None

        # store weights
        self.weight = None
        self.bias = None

        # reset parameters
        self.reset_params()

    @torch.no_grad()
    def reset_params(self):
        for m in (self.w0, self.w1, self.b):
            if m is not None:
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='linear')
                m.weight.data *= 0.5
                m.bias.data *= 0.5

    def produce_parameters(self, latent_code: torch.Tensor):
        """
        Produce parameters
        :param latent_code: (1, d)
        :return:
        """
        assert latent_code.shape[0] == 1

        feature = self.mlp(latent_code)

        w0 = self.w0(feature).view(self._dim_middle, self._dim_in)
        w1 = self.w1(feature).view(self._dim_out, self._dim_middle)

        self.weight = (w1 @ w0).view(self._dim_out, self._dim_in)  # (dim_out, dim_in)

        if self.b is None:
            return self.weight,

        self.bias = self.b(feature).view(self._dim_out)  #

        return self.weight, self.bias

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)


class HyperLinearV2(HyperModule):
    """
    A linear hyper module
    """
    def __init__(self, dim_in: int, dim_out: int, dim_middle: int, hyper_dim_in: int, hyper_dim_hidden: int,
                 hyper_norm: bool = True, bias: bool = True):
        super(HyperLinearV2, self).__init__(
            hyper_dim_in,
            hyper_dim_hidden,
            hyper_norm
        )

        self._dim_in = dim_in
        self._dim_out = dim_out

        # heads
        self.w = nn.Linear(hyper_dim_hidden, dim_out * dim_in)
        self.b = nn.Linear(hyper_dim_hidden, dim_out) if bias else None

        # store weights
        self.weight = None
        self.bias = None

        # reset parameters
        self.reset_params()

    @torch.no_grad()
    def reset_params(self):
        for m in (self.w, self.b):
            if m is not None:
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='linear')
                m.weight.data *= 0.1
                m.bias.data *= 0.1

    def produce_parameters(self, latent_code: torch.Tensor):
        """
        Produce parameters
        :param latent_code: (1, d)
        :return:
        """
        assert latent_code.shape[0] == 1

        feature = self.mlp(latent_code) #提特征的残差mlp，两层linear+Res

        self.weight = self.w(feature).view(self._dim_out, self._dim_in)  # (dim_out, dim_in) 取weight

        if self.b is None:
            return self.weight,

        self.bias = self.b(feature).view(self._dim_out)  # (dim_out) 去bias

        return self.weight, self.bias

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias) #实际上就一层 HyperLinear


# version control
HyperLinear = HyperLinearV2
print(f'HyperLinear Version: {HyperLinear.__name__}.')
