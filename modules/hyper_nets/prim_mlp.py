import math

import torch
import torch.nn.functional as F

from .hyper_nets import HyperBase


class PrimResMlp(torch.nn.Module):
    """
    A Primary mlp with residual connections
    """
    def __init__(self, hyper_net: HyperBase, dim_in: int, dim_hidden: int, n_blocks: int,
                 dim_z: int = 64, max_norm: float = None, norm: bool = False):
        """
        Initialize
        :param dim_in: input features
        :param dim_hidden: hidden features
        :param n_blocks: number of residual blocks
        :param dim_z: embedding features
        :param max_norm: max norm of layer embeddings
        """
        super(PrimResMlp, self).__init__()

        # hyper layers
        self._hyper_layers = n_blocks * 2
        self._n_blocks = n_blocks

        # hyper net
        assert isinstance(hyper_net, HyperBase)
        self.hyper_net = hyper_net

        # activation
        self.act = torch.nn.ELU(inplace=True)

        # layer embeddings
        self.layer_embeddings = torch.nn.Embedding(self._hyper_layers, dim_z,
                                                   max_norm=(math.sqrt(float(dim_z)) if max_norm is None else max_norm))

        # input linear
        self.linear_in = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_hidden),
            torch.nn.LayerNorm(dim_hidden) if norm else torch.nn.Identity(),
            torch.nn.ELU(inplace=True)
        )

        # register layer indexes
        self.register_buffer("_layer_ids", torch.arange(self._hyper_layers, dtype=torch.int32), persistent=False)

        # store weights
        self.weights = None
        self.biases = None
        self.alphas = None
        self.betas = None

        self._norm_shape = dim_hidden

        print(f'Name: {type(self).__name__}, Norm: {norm}.')

    def generate_weights(self, latent_code: torch.Tensor):
        """
        Generate weights
        :param latent_code: a latent code that controls how to generate weights, (1, d)
        :return:
        """
        assert latent_code.dim() == 2

        # get all layer_embeddings
        zs = self.layer_embeddings(self._layer_ids.view(-1))  # (D, dz)

        # get all weights and biases
        self.weights, self.biases, self.alphas, self.betas = self.hyper_net(zs, latent_code)

        return self.weights, self.biases, self.alphas, self.betas

    def forward(self, x: torch.Tensor):
        """
        Forward step
        :param x: input tensor
        :return:
        """
        # input feature
        out = self.linear_in(x)

        weight = self.weights
        bias = self.biases
        alpha = self.alphas + 1.0
        beta = self.betas

        # forward
        for i in range(self._n_blocks):
            block_in = out

            out = self.act(
                F.layer_norm(
                    input=F.linear(out, weight[i * 2, :, :], bias[i * 2, :]),
                    normalized_shape=[self._norm_shape],
                    weight=alpha[i * 2, :],
                    bias=beta[i * 2, :]
                )
            )

            out = self.act(
                block_in + F.layer_norm(
                    input=F.linear(out, weight[i * 2 + 1, :, :], bias[i * 2 + 1, :]),
                    normalized_shape=[self._norm_shape],
                    weight=alpha[i * 2 + 1, :],
                    bias=beta[i * 2 + 1, :]
                )
            )

        return out
