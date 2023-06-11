import torch

from ..component import ResMlp


class HyperBase(torch.nn.Module):
    """
    A base class for all hyper-networks
    """
    def forward(self, layer_embedding: torch.Tensor, latent_code: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Generate weights
        :param layer_embedding: (1, dz)
        :param latent_code: (1, dc)
        :return: (weight, bias)
        """
        raise NotImplementedError


class HyperNetwork(HyperBase):
    """
    A hyper-network that generate weights and biases
    """
    def __init__(self, dim_embed: int, dim_code: int, dim_hidden: int, dim_weight: int, n_blocks: int, dof: int,
                 norm: bool = False):
        super(HyperNetwork, self).__init__()

        self.mlp = ResMlp(
            dim_in=(dim_embed + dim_code),
            dim_hidden=dim_hidden,
            dim_out=dim_hidden,
            n_blocks=n_blocks,
            norm=norm
        )

        self.w0 = torch.nn.Linear(dim_hidden, dim_weight * dof)
        self.w1 = torch.nn.Linear(dim_hidden, dim_weight * dof)
        self.b = torch.nn.Linear(dim_hidden, dim_weight)
        self.alpha = torch.nn.Linear(dim_hidden, dim_weight)
        self.beta = torch.nn.Linear(dim_hidden, dim_weight)

        self._dim_weight = dim_weight
        self._dof = dof

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        for linear in (self.w0, self.w1, self.b, self.alpha, self.beta):
            torch.nn.init.normal_(linear.weight.data, 0.0, 0.002)
            if linear.bias is not None:
                torch.nn.init.constant_(linear.bias.data, 0.0)

    def forward(self, layer_embedding: torch.Tensor, latent_code: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Generate weights
        :param layer_embedding: (D, dz)
        :param latent_code: (1, dc)
        :return:
        """
        # cat
        n_layers = layer_embedding.shape[0]
        x = torch.cat(
            [
                layer_embedding,
                latent_code.expand(n_layers, -1)
            ], dim=-1)  # (D, dz + dc)

        # weights and biases
        feature = self.mlp(x)  # (D, d)

        w0 = self.w0(feature).view(n_layers, self._dim_weight, self._dof)
        w1 = self.w1(feature).view(n_layers, self._dof, self._dim_weight)

        weight = w0 @ w1  # (n_layers, dim_w, dim_w)
        bias = self.b(feature)  # (n_layers, dim_w)
        alpha = self.alpha(feature)  # (n_layers, dim_w)
        beta = self.beta(feature)  # # (n_layers, dim_w)

        return weight, bias, alpha, beta
