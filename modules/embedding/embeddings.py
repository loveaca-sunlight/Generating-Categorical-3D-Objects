import torch
import math

from typing import List, Dict, Union, Callable
from torch.distributions import Normal


class EmbeddingsBase(torch.nn.Module):
    def forward(self, names: Union[List[str], Dict[str, List[str]]]) -> Dict:
        raise NotImplementedError

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.

        Example::

            model = MyLightningModule(...)
            model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()

        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()


class LatentCodeEmbeddings(EmbeddingsBase):
    """
    Embeddings that stores latent codes
    """

    def __init__(
            self,
            n_embeddings: int,
            embedding_dims: Dict[str, int],
            name_idx_mapping: Dict[str, int],
            max_norm: float = None
    ):
        """
        Initialize
        :param n_embeddings:
        :param embedding_dims:
        :param name_idx_mapping:
        :param max_norm: ignored, set to sqrt(dim)
        """
        super(LatentCodeEmbeddings, self).__init__()

        self.n_embeddings = n_embeddings

        self.name_idx_mapping = name_idx_mapping

        assert len(embedding_dims) == n_embeddings
        self.embeddings = torch.nn.ModuleDict(
            {
                name: torch.nn.Embedding(len(name_idx_mapping), dims, max_norm=math.sqrt(dims))
                for name, dims in embedding_dims.items()
            }
        )

        self.register_buffer('buf', torch.zeros(1), persistent=False)

    def _names_to_ids(self, names: List[str]):
        ids = [self.name_idx_mapping[name] for name in names]
        idx_tensor = torch.tensor(ids, dtype=torch.int, device=self.buf.device)
        # (n, d)
        return idx_tensor

    def reset_parameters(self, funcs: Union[Callable, Dict[str, Callable]], **kwargs):
        if isinstance(funcs, Callable):
            funcs = {
                k: funcs for k in self.embeddings.keys()
            }

        for name, func in funcs.items():
            for params in self.embeddings[name].parameters():
                func(params, **kwargs)

    def forward(self, names: Union[List[str], Dict[str, List[str]]]):
        """
        Return embedded vectors according to given indexes
        :param names:
        :return:
        """
        if isinstance(names, List):
            ids = self._names_to_ids(names)
            embeddings = {
                k: v(ids) for k, v in self.embeddings.items()
            }
        elif isinstance(names, Dict):
            embeddings = {
                k: self.embeddings[k](self._names_to_ids(v)) for k, v in names.items()
            }
        else:
            raise TypeError(f'names can only be a list or dict, but given {type(names)}.')

        return embeddings


def _zero_param(param: torch.nn.Parameter):
    param.data.zero_()


class VAELikeEmbeddingsModule(torch.nn.Module):
    def __init__(
            self,
            n_dims: int,
            name_idx_mapping: Dict[str, int]
    ):
        super(VAELikeEmbeddingsModule, self).__init__()

        self.embeddings = LatentCodeEmbeddings(
            n_embeddings=2,
            embedding_dims={
                'mu': n_dims,
                'log_var': n_dims
            },
            name_idx_mapping=name_idx_mapping
        )

        # zero embedded parameters
        self.embeddings.reset_parameters(_zero_param)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, names: List[str]):
        embeddings = self.embeddings(names)
        mu, log_var = embeddings['mu'], embeddings['log_var']  # (n, d)

        z = self.reparameterize(mu, log_var)

        # KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        return z, kld_loss


class VAELikeEmbeddings(EmbeddingsBase):
    def __init__(
            self,
            n_embeddings: int,
            embedding_dims: Dict[str, int],
            name_idx_mapping: Dict[str, int]
    ):
        super(VAELikeEmbeddings, self).__init__()

        assert len(embedding_dims) == n_embeddings
        self.embeddings = torch.nn.ModuleDict(
            {
                name: VAELikeEmbeddingsModule(dims, name_idx_mapping)
                for name, dims in embedding_dims.items()
            }
        )

    def forward(self, names: Union[List[str], Dict[str, List[str]]]):
        if isinstance(names, List):
            embeddings = {
                k: v(names) for k, v in self.embeddings.items()
            }
        elif isinstance(names, Dict):
            embeddings = {
                k: self.embeddings[k](v) for k, v in names.items()
            }
        else:
            raise TypeError(f'names can only be a list or dict, but given {type(names)}.')

        return embeddings
