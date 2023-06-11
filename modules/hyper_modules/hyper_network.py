import typing

import torch
import torch.nn as nn


class HyperNetwork(nn.Module):
    """
    Hyper network
    """
    def produce_parameters(self, latent_code: torch.Tensor) -> typing.List[torch.Tensor]:
        raise NotImplementedError

    def hyper_parameters(self) -> typing.Iterator[nn.Parameter]:
        raise NotImplementedError
