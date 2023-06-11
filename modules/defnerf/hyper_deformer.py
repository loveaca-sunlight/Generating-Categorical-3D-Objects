import torch
import torch.nn as nn
import typing

from nerf.harmonic_embedding import HarmonicEmbedding
from ..hyper_modules import HyperMLP, HyperLinear, HyperNetwork


class HyperDeformationField(HyperNetwork):
    def __init__(
            self,
            n_harmonic_functions: int,
            dim_hidden: int,
            dim_middle: int,
            n_blocks: int,
            norm: bool,
            shape_code_dim: int,
            hyper_dim_hidden: int,
            hyper_norm: bool
    ):
        super(HyperDeformationField, self).__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions)
        embedding_dim_xyz = n_harmonic_functions * 2 * 3 + 3

        self.mlp = HyperMLP(
            dim_in=embedding_dim_xyz,
            dim_hidden=dim_hidden,
            dim_middle=dim_middle,
            n_blocks=n_blocks,
            norm=norm,
            hyper_dim_in=shape_code_dim,
            hyper_dim_hidden=hyper_dim_hidden,
            hyper_norm=hyper_norm
        )

        self.deformation_layer = HyperLinear(
            dim_in=dim_hidden,
            dim_out=3,
            dim_middle=dim_middle,
            hyper_dim_in=shape_code_dim,
            hyper_dim_hidden=hyper_dim_hidden,
            hyper_norm=hyper_norm
        )

        self.feature_dim = dim_hidden

        # reset the parameters in deformation layer to a small value
        self.reset_params()

    @torch.no_grad()
    def reset_params(self):
        for m in (self.deformation_layer.w, self.deformation_layer.b):
            if isinstance(m, nn.Linear):
                m.weight.data *= 1.0  # 0.5
                if m.bias is not None:
                    m.bias.data *= 1.0  # 0.5

    def produce_parameters(self, latent_code: torch.Tensor):
        parameters = []
        for module in (self.mlp, self.deformation_layer):
            parameters.extend(module.produce_parameters(latent_code))
        return parameters

    def hyper_parameters(self) -> typing.Iterator[nn.Parameter]:
        for module in (self.mlp, self.deformation_layer):
            yield from module.parameters()

    def forward(self, rays_points_world: torch.Tensor):
        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds_xyz)

        # points deformation
        points_deformation = self.deformation_layer(features)

        return points_deformation, features
