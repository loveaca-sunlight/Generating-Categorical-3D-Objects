import torch

from ..hyper_nets import HyperNetwork, PrimResMlp
from nerf.harmonic_embedding import HarmonicEmbedding


class PrimDeformableField(torch.nn.Module):
    """
    A deformable field that maps input points to the deformation to template
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            layer_embed_dim: int,
            shape_code_dim: int,
            n_hidden_neurons_prim: int,
            n_hidden_neurons_hyper: int,
            n_blocks_prim: int,
            n_blocks_hyper: int,
            hyper_dof: int,
            norm: bool
    ):
        super(PrimDeformableField, self).__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        hyper_net = HyperNetwork(
            dim_embed=layer_embed_dim,
            dim_code=shape_code_dim,
            dim_hidden=n_hidden_neurons_hyper,
            dim_weight=n_hidden_neurons_prim,
            n_blocks=n_blocks_hyper,
            dof=hyper_dof,
            norm=norm
        )

        self.mlp = PrimResMlp(
            hyper_net=hyper_net,
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_prim,
            n_blocks=n_blocks_prim,
            dim_z=layer_embed_dim,
            norm=norm
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_prim, 3)
        # set the weight to a small number and zero bias
        torch.nn.init.normal_(self.density_layer.weight.data, 0.0, 0.001)
        torch.nn.init.constant_(self.density_layer.bias.data, 0.0)

        self._feature_dim = n_hidden_neurons_prim

    @property
    def feature_dim(self):
        return self._feature_dim

    def generate_weights(self, shape_latent_code: torch.Tensor):
        """
        Generate network weights
        :return:
        """
        return self.mlp.generate_weights(shape_latent_code)

    def forward(
        self,
        rays_points_world: torch.Tensor
    ):
        """
        Compute points deformation and features
        :param rays_points_world:
        :return:
        """
        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds_xyz)

        # points deformation
        points_deformation = self.density_layer(features)

        return points_deformation, features
