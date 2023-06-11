import torch

from modules.component import ResMlp
from nerf.harmonic_embedding import HarmonicEmbedding


class DeformableField(torch.nn.Module):
    """
    A deformable field that maps input points to the offsets to template
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_hidden_neurons_xyz: int,
            n_blocks_xyz: int,
            shape_code_dim: int
    ):
        super(DeformableField, self).__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz + shape_code_dim,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 3)
        # set the weight to a small number and zero bias
        torch.nn.init.normal_(self.density_layer.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(self.density_layer.bias.data, 0.0)

        self._feature_dim = n_hidden_neurons_xyz

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward(
            self,
            rays_points_world: torch.Tensor,
            shape_latent_code: torch.Tensor
    ):
        """
        Compute points deformation and features
        :param rays_points_world:
        :param shape_latent_code:
        :return:
        """
        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # input
        n, i, p, _ = embeds_xyz.shape
        mlp_input = torch.cat([embeds_xyz, shape_latent_code[:, None, None, :].expand(n, i, p, -1)], dim=-1)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(mlp_input)

        # points deformation
        points_deformation = self.density_layer(features)

        return points_deformation, features
