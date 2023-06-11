import torch

from modules.component import ResMlp
from nerf.harmonic_embedding import HarmonicEmbedding
from ..util import deprecate_warning


class ReflectionNet(torch.nn.Module):
    """
    This network takes as input the rendered features, colors and ray directions and
    outputs the specular residual colors.
    """
    def __init__(
            self,
            n_harmonic_functions_dir: int,
            feature_dim: int,
            specular_code_dim: int,
            n_hidden_neurons: int,
            n_blocks: int
    ):
        super(ReflectionNet, self).__init__()

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.intermediate_linear = torch.nn.Linear(feature_dim, min(feature_dim, n_hidden_neurons))

        self.mlp = ResMlp(
            dim_in=(specular_code_dim + embedding_dim_dir + min(feature_dim, n_hidden_neurons) + 3),  # 3 for diffuse
            dim_hidden=n_hidden_neurons,
            dim_out=n_hidden_neurons,
            n_blocks=n_blocks
        )

        self.specular_layer = torch.nn.Linear(n_hidden_neurons, 3)
        # init the weight to a small value and zero bias
        torch.nn.init.normal_(self.specular_layer.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(self.specular_layer.bias.data, 0.0)

        deprecate_warning(self)

    def forward(self, specular_code: torch.Tensor, features: torch.Tensor, diffuse: torch.Tensor,
                directions: torch.Tensor):
        """
        Compute specular color
        :param specular_code: (n, d)
        :param features: (n, i, d)
        :param diffuse: (n, i, 3)
        :param directions: (n, i, 3)
        :return:
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions. (n, i, dr)
        dir_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # Cat for input
        n, i, _ = features.shape
        inputs = torch.cat(
            [
                specular_code[:, None, :].expand(n, i, -1),
                self.intermediate_linear(features),
                diffuse,
                dir_embedding
            ],
            dim=-1
        )  # (n, i, df + 3 + dr)

        # Raw specular, (n, i, 3)
        raw_specular = self.specular_layer(
            self.mlp(inputs)
        )

        # Use relu for specular, (n, i, 3)
        specular = torch.relu(raw_specular)

        return specular


class ReflectionNetV2(torch.nn.Module):
    """
    Model the view-dependent colors and aleatoric uncertainty
    """
    def __init__(
            self,
            n_harmonic_functions_xy: int,
            n_harmonic_functions_dir: int,
            specular_code_dim: int,
            n_hidden_neurons: int,
            n_blocks: int,
            beta_min: float = 0.01
    ):
        super(ReflectionNetV2, self).__init__()

        assert beta_min > 0, f'beta_min must be positive.'

        self.harmonic_embedding_xy = HarmonicEmbedding(n_harmonic_functions_xy)
        embedding_dim_xy = n_harmonic_functions_xy * 2 * 2 + 2

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp = ResMlp(
            dim_in=(embedding_dim_xy + embedding_dim_dir + specular_code_dim),
            dim_hidden=n_hidden_neurons,
            dim_out=n_hidden_neurons,
            n_blocks=n_blocks
        )

        self.specular_layer = torch.nn.Linear(n_hidden_neurons, 3)
        # init the weight to a small value and zero bias
        torch.nn.init.normal_(self.specular_layer.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(self.specular_layer.bias.data, 0.0)

        self.beta_layer = torch.nn.Linear(n_hidden_neurons, 2)
        # init the weight to a small value and zero bias
        torch.nn.init.normal_(self.beta_layer.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(self.beta_layer.bias.data, 0.0)
        self.softplus = torch.nn.Softplus(beta=1.4)  # softplus(0) = 0.49 when beta == 1.4
        self._beta_min = beta_min

    def forward(self, xys: torch.Tensor, directions: torch.Tensor, specular_code: torch.Tensor):
        """
        Compute view/direction- dependent colors
        :param xys: (n, i, 2)
        :param directions: (n, i, 3)
        :param specular_code: (n, d)
        :return:
        """
        # Obtain the harmonic embedding of xy
        xy_embedding = self.harmonic_embedding_xy(xys)

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(directions, dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions. (n, i, dr)
        dir_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # Cat for input
        n, i, _ = xys.shape
        inputs = torch.cat(
            [xy_embedding, dir_embedding, specular_code[:, None, :].expand(n, i, -1)],
            dim=-1
        )

        mlp_features = self.mlp(inputs)

        # Compute specular
        raw_specular = self.specular_layer(mlp_features)
        # use tanh as activation, (n, i, 3)
        specular = torch.tanh(raw_specular)

        # Compute uncertainty
        raw_uncertainty = self.beta_layer(mlp_features)
        # use softplus as activation, (n, i, 1)
        uncertainty = self.softplus(raw_uncertainty) + self._beta_min

        return specular, uncertainty
