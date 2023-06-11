import torch

from nerf.harmonic_embedding import HarmonicEmbedding
from modules.component import ResMlp


class UncertaintyNet(torch.nn.Module):
    """
    Model the view-dependent colors and aleatoric uncertainty
    """
    def __init__(
            self,
            n_harmonic_functions_xy: int,
            n_harmonic_functions_dir: int,
            beta_code_dim: int,
            n_hidden_neurons: int,
            n_blocks: int,
            beta_min: float = 0.4
    ):
        super(UncertaintyNet, self).__init__()

        assert beta_min > 0, f'beta_min must be positive.'

        self.harmonic_embedding_xy = HarmonicEmbedding(n_harmonic_functions_xy)
        embedding_dim_xy = n_harmonic_functions_xy * 2 * 2 + 2

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp = ResMlp(
            dim_in=(embedding_dim_xy + embedding_dim_dir + beta_code_dim),
            dim_hidden=n_hidden_neurons,
            dim_out=n_hidden_neurons,
            n_blocks=n_blocks
        )

        # model the uncertainty of mask and color
        self.beta_layer = torch.nn.Linear(n_hidden_neurons, 2)
        # init the weight to a small value and zero bias
        torch.nn.init.normal_(self.beta_layer.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(self.beta_layer.bias.data, 0.0)
        self.softplus = torch.nn.Softplus(beta=4)
        self._beta_min = beta_min

        print(f'beta_min: {self._beta_min}.')

    def forward(self, xys: torch.Tensor, directions: torch.Tensor, beta_code: torch.Tensor):
        """
        Compute view/direction- dependent colors
        :param xys: (n, i, 2)
        :param directions: (n, i, 3)
        :param beta_code: (n, d)
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
            [xy_embedding, dir_embedding, beta_code[:, None, :].expand(n, i, -1)],
            dim=-1
        )

        mlp_features = self.mlp(inputs)

        # Compute uncertainty
        raw_uncertainty = self.beta_layer(mlp_features)
        # use softplus as activation, (n, i, 1)
        uncertainty = self.softplus(raw_uncertainty) + self._beta_min
        # split to beta_m and beta_c
        beta_m, beta_c = torch.chunk(uncertainty, 2, -1)

        return beta_m, beta_c
