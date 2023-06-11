from typing import Tuple

import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from nerf.harmonic_embedding import HarmonicEmbedding
from nerf.linear_with_repeat import LinearWithRepeat
from nerf.implicit_function import _xavier_init, MLPWithInputSkips


class NeuralRadianceField(torch.nn.Module):
    """
    An adapted nerf implementation that can take as input an appearance embedding
    """

    def __init__(
            self,
            dir_position: str,
            appearance_code_dim: int = None,

            n_harmonic_functions_xyz: int = 6,
            n_harmonic_functions_dir: int = None,
            n_hidden_neurons_density: int = 256,
            n_hidden_neurons_color: int = 128,
            n_layers_density: int = 8,
            append_density: Tuple[int] = (5,),
            use_multiple_streams: bool = True,
            **kwargs,
    ):
        """
        Args:
            dir_position: can be either 'front' or 'tail'
            appearance_code_dim: dimensions of appearance code
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
                None means don't use position embedding
            n_hidden_neurons_density: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_color: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_density: The number of layers of the MLP that outputs the
                occupancy field.
            append_density: The list of indices of the skip layers of the occupancy MLP.
            use_multiple_streams: Whether density and color should be calculated on
                separate CUDA streams.
        """
        super().__init__()

        assert dir_position in ['front', 'tail'], f'dir_position can only be "front" or "tail".'
        self.dir_position = dir_position

        if appearance_code_dim is None:
            assert dir_position == 'tail', 'dir_position must be "tail" when appearance_code_dim is None.'
        self.appearance_code_dim = 0 if appearance_code_dim is None else appearance_code_dim

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        if n_harmonic_functions_dir is not None:
            self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
            embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3
        else:
            self.harmonic_embedding_dir = None
            embedding_dim_dir = 3

        mlp_params = {
            'n_layers': n_layers_density,
            'input_dim': embedding_dim_xyz if dir_position == 'tail' else embedding_dim_xyz + embedding_dim_dir,
            'output_dim': n_hidden_neurons_density,
            'skip_dim': embedding_dim_xyz if dir_position == 'tail' else embedding_dim_xyz + embedding_dim_dir,
            'hidden_dim': n_hidden_neurons_density,
            'input_skips': append_density
        }

        self.mlp_density = MLPWithInputSkips(
            **mlp_params
        )

        self.intermediate_linear = torch.nn.Linear(
            n_hidden_neurons_density, n_hidden_neurons_density
        )
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(n_hidden_neurons_density, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough

        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(
                n_hidden_neurons_density + (embedding_dim_dir + self.appearance_code_dim), n_hidden_neurons_color
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_neurons_color, 3),
            torch.nn.Sigmoid(),
        )
        self.use_multiple_streams = use_multiple_streams

    def _get_densities(
            self,
            features: torch.Tensor,
            depth_values: torch.Tensor,
            density_noise_std: float,
    ) -> torch.Tensor:
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]
        if density_noise_std > 0.0:
            raw_densities = (
                    raw_densities + torch.randn_like(raw_densities) * density_noise_std
            )
        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()

        # TODO densities can not be negative
        assert torch.all(densities >= 0), 'densities can not be negative.'

        return densities

    def _get_colors(
            self, features: torch.Tensor, rays_directions: torch.Tensor, appearance_code: torch.Tensor
    ) -> torch.Tensor:
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # color input
        if self.dir_position == 'tail':
            # Obtain the harmonic embedding of the normalized ray directions.  # (n, i, d)
            color_input = self._get_direction_embedding(rays_directions)
            if appearance_code is not None:
                appearance_code = appearance_code[:, None, :].expand(-1, color_input.shape[1], -1)
                color_input = torch.cat([color_input, appearance_code], dim=-1)
        else:
            color_input = appearance_code[:, None, :].expand(-1, features.shape[1], -1)

        return self.color_layer((self.intermediate_linear(features), color_input))

    def _get_densities_and_colors(
            self, features: torch.Tensor, ray_bundle: RayBundle,
            appearance_code: torch.Tensor, density_noise_std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The second part of the forward calculation.

        Args:
            features: the output of the common mlp (the prior part of the
                calculation), shape
                (minibatch x ... x self.n_hidden_neurons_xyz).
            ray_bundle: As for forward().
            appearance_code: embedding code of appearance.
            density_noise_std:  As for forward().

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        if self.use_multiple_streams and features.is_cuda:
            current_stream = torch.cuda.current_stream(features.device)
            other_stream = torch.cuda.Stream(features.device)
            other_stream.wait_stream(current_stream)

            with torch.cuda.stream(other_stream):
                rays_densities = self._get_densities(
                    features, ray_bundle.lengths, density_noise_std
                )
                # rays_densities.shape = [minibatch x ... x 1] in [0-1]

            rays_colors = self._get_colors(features, ray_bundle.directions, appearance_code)
            # rays_colors.shape = [minibatch x ... x 3] in [0-1]

            current_stream.wait_stream(other_stream)
        else:
            # Same calculation as above, just serial.
            rays_densities = self._get_densities(
                features, ray_bundle.lengths, density_noise_std
            )
            rays_colors = self._get_colors(features, ray_bundle.directions, appearance_code)
        return rays_densities, rays_colors

    def _get_direction_embedding(self, rays_directions):
        # Normalize the ray_directions to unit l2 norm.
        rays_embedding = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        if self.harmonic_embedding_dir is not None:
            rays_embedding = self.harmonic_embedding_dir(rays_embedding)

        return rays_embedding

    def forward(
            self,
            ray_bundle: RayBundle,
            appearance_code: torch.Tensor = None,
            density_noise_std: float = 0.0,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            appearance_code: embedding code of appearance
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        if appearance_code is None:
            assert self.appearance_code_dim == 0, f'appearance_code must be None if appearance_code_dim is 0.'

        # We first convert the ray parametrizations to world, (n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        # For each 3D world coordinate, we obtain its harmonic embedding. (n, i, p, d)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        # input to mlp
        mlp_input = embeds_xyz
        if self.dir_position == 'front':
            # get embedding of direction
            embeds_dir = self._get_direction_embedding(ray_bundle.directions)

            embeds_dir = embeds_dir[:, :, None, :].expand_as(embeds_xyz)
            mlp_input = torch.cat([embeds_xyz, embeds_dir], dim=-1)

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_density(mlp_input, mlp_input)

        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle, appearance_code, density_noise_std
        )
        return rays_densities, rays_colors
