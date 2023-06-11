import torch
import torch.nn.functional as F
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp, _init_linear
from modules.defnerf import PrimDeformableField
from modules.hyper_nets import HyperNetwork, PrimResMlp
from modules.transform_new import TransformModule
from modules.util import freeze
from nerf.harmonic_embedding import HarmonicEmbedding
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities


def _init_weight(linear):
    torch.nn.init.kaiming_uniform_(linear.weight.data, mode='fan_in', nonlinearity='linear')


@IMPLICIT_FUNCTIONS.register_module(name='prim_mlp')
class PrimMlp(torch.nn.Module):
    """
    An adapted version that ignores view direction
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons_density: int,
            n_hidden_neurons_color_prim: int,
            n_hidden_neurons_color_hyper: int,
            n_hidden_neurons_specular: int,
            n_blocks_density: int,
            n_blocks_color_prim: int,
            n_blocks_color_hyper: int,
            n_blocks_specular: int,
            feature_dim: int,
            shape_code_dim: int,
            color_code_dim: int,
            layer_embed_dim: int,
            hyper_dof: int,
            norm: bool,
            **kwargs
    ):
        """
        Initialize
        :param n_harmonic_functions_xyz:
        :param n_hidden_neurons_density:
        :param n_hidden_neurons_color:
        :param n_blocks_density:
        :param n_blocks_color:
        :param feature_dim: feature dimension of deformable field
        :param shape_code_dim:
        :param color_code_dim:
        :param density_activation:
        """
        super(PrimMlp, self).__init__()

        # Rigid transform module
        self.transformer = TransformModule()

        # Deformable module shared by all functions
        self.deformer = None
        self._feature_dim = feature_dim  # feature dims of deformation output

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        #
        # Density layers for storing template shape, don't modify the module name
        #
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_density,
            dim_out=n_hidden_neurons_density,
            n_blocks=n_blocks_density,
            norm=norm
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_density, 1)
        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0

        #
        # Color layers
        #
        # Intermediate linear of density
        intermediate_linear = torch.nn.Linear(
            feature_dim + n_hidden_neurons_density, feature_dim + n_hidden_neurons_density
        )  # project to low feature dimensions
        _init_weight(intermediate_linear)
        self.intermediate_linear = torch.nn.Sequential(
            intermediate_linear,
            torch.nn.LayerNorm(feature_dim + n_hidden_neurons_density) if norm else torch.nn.Identity()
        )

        # hyper network
        hyper_color = HyperNetwork(
            dim_embed=layer_embed_dim,
            dim_code=color_code_dim,
            dim_hidden=n_hidden_neurons_color_hyper,
            dim_weight=n_hidden_neurons_color_prim,
            n_blocks=n_blocks_color_hyper,
            dof=hyper_dof,
            norm=norm
        )

        self.mlp_diffuse = PrimResMlp(
            hyper_net=hyper_color,
            dim_in=(feature_dim + n_hidden_neurons_density),
            dim_hidden=n_hidden_neurons_color_prim,
            n_blocks=n_blocks_color_prim,
            dim_z=layer_embed_dim,
            max_norm=None,  # use default config
            norm=norm
        )

        self.diffuse_layer = torch.nn.Linear(n_hidden_neurons_color_prim, 3)
        _init_weight(self.diffuse_layer)

        #
        # Specular layers
        #
        intermediate_specular = torch.nn.Linear(
            feature_dim + n_hidden_neurons_density + n_hidden_neurons_color_prim,
            n_hidden_neurons_specular
        )
        _init_weight(intermediate_specular)
        self.intermediate_specular = torch.nn.Sequential(
            intermediate_specular,
            torch.nn.LayerNorm(n_hidden_neurons_specular) if norm else torch.nn.Identity()
        )

        if n_blocks_specular > 0:
            self.mlp_specular = ResMlp(
                dim_in=n_hidden_neurons_specular + embedding_dim_dir,
                dim_hidden=n_hidden_neurons_specular,
                dim_out=n_hidden_neurons_specular,
                n_blocks=n_blocks_specular,
                norm=norm
            )
        else:
            linear = torch.nn.Linear(
                n_hidden_neurons_specular + embedding_dim_dir,
                n_hidden_neurons_specular
            )
            _init_linear(linear)
            self.mlp_specular = torch.nn.Sequential(
                linear,
                torch.nn.LayerNorm(n_hidden_neurons_specular) if norm else torch.nn.Identity(),
                torch.nn.ELU(inplace=True)
            )

        self.specular_layer = torch.nn.Linear(n_hidden_neurons_specular, 3)
        # init the weight of specular layer to a small value and zero bias
        torch.nn.init.normal_(self.specular_layer.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.specular_layer.bias.data, 0.0)

        self._shape_code_dim = shape_code_dim
        self._color_code_dim = color_code_dim

        self._density_activation = 'relu'  # fix

        self._latest_points_deformation = None
        self._latest_points_position = None
        self._latest_points_specular = None

    def set_deformable_field(self, module: PrimDeformableField):
        assert isinstance(module, PrimDeformableField)
        assert self._feature_dim == module.feature_dim
        assert self.deformer is None, 'Deformable field has been set.'

        self.deformer = module

    def freeze_template_layers(self):
        freeze(self.mlp_xyz)
        freeze(self.density_layer)

    @property
    def latest_points_deformation(self):
        """
        Return latest points deformation
        :return:
        """
        return self._latest_points_deformation

    @property
    def latest_points_position(self):
        """
        Return latest points position
        :return:
        """
        return self._latest_points_position

    @property
    def latest_points_specular(self):
        """
        Return latest points specular
        :return:
        """
        return self._latest_points_specular

    def _get_densities(
            self,
            features: torch.Tensor,
            depth_values: torch.Tensor,
            density_noise_std: float,
            return_raw_density: bool = False
    ) -> torch.Tensor:
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)

        if return_raw_density:
            return raw_densities

        densities = get_densities(raw_densities, depth_values, density_noise_std,
                                  densities_activation=self._density_activation)
        return densities

    def _get_diffused_colors(
            self, features: torch.Tensor
    ):
        """
        This function predicts the diffuse color of each points in 3d space
        """
        diffuse_features = self.mlp_diffuse(
            self.intermediate_linear(features)
        )
        raw_diffuse = self.diffuse_layer(diffuse_features)

        diffuse = torch.sigmoid(raw_diffuse)

        return diffuse, diffuse_features

    def _get_specular_colors(
            self, features: torch.Tensor, directions: torch.Tensor
    ):
        """
        This function predicts the specular color and uncertainty
        :param features: cat of (deformer features, density features and color features), (n, i, p, d)
        :param directions: (n, i, 3)
        :return:
        """
        # Normalize the ray_directions to unit l2 norm. detach
        rays_directions_normed = torch.nn.functional.normalize(directions.detach(), dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions. (n, i, dr)
        dir_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # inputs
        n, i, p, _ = features.shape
        inputs = torch.cat(
            [
                self.intermediate_specular(features),
                dir_embedding[:, :, None, :].expand(-1, -1, p, -1)
            ], dim=-1
        )

        # outputs
        raw_specular = self.specular_layer(
            self.mlp_specular(
                inputs
            )
        )

        # model specular and shadow
        specular = torch.tanh(raw_specular)

        return specular

    def generate_weights(self, color_latent_code: torch.Tensor):
        """
        Generate weights for prim-net
        :param color_latent_code:
        :return:
        """
        return self.mlp_diffuse.generate_weights(color_latent_code)

    def forward(
            self,
            ray_bundle: RayBundle,

            transform_code: torch.Tensor,
            enable_deformation: bool = True,
            enable_specular: bool = True,

            density_noise_std: float = 0.0,

            return_raw_density: bool = False,
            **kwargs,
    ):
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param transform_code:
        :param enable_deformation:
        :param enable_specular:
        :param density_noise_std:
        :param return_raw_density:
        :param kwargs:
        :return:
        """
        # rigid transform
        if transform_code is not None:
            ray_bundle = self.transformer(ray_bundle, transform_code)
        else:
            assert not self.training

        # Convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`. (n, i, p, 3)
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # Store latest points position
        self._latest_points_position = rays_points_world

        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(rays_points_world)
            # use translation for points deformation
            rays_points_template = rays_points_world + points_deformation

            # Store latest points deformation
            self._latest_points_deformation = points_deformation
        else:
            assert not self.training, 'Deformation can not be disabled when training.'
            rays_points_template = rays_points_world
            deformer_features = None

        # For each 3D world coordinate, we obtain its harmonic embedding.
        # embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_template)

        # self.mlp maps each harmonic embedding to a latent feature space.
        xyz_features = self.mlp_xyz(embeds_xyz)

        # rays densities from template
        assert not (self.training and return_raw_density)
        rays_densities = self._get_densities(xyz_features, ray_bundle.lengths, density_noise_std,
                                             return_raw_density=return_raw_density)

        # get rays colors when deformation is enabled
        if enable_deformation:
            # concatenate the features from mlp_xyz and deformer as input
            cat_features = torch.cat([deformer_features, xyz_features], dim=-1)  # (n, i, p, dx + dd)

            # get diffused color and feature
            rays_diffuse, diffuse_features = self._get_diffused_colors(cat_features)  # (n, i, p, 3)

            # get specular color
            if enable_specular:
                feature_in = torch.cat([cat_features, diffuse_features], dim=-1)
                rays_specular = self._get_specular_colors(feature_in, ray_bundle.directions)
                self._latest_points_specular = rays_specular

                rays_colors = rays_diffuse + rays_specular
            else:
                rays_colors = rays_diffuse
        else:
            rays_colors = rays_points_template.new_zeros((*rays_points_template.shape[: 3], 3))  # (n, i, p, 3)

        # cat with depths
        rays_features = torch.cat(
            [
                rays_colors,
                ray_bundle.lengths.detach()[..., None]
            ], dim=-1
        )

        return rays_densities, rays_features
