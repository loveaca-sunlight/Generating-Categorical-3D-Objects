import typing
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp, ResBlocks, LinearBlock
from modules.defnerf import HyperDeformationField
from modules.hyper_modules import HyperNetwork, HyperLinear, HyperBlocks
from modules.transform_new import TransformModule
from modules.util import freeze
from nerf.embedder import get_embedder
from nerf.harmonic_embedding import HarmonicEmbedding
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities


def _init_weight(linear):
    torch.nn.init.kaiming_uniform_(linear.weight.data, mode='fan_in', nonlinearity='linear')


@IMPLICIT_FUNCTIONS.register_module(name='as2_neus_mlp_res')
class AS2NeusMlpRes(HyperNetwork):
    """
    An adapted version that ignores view direction
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            dim_hidden_density: int,
            dim_hidden_diffuse: int,
            dim_hidden_deformer: int,
            dim_middle_diffuse: int,
            n_blocks_density: int,
            n_blocks_diffuse: int,
            norm: bool,
            shape_code_dim: int,
            color_code_dim: int,
            hyper_dim_hidden_diffuse: int,
            hyper_norm: bool,
            **kwargs
    ):
        """
        Initialize
        """
        super(AS2NeusMlpRes, self).__init__()

        # Rigid transform module
        self.transformer = TransformModule()

        # Deformable module shared by all functions
        self.deformer = None
        self._dim_hidden_deformer = dim_hidden_deformer  # feature dims of deformation output

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)

        self.embed_fn_fine = None
        embed_fn, input_ch = get_embedder(n_harmonic_functions_xyz, input_dims=3)
        self.embed_fn_fine = embed_fn
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3 #39

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

         # Density layer
        dims = [embedding_dim_xyz] + [dim_hidden_density for _ in range(n_blocks_density)] + [dim_hidden_density + 1]

        self.num_layers = len(dims)
        self.skip_in = [2,4,6]
        self.scale = 1.0
        bias = 0.5
        weight_norm = True

        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=dim_hidden_density,
            dim_out=dim_hidden_density,
            n_blocks=n_blocks_density,
            norm=norm,
            weight_norm=weight_norm
        )
        self.density_layer = torch.nn.Linear(dim_hidden_density, 1)
        if weight_norm:
            self.density_layer=nn.utils.weight_norm(self.density_layer)
        #
        # Color layers
        #
        '''self.intermediate_linear = torch.nn.Linear(
            dim_hidden_density, dim_hidden_density
        )
        torch.nn.init.kaiming_uniform_(self.intermediate_linear.weight.data, nonlinearity='linear')'''

        self.mlp_diffuse = HyperBlocks(
            dim_in=(dim_hidden_density + dim_hidden_deformer + embedding_dim_dir + 6),
            dim_hidden=dim_hidden_diffuse,
            dim_middle=dim_middle_diffuse,
            n_blocks=n_blocks_diffuse,
            norm=norm,
            hyper_dim_in=color_code_dim,
            hyper_dim_hidden=hyper_dim_hidden_diffuse,
            hyper_norm=hyper_norm
        )

        self.diffuse_layer = HyperLinear(
            dim_in=dim_hidden_diffuse,
            dim_out=3,
            dim_middle=3,
            hyper_dim_in=color_code_dim,
            hyper_dim_hidden=hyper_dim_hidden_diffuse,
            hyper_norm=hyper_norm
        )

        self._shape_code_dim = shape_code_dim
        self._color_code_dim = color_code_dim

        self._density_activation = 'relu'  # fix

        self._latest_points_deformation = None
        self._latest_points_position = None
        self._latest_points_specular = None

        print(f'{type(self).__name__}: norm: {norm}, hyper_norm: {hyper_norm}.')

    def set_deformable_field(self, module: HyperDeformationField):
        assert isinstance(module, HyperDeformationField)
        assert self._dim_hidden_deformer == module.feature_dim
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
        return self._latest_points_deformation * 8

    @property
    def latest_points_position(self):
        """
        Return latest points position
        :return:
        """
        return self._latest_points_position * 8

    @property
    def latest_points_specular(self):
        """
        Return latest points specular
        :return:
        """
        return self._latest_points_specular * 8

    def get_sdf(
            self,
            ray_bundle: RayBundle,
            enable_deformation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(rays_points_world)
            # use translation for points deformation
            rays_points_template = rays_points_world + points_deformation
        else:
            rays_points_template = rays_points_world

        #nn.functional.normalize(rays_points_world,dim=-1)
        sdf, _ = self.density_layer_forward(rays_points_template)
    
        return sdf

    def get_sdf_withpts(
            self,
            pts: torch.Tensor,
            enable_deformation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(pts)
            # use translation for points deformation
            rays_points_template = pts + points_deformation
        else:
            rays_points_template = pts

        sdf, _ = self.density_layer_forward(rays_points_template)
    
        return sdf

    def _get_colors(
            self, features: torch.Tensor, directions: torch.Tensor, rays_points_world: torch.Tensor, gradient: torch.Tensor
    ):
        """
        This function predicts the diffuse color of each points in 3d space
        """
        # Normalize the ray_directions to unit l2 norm. detach
        rays_directions_normed = torch.nn.functional.normalize(directions.detach(), dim=-1)
        # Obtain the harmonic embedding of the normalized ray directions. (n, i, dr)
        dir_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        # inputs
        n, i, p, _ = features.shape
        inputs = torch.cat(
            [
                features,
                dir_embedding[:, :, None, :].expand(-1, -1, p, -1),
                rays_points_world,
                gradient
            ], dim=-1
        )

        diffuse_features = self.mlp_diffuse(
            inputs
        )
        raw_diffuse = self.diffuse_layer(diffuse_features)

        color = torch.sigmoid(raw_diffuse)

        return color

    def produce_parameters(self, latent_code: torch.Tensor) -> typing.List[torch.Tensor]:
        parameters = []
        for module in (self.mlp_diffuse, self.diffuse_layer):
            parameters.extend(module.produce_parameters(latent_code))
        return parameters

    def hyper_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        for module in (self.mlp_diffuse, self.diffuse_layer):
            yield from module.parameters()
    
    def _get_gradient(
            self,
            rays_points_world: torch.Tensor,
    ):
        batch, chunk, sample, _ = rays_points_world.shape
        x = rays_points_world.reshape(-1,3).clone()
        x.requires_grad_(True)

        #getsdf
        #embeds_xyz = self.harmonic_embedding_xyz(x)
        y, _ = self.density_layer_forward(x)

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True, #可以试试不保留计算图优化
            only_inputs=True)[0]
        return gradients.reshape(batch, chunk, sample, 3)

    def density_layer_forward(self, inputs): #inputs = pts
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        features = self.mlp_xyz(inputs)
        sdf = self.density_layer(features)
        return sdf, features

    def compute_density(
            self,
            points: torch.Tensor,
            enable_deformation: bool = True,
    ):
        """
        Compute density of given points for matching cubes
        :param points: (p, 3)
        :param enable_deformation:
        :return:
        """
        if enable_deformation:
            # Map points to template using deformer
            points_deformation, deformer_features = self.deformer(points)
            # use translation for points deformation
            rays_points_template = points + points_deformation
        else:
            rays_points_template = points

        rays_densities, xyz_features = self.density_layer_forward(rays_points_template)

        # get densities
        #rays_densities = self._get_densities(xyz_features, None, None, return_raw_density=True)
        #rays_densities = torch.relu(rays_densities)

        return rays_densities

    def save_latest_points_position(self, pts: torch.Tensor,):
        # Store latest points position
        self._latest_points_position = pts

    def forward(
            self,
            ray_bundle: RayBundle,
            transform_code: torch.Tensor,
            enable_deformation: bool = True,
            density_noise_std: float = 0.0,
            return_raw_density: bool = False,
            **kwargs,
    ):
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param transform_code:
        :param enable_deformation:
        :param density_noise_std:
        :param return_raw_density:
        :param kwargs:
        :return:
        """
        # Convert the ray parametrizations to world coordinates with `ray_bundle_to_ray_points`. (n, i, p, 3)
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
        
        '''import random
        index1 = torch.LongTensor(random.sample(range(0,40), 1))
        index2 = torch.LongTensor(random.sample(range(40,ray_bundle[2].shape[2]), 20))
        index = torch.cat((index1,index2))
        if kwargs['chunk_idx'] == 0:
            pts_plt = torch.index_select(rays_points_template[0,::60,...],1,index.cuda())
            np.save(('plt_camera.npy'),pts_plt.cpu().numpy())
        else:
            pts_plt = torch.index_select(rays_points_template[0,::60,...],1,index.cuda())
            plt=np.load('plt_deform.npy')
            plt_new=np.concatenate((plt,pts_plt.cpu().numpy()),axis=0)
            np.save(('plt_camera.npy'),plt_new)'''

        if kwargs['valid'] == False:
            x = rays_points_template.detach()
            
            #embeds_xyz = self.harmonic_embedding_xyz(x)
            sdf, xyz_features = self.density_layer_forward(x)

            #torch.set_grad_enabled(True)
            with torch.set_grad_enabled(True):
                gradient = self._get_gradient(x)
                #gradient = None
                gradients = gradient.detach()
                del gradient
            #torch.set_grad_enabled(False)
        else:
            sdf, xyz_features = self.density_layer_forward(rays_points_template)
            #gradients = self._get_gradient(rays_points_template)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=rays_points_template,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True, #可以试试不保留计算图优化
                only_inputs=True)[0]

        # get rays colors when deformation is enabled
        if enable_deformation:
            # concatenate the features from mlp_xyz and deformer as input
            cat_features = torch.cat([deformer_features, xyz_features], dim=-1)  # (n, i, p, dx + dd)

            # get color
            rays_colors = self._get_colors(cat_features, ray_bundle.directions, rays_points_template, gradients)  # (n, i, p, 3)

            self._latest_points_specular = 0
        else:
            cat_features = torch.cat([xyz_features, xyz_features], dim=-1)  # (n, i, p, dx + dd)

            # get color
            rays_colors = self._get_colors(cat_features, ray_bundle.directions, rays_points_template, gradients)  # (n, i, p, 3)

        # cat with depths
        rays_features = torch.cat(
            [
                rays_colors,
                ray_bundle.lengths.detach()[..., None],
                gradients,
            ], dim=-1
        )

        return sdf, rays_features, gradients