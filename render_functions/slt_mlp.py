from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp
from modules.transform_new import TransformModule
from modules.util import freeze
from nerf.harmonic_embedding import HarmonicEmbedding
from nerf.embedder import get_embedder
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities, cat_ray_bundles


@IMPLICIT_FUNCTIONS.register_module(name='slt_mlp')
class SilhouetteMlp(torch.nn.Module):
    """
    Mlp that stores the density information of a template
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_hidden_neurons_xyz: int,
            n_blocks_xyz: int,
            n_transforms: int,
            density_activation: str,
            norm: bool
    ):
        super(SilhouetteMlp, self).__init__()

        # Transform Module
        self._n_transforms = n_transforms
        self.transformer = TransformModule()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        
        self.embed_fn_fine = None
        embed_fn, input_ch = get_embedder(n_harmonic_functions_xyz, input_dims=3)
        self.embed_fn_fine = embed_fn
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3 #39

        # Density layer
        dims = [embedding_dim_xyz] + [n_hidden_neurons_xyz for _ in range(n_blocks_xyz)] + [n_hidden_neurons_xyz + 1]

        self.num_layers = len(dims)
        self.skip_in = [2,4,6]
        self.scale = 1.0
        bias = 0.5
        weight_norm = True

        # Density layer
        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz,
            norm=norm,
            weight_norm=weight_norm,
            act='elu'
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        if weight_norm:
            self.density_layer=nn.utils.weight_norm(self.density_layer)

        assert density_activation in ['softplus', 'relu']
        if density_activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            self.activation = nn.ReLU(True)

        self._transformed_key_points = None

    @property
    def transformed_key_points(self):
        return self._transformed_key_points

    def density_layer_forward(self, inputs): #inputs = pts
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        features = self.mlp_xyz(inputs)
        sdf = self.density_layer(features)
        return sdf, features

    def get_sdf(
            self,
            ray_bundle: RayBundle,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        #embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        #nn.functional.normalize(rays_points_world,dim=-1)
        sdf, _ = self.density_layer_forward(rays_points_world)
    
        return rays_points_world, sdf

    def get_sdf_withpts(
            self,
            pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        #embeds_xyz = self.harmonic_embedding_xyz(pts)
        #nn.functional.normalize(pts,dim=-1)
        sdf, _ = self.density_layer_forward(pts)
    
        return sdf
    
    def get_gradient(
            self,
            rays_points_world: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def freeze_template_layers(self):
        freeze(self.mlp_xyz)
        freeze(self.density_layer)

    def forward(
            self,
            ray_bundle: RayBundle,
            transform_codes: torch.Tensor,
            key_points: torch.Tensor,
            density_noise_std: float = 0.0,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param transform_codes:
        :param key_points: (1, p, 3)
        :param density_noise_std:
        :param kwargs:
        :return:
        """
        # transform ray bundle with multiple transform parameters
        if transform_codes is not None:
            n_transforms = self._n_transforms if self.training else 1
            assert transform_codes.shape[0] == n_transforms, \
                f'Excepted number of transforms is {n_transforms}, but given {transform_codes.shape[0]}.'

            n_multiplex = transform_codes.shape[0]
            transforms = torch.chunk(transform_codes, n_multiplex, dim=0)  # [(1, d), ... x m]
            # transform ray bundle and key points
            transformed = [self.transformer(ray_bundle, transform, key_points=key_points)
                           for transform in transforms]  # [(bundle, (points)), ... x m]
            # if key points is not None
            if key_points is not None:
                # transformed ray bundles
                ray_bundle = cat_ray_bundles([t[0] for t in transformed])
                # transformed key points
                self._transformed_key_points = torch.cat(
                    [t[1] for t in transformed],
                    dim=0
                )  # (m, p, 3)
            else:
                # transformed ray bundles
                ray_bundle = cat_ray_bundles(transformed)

        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        batch, chunk, sample, _ = rays_points_world.shape

        if kwargs['valid'] == False:
            x = rays_points_world.detach()
            
            #embeds_xyz = self.harmonic_embedding_xyz(x)
            sdf, xyz_features = self.density_layer_forward(x)

        else:
            sdf, xyz_features = self.density_layer_forward(rays_points_world)


        rays_colors = sdf.new_zeros(size=(*sdf.shape[: 3], 3))
        #rays_colors = self._get_colors(features, ray_bundle.directions ,rays_points_world)
        '''d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]'''

        return sdf.reshape(batch, chunk, sample, 1), rays_colors.reshape(batch, chunk, sample, 3), ray_bundle#, rays_points_world.reshape(batch, chunk, sample, 3)