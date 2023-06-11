from audioop import bias
from random import sample
from typing import Tuple
from matplotlib.font_manager import weight_dict

import torch
import torch.nn as nn
import numpy as np
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from modules.component import ResMlp
from nerf.harmonic_embedding import HarmonicEmbedding
from nerf.linear_with_repeat import LinearWithRepeat
from nerf.embedder import get_embedder
from .registry import IMPLICIT_FUNCTIONS
from .util import get_densities


@IMPLICIT_FUNCTIONS.register_module(name='neus_mlp_res')
class NeusMlpRes(torch.nn.Module):
    """
    Mlp that stores the SDF information of a template
    """
    def __init__(
            self,
            n_harmonic_functions_xyz: int,
            n_harmonic_functions_dir: int,
            n_hidden_neurons_xyz: int,
            n_hidden_neurons_dir: int,
            n_blocks_xyz: int,
            n_blocks_dir: int,
            densities_only: bool,
            density_activation: str,
            norm: bool = False #此处的设置由cfg修改
    ):
        super(NeusMlpRes, self).__init__()

        self._densities_only = densities_only

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        
        self.embed_fn_fine = None
        embed_fn, input_ch = get_embedder(n_harmonic_functions_xyz, input_dims=3)
        self.embed_fn_fine = embed_fn
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3 #39

        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3 #33

        # Density layer
        dims = [embedding_dim_xyz] + [n_hidden_neurons_xyz for _ in range(n_blocks_xyz)] + [n_hidden_neurons_xyz + 1]

        self.num_layers = len(dims)
        self.skip_in = [2,5,8]#[2,4,6]
        self.scale = 1.0
        bias = 0.5
        weight_norm = True

        self.mlp_xyz = ResMlp(
            dim_in=embedding_dim_xyz,
            dim_hidden=n_hidden_neurons_xyz,
            dim_out=n_hidden_neurons_xyz,
            n_blocks=n_blocks_xyz,
            norm=norm,
            weight_norm=weight_norm,
        )

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        #torch.nn.init.kaiming_uniform_(self.density_layer.weight.data, nonlinearity='linear')
        torch.nn.init.normal_(self.density_layer.weight, mean=np.sqrt(np.pi) / np.sqrt(n_hidden_neurons_xyz), std=0.0001)
        torch.nn.init.constant_(self.density_layer.bias, -bias)
        if weight_norm:
            self.density_layer=nn.utils.weight_norm(self.density_layer)
        
        #self.norm0 = nn.LayerNorm(out_dim)

        # Color layer
        if not self._densities_only:
            # Intermediate linear of density
            self.intermediate_linear = torch.nn.Linear(
                n_hidden_neurons_xyz, n_hidden_neurons_xyz
            )
            torch.nn.init.kaiming_uniform_(self.intermediate_linear.weight.data, nonlinearity='linear')

            color_layer = [
                LinearWithRepeat(
                    n_hidden_neurons_xyz + embedding_dim_dir + 6, n_hidden_neurons_dir
                ),
                torch.nn.LayerNorm(n_hidden_neurons_dir) if norm else torch.nn.Identity(),
                torch.nn.ELU(inplace=True)
            ]

            assert n_blocks_dir == 0
            # if n_blocks_dir > 0:
            #     color_layer.extend([
            #         ResidualBlockFC(n_hidden_neurons_dir, n_hidden_neurons_dir, n_hidden_neurons_dir)
            #         for _ in range(n_blocks_dir)
            #     ])

            color_layer.extend([
                torch.nn.Linear(n_hidden_neurons_dir, 3),
                torch.nn.Sigmoid(),
            ])
            self.color_layer = torch.nn.Sequential(*color_layer)

    
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


    def _get_colors(
            self, features: torch.Tensor, rays_directions: torch.Tensor ,rays_points_world: torch.Tensor, gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions. (n, i, 3)
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer((self.intermediate_linear(features), rays_embedding, rays_points_world, gradient))

    def forward(
            self,
            ray_bundle: RayBundle,
            density_noise_std: float = 0.0,

            return_raw_density: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rays densities and colors
        :param ray_bundle: original ray bundle
        :param density_noise_std:
        :param return_raw_density:
        :param kwargs:
        :return:
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]
        batch, chunk, sample, _ = rays_points_world.shape
            
        if kwargs['valid'] == False:
            x = rays_points_world.detach()
            
            #embeds_xyz = self.harmonic_embedding_xyz(x)
            sdf, features = self.density_layer_forward(x)

            #torch.set_grad_enabled(True)
            with torch.set_grad_enabled(True):
                gradient = self.get_gradient(x)
                #gradient = None
                gradients = gradient.detach()
                del gradient
            #torch.set_grad_enabled(False)
        else:
            sdf, features = self.density_layer_forward(rays_points_world)
            gradients = self.get_gradient(rays_points_world)

        rays_colors = self._get_colors(features, ray_bundle.directions ,rays_points_world, gradients)

        '''d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]'''

        return sdf.reshape(batch, chunk, sample, 1), rays_colors.reshape(batch, chunk, sample, 3), gradients.reshape(batch, chunk, sample, 3)#, rays_points_world.reshape(batch, chunk, sample, 3)
