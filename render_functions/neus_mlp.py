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


@IMPLICIT_FUNCTIONS.register_module(name='neus_mlp')
class NeusMlp(torch.nn.Module):
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
        super(NeusMlp, self).__init__()

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
        self.skip_in = [2,4,6]#[2,4,6]
        self.scale = 1.0
        bias = 0.5
        weight_norm = True

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            elif n_harmonic_functions_xyz > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif n_harmonic_functions_xyz > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        assert density_activation in ['softplus', 'relu']
        if density_activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            self.activation = nn.ReLU(True)

        # Color layer
        if not self._densities_only:
            # Intermediate linear of density
            self.intermediate_linear = torch.nn.Linear(
                n_hidden_neurons_xyz, n_hidden_neurons_xyz
            )
            torch.nn.init.kaiming_uniform_(self.intermediate_linear.weight.data, nonlinearity='linear')

            color_layer = [
                LinearWithRepeat(
                    n_hidden_neurons_xyz + embedding_dim_dir + 3, n_hidden_neurons_dir #特征+方向+原始坐标(3)
                ),
                torch.nn.LayerNorm(n_hidden_neurons_dir) if norm else torch.nn.Identity(),
                #torch.nn.ELU(inplace=True),
                torch.nn.ReLU(True),
                torch.nn.Linear(n_hidden_neurons_dir,n_hidden_neurons_dir),
                torch.nn.LayerNorm(n_hidden_neurons_dir) if norm else torch.nn.Identity(),
                torch.nn.ReLU(True),
                torch.nn.Linear(n_hidden_neurons_dir,n_hidden_neurons_dir),
                torch.nn.LayerNorm(n_hidden_neurons_dir) if norm else torch.nn.Identity(),
                torch.nn.ReLU(True),
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

        assert density_activation in ['softplus', 'relu']
        self._density_activation = density_activation
    
    def density_layer(self, inputs): #inputs = pts
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x[..., :1], x[..., 1:]

    def get_sdf(
            self,
            ray_bundle: RayBundle,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        #embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)

        #nn.functional.normalize(rays_points_world,dim=-1)
        sdf, _ = self.density_layer(rays_points_world)
    
        return rays_points_world, sdf

    def get_sdf_withpts(
            self,
            pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #rays_points_world = ray_bundle_to_ray_points(ray_bundle)

        #embeds_xyz = self.harmonic_embedding_xyz(pts)
        #nn.functional.normalize(pts,dim=-1)
        sdf, _ = self.density_layer(pts)
    
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
        y, _ = self.density_layer(x)

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
            self, features: torch.Tensor, rays_directions: torch.Tensor ,rays_points_world: torch.Tensor
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

        return self.color_layer((self.intermediate_linear(features), rays_embedding, rays_points_world))

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

        if rays_points_world.grad_fn == None:
            x = rays_points_world.clone()
            x.requires_grad_(True)
            #embeds_xyz = self.harmonic_embedding_xyz(x)
            sdf, features = self.density_layer(x)
        else:
        # For each 3D world coordinate, we obtain its harmonic embedding.
            #embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
            
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # (n, i, p, 1)
        #assert not (self.training and return_raw_density)
            sdf, features = self.density_layer(rays_points_world)
        rays_colors = self._get_colors(features, ray_bundle.directions ,rays_points_world)

        '''d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]'''

        return sdf.reshape(batch, chunk, sample, 1), rays_colors.reshape(batch, chunk, sample, 3)#, rays_points_world.reshape(batch, chunk, sample, 3)
