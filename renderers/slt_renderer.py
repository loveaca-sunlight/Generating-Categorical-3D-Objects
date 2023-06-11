import typing

import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.raysampling import RayBundle as RayBunle_i
from pytorch3d.renderer import RayBundle

from loss_functions import calc_weights_reg, KeyPointMatchLoss
from modules.multiplex_new import TransformMultiplexDict
from nerf.utils import sample_images_at_mc_locs
from .renderer_base import RendererBase
from .util import select_template_weights, select_template_weights_s, select_template_weights_r

from nerf.raysampler import NeRFRaysampler, ProbabilisticRaysampler, NeuSRaysampler
from nerf.raymarcher import SDFNeRFRaymarcher
from pytorch3d.renderer import ImplicitRenderer
from render_functions import IMPLICIT_FUNCTIONS

from typing import Tuple


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]).cuda()
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), 3, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), 3, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def ray_bundle_variables_to_ray_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
) -> torch.Tensor:
    rays_points = (
        rays_origins[..., None, :]
        + rays_lengths[..., :, None] * rays_directions[..., None, :]
    )
    return rays_points

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)

class SilhouetteRenderer(RendererBase):
    """
    A renderer that only renders silhouette
    """
    def __init__(
            self,
            sequences: typing.List[str],
            image_size: typing.Tuple[int, int],
            n_pts_per_ray: int,
            # n_pts_per_ray_fine: int,  only coarse
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            function_config: dict,
            mask_thr: float,
            n_transform_ways: int,
            template_key_points: torch.Tensor,
            match_tol: float,
            density_noise_std: float,
            edge_thr: float = 0,
            negative_z: bool = False
    ):
        super(RendererBase, self).__init__()
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        self.raymarcher = SDFNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        #for render_pass in render_passes: #没有先粗后精， 而是用sdf周围密度采样,这里还没改完

        self.raysampler = NeuSRaysampler(
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified=stratified,
            stratified_test=stratified_test,
            n_rays_per_image=n_rays_per_image,
            image_height=image_height,
            image_width=image_width,
            negative_z=negative_z,
            edge_thr=edge_thr
        )

        # Initialize the fine/coarse renderer.
        '''self._renderer = ImplicitRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        ) '''
        #1、ray_bundle = self.raysampler 2、rays_densities, rays_features = volumetric_function 3、images = self.raymarcher
        #试图重新写一份renderer

        # Instantiate the fine/coarse NeuralRadianceField module.
        self.volumetric_function = IMPLICIT_FUNCTIONS.build(function_config)

        # transform multiplexes
        self.transforms = TransformMultiplexDict(sequences, n_transform_ways)

        # key point match loss
        self.match_loss = KeyPointMatchLoss(
            target_points=template_key_points,
            tol=match_tol
        )

        self.valida = True

        self._mask_thr = mask_thr
        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        #self.near = min_depth
        #self.far = max_depth
        self.n_samples = n_pts_per_ray
        #这里手动调整 懒得写yaml了  精采样全部取0
        self.up_sample_steps = 2
        self.n_importance = 32
        init_val = 550.0
        self.deviation_network = SingleVarianceNetwork(init_val=init_val).to(torch.device('cuda'))

    def load_template_weights(self, state_dict: dict):
        """
        Only suit for template_mlp
        :param state_dict:
        :return:
        """
        if True:
            weights = select_template_weights_r(state_dict, fine_only=True)
            # Load Weights only from the fine function
            with torch.no_grad():
                for module_name, module in [('mlp_xyz', self.volumetric_function.mlp_xyz),
                                            ('density_layer', self.volumetric_function.density_layer)]:
                    for name, param in module.named_parameters():
                        weight_name = f'volumetric_function.{module_name}.{name}'
                        param.data.copy_(weights[weight_name])
        else:
            # Select Weights from fine pass
            weights = select_template_weights_s(state_dict, fine_only=True)

            # Load Weights from fine pass
            with torch.no_grad():
                for module_name, module in [('lin'+str(l), getattr(self.volumetric_function,"lin" + str(l))) for l in range(0,9)]:
                    for name, param in module.named_parameters():
                        param.data.copy_(weights[f'volumetric_function.{module_name}.{name}'])

    def freeze_template_weights(self):
        self.volumetric_function.freeze_template_layers()

    def transform_parameters(self):
        """
        Return parameters of transforms
        :return:
        """
        return self.transforms.parameters()

    def _get_transform_code(self, sequence_name: str):
        # (m, d)
        return self.transforms(sequence_name)

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            transform_codes: torch.Tensor,
            key_points: torch.Tensor,
            chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """
        # First evaluate the coarse rendering pass, then the fine one.
        (rgb, weights, _), ray_bundle_out = self._renderer(
                cameras=target_camera,

                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=0.0,
                camera_hash=None,

                transform_codes=transform_codes,
                key_points=key_points,

                fg_probability=target_mask,
                fg_thr=self._mask_thr,
                valid = self.valida
            )

        n_multiplexes = self.transforms.num_multiplexes

        if target_image is not None:
            # Sample the ground truth images at the xy locations of the
            # rendering ray pixels.
            with torch.no_grad():
                rgb_gt = sample_images_at_mc_locs(
                    target_image.permute(0, 2, 3, 1).contiguous(),
                    ray_bundle_out.xys,
                ).repeat(n_multiplexes, 1, 1)  # (m * n, i, 3)
        else:
            rgb_gt = None

        # Sample the ground truth fg_probability
        if target_fg_probability is not None:
            with torch.no_grad():
                fg_gt = sample_images_at_mc_locs(
                    target_fg_probability.permute(0, 2, 3, 1).contiguous(),
                    ray_bundle_out.xys
                ).repeat(n_multiplexes, 1, 1)  # (m * n, i, 1)
        else:
            fg_gt = None


        if self.valida:
            weights_coarse = weights.sum(dim=-1)
        else:
            # Sum of weights along a ray, (n, i)
            weights_coarse = weights.sum(dim=-1).detach()
        # Sum of weights along a ray, (m * n, i)

        # render depth
        depth_coarse = torch.sum(weights * ray_bundle_out.lengths.repeat(n_multiplexes, 1, 1),
                                    dim=-1, keepdim=True)  # (m * n, i, 1)

        # key points
        transformed_key_points = self.volumetric_function.transformed_key_points

        out = {"rgb_gt": rgb_gt}

        out.update({
            'depth_coarse': depth_coarse,
            'fg_gt': fg_gt,
            'weights_coarse': weights_coarse,
            'key_points': transformed_key_points
        })

        return out

    def _renderer(
        self, cameras: CamerasBase, **kwargs
    ) -> Tuple[torch.Tensor, RayBunle_i]:
        """
        Render a batch of images 
        渲染采样点随着SDF进行更新update

        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.

        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `RayBundle` containing the parametrizations of the
                sampled rendering rays.
        """
        # first call the ray sampler that returns the RayBundle parametrizing
        # the rendering rays.
        ray_bundle = self.raysampler(
            cameras=cameras, **kwargs  #好像raysampler用不到volumetric
        )
        # ray_bundle.origins - minibatch x ... x 3
        # ray_bundle.directions - minibatch x ... x 3
        # ray_bundle.lengths - minibatch x ... x n_pts_per_ray
        # ray_bundle.xys - minibatch x ... x 2

        # given sampled rays, call the volumetric function that
        # evaluates the densities and features at the locations of the
        # ray points
        pts_big = ray_bundle_variables_to_ray_points(ray_bundle[0],ray_bundle[1],ray_bundle[2])
        scale = 8.
        pts = pts_big / scale
        rays_o = ray_bundle.origins / scale
        z_vals = ray_bundle.lengths / scale
        rays_d = ray_bundle.directions

        if self.n_importance > 0:
            with torch.no_grad():
                sdf = self.volumetric_function.get_sdf_withpts(pts)
                sdf = sdf.squeeze(-1)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o=rays_o,
                                                rays_d=rays_d,
                                                z_vals=z_vals,
                                                sdf=sdf, 
                                                n_importance=self.n_importance // self.up_sample_steps,
                                                inv_s=64 *2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o=rays_o,
                                                  rays_d=rays_d,
                                                  z_vals=z_vals,
                                                  xys=ray_bundle[3],
                                                  new_z_vals=new_z_vals,
                                                  sdf=sdf,
                                                  last=(i + 1 == self.up_sample_steps))
                    ray_bundle = ray_bundle._replace(lengths=z_vals)
        
        #目前不确定mid_z_val的好处在哪？
        sample_dist = 2.0 / self.n_samples # Assuming the region of interest is a unit sphere [-1 1] or [-8 8]
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        ray_bundle = ray_bundle._replace(lengths=mid_z_vals)
        ray_bundle = ray_bundle._replace(origins=rays_o)
        pts = ray_bundle_variables_to_ray_points(ray_bundle[0],ray_bundle[1],ray_bundle[2])
        old_ray_bundle=ray_bundle
        
        rays_sdf, rays_features, ray_bundle = self.volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )

        #new *4 raybundle
        pts = ray_bundle_variables_to_ray_points(ray_bundle[0],ray_bundle[1],ray_bundle[2])
        dists = ray_bundle[2][..., 1:] - ray_bundle[2][..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)

        # ray_sdf - minibatch x ... x n_pts_per_ray x sdf_dim (1)
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim (3)
        if not self.valida:
            torch.set_grad_enabled(True)
            gradient = self.volumetric_function.get_gradient(pts)
            #gradient = None
            gradients = gradient.detach()
            del gradient
            torch.set_grad_enabled(False)
        else:
            gradients = self.volumetric_function.get_gradient(pts)
    
        batch_size, chunk, n_samples, _ = rays_sdf.shape 
        #get inv
        #inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = torch.tensor(550.).cuda()
        inv_s = inv_s.expand(batch_size * chunk * n_samples, 1)

        cos_anneal_ratio = 1.0
        dirs = ray_bundle.directions[...,None,:].expand(pts.shape)
        pts = pts.reshape(-1,3)
        dirs = dirs.reshape(-1,3)
        gradients = gradients.reshape(-1,3)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        # 现在默认1.0
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = rays_sdf.reshape(-1,1) + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = rays_sdf.reshape(-1,1) - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, chunk, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, chunk, n_samples)
        #inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < torch.linalg.norm(torch.tensor([1.,1.,1.]),ord=2)).float().detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, chunk, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[..., :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (rays_features * weights[..., None]).sum(dim=-2)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, chunk, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        # images - minibatch x ... x (feature_dim + opacity_dim)

        return (color, weights, gradient_error), old_ray_bundle

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, chunk, n_samples = z_vals.shape
        pts = rays_o[... , None, :] + rays_d[... , None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[..., :-1] < 1.0) | (radius[..., 1:] < 1.0)
        sdf = sdf.reshape(batch_size, chunk, n_samples)
        prev_sdf, next_sdf = sdf[..., :-1], sdf[..., 1:]
        prev_z_vals, next_z_vals = z_vals[..., :-1], z_vals[..., 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5) #sdf和距离的比例  sdf衰减若比距离快则大于1

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, chunk, 1]).cuda(), cos_val[..., :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere #为什么要只取负值？ cosval为负的时候就是sdf在衰减，说明从外部逼近边缘

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5 #通过均分比例计算出来前后大约是多少sdf值
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s) #负数的sdf就视作实体
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) #推导出的alpha公式 见NeuS第五页
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, chunk, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[..., :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, xys, new_z_vals, sdf, last=False):
        batch_size, chunk, n_samples = z_vals.shape
        _, _, n_importance = new_z_vals.shape
        #pts = rays_o[..., None, :] + rays_d[..., None, :] * new_z_vals[..., :, None]
        New_RayBundle = RayBundle(rays_o,rays_d,new_z_vals,xys)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            _,new_sdf = self.volumetric_function.get_sdf(New_RayBundle)
            new_sdf = new_sdf.squeeze(-1)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            for i in range(batch_size):
                xx = torch.arange(chunk)[..., None].expand(chunk, n_samples + n_importance).reshape(-1)
                index_i = index[i].reshape(-1)
                sdf[i] = sdf[i][(xx, index_i)].reshape(chunk, n_samples + n_importance)

        return z_vals, sdf

    def update_multiplex_score(self, sequence_name: str, scores: torch.Tensor):
        self.transforms.update_scores(sequence_name, scores)

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            key_points: torch.Tensor,
            key_ids: typing.List[int],
            valida: bool,
            sequence_name: str
    ) -> typing.Tuple[dict, dict]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            target_camera:
            target_image: can be None
            target_fg_probability:
            target_mask: probability of one pixel can be foreground or mask to indicate valid area
            key_points: (1, 2, 3)
            key_ids: the indexes of key points
            sequence_name:
        """
        self.valida = valida

        batch_size = target_camera.R.shape[0]
        n_multiplexes = self.transforms.num_multiplexes

        # get rigid transform codes 为什么是4*10
        rigid_codes = self._get_transform_code(sequence_name)

        if not self.valida:
            # Full evaluation pass.
            n_chunks = self.raysampler.get_n_chunks(
                self._chunk_size_test,
                batch_size,  # n_multiplexes = 1 when self.training == false
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                target_camera=target_camera,
                target_image=target_image,
                target_fg_probability=target_fg_probability,
                target_mask=target_mask,
                transform_codes=rigid_codes,
                key_points=(key_points if chunk_idx == 0 else None),
                chunk_idx=chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.valida:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            outs = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_gt",)
            }

            outs.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 1)  # n_multiplexes = 1 when self.training == false
                if chunk_outputs[0][k] is not None
                else None
                for k in ("depth_coarse", 'fg_gt')
            })

            outs.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size)  # n_multiplexes = 1 when self.training == false
                if chunk_outputs[0][k] is not None
                else None
                for k in ("weights_coarse", )
            })

            outs['key_points'] = chunk_outputs[0]['key_points']
        else:
            outs = chunk_outputs[0]

        #
        # Calc error metrics
        #

        # Dict[Tensor] to List[Dict[Tensor]]
        outs = {
            name: torch.chunk(tensor, n_multiplexes, dim=0) if tensor is not None else None
            for name, tensor in outs.items()
        }
        outs = [
            {name: (tensors[i] if tensors is not None else None) for name, tensors in outs.items()}
            for i in range(n_multiplexes)
        ]

        # Compute metrics for each transform multiplex
        multiplex_metrics = []
        render_pass = "coarse"
        for out in outs:
            metrics = {}

            # Calc weights reg
            fg_probability = out['fg_gt']
            fg_mask = (fg_probability > 0.5).float()
            metrics[f'weights_reg_{render_pass}'] =calc_weights_reg(out[f'weights_{render_pass}'],
                                                                         fg_probability.squeeze(-1))

            # Calc match loss
            transformed_points = out['key_points']
            metrics['match_loss'] = self.match_loss(
                input_points=transformed_points,
                input_ids=key_ids
            )

            # append to list
            multiplex_metrics.append(metrics)

        # each element stores all outputs and metrics
        outs = outs[0] if len(outs) == 1 else outs
        multiplex_metrics = multiplex_metrics[0] if len(multiplex_metrics) == 1 else multiplex_metrics

        return outs, multiplex_metrics
