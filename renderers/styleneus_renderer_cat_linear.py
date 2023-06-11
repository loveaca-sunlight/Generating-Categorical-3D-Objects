import collections
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.raysampling import RayBundle as RayBunle_i
from pytorch3d.renderer import RayBundle

from loss_functions import calc_l2_reg, calc_weights_reg, calc_deformation_loss, calc_mse_prob, weight_norm_l2
from modules.defnerf import HyperDeformationField
from modules.encoder import ENCODERS
from modules.multiplex_new import TransformDict
from nerf.utils import calc_psnr, sample_images_at_mc_locs, calc_mse
from .renderer_base import RendererBase
from .util import select_template_weights_hyper, img2mse

from nerf.raymarcher import SDFNeRFRaymarcher
from nerf.raysampler import NeRFRaysampler, ProbabilisticRaysampler, NeuSRaysampler, BatchRaysampler
from pytorch3d.renderer import ImplicitRenderer
from render_functions import IMPLICIT_FUNCTIONS
import mcubes

class StyleNeusRendererCatLinear(RendererBase):
    """
    Model the density, diffused color and aleatoric uncertainty
    """

    def __init__(
            self,
            sequences: List[str],
            image_size: Tuple[int, int],
            n_pts_per_ray: int,
            n_pts_importance: int,
            up_sample_steps: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            function_config: dict,
            deformer_config: dict,
            encoder_config: dict,
            mask_thr: float,
            pairs_per_image: int,
            epsilon: float,
            weighted_mse: bool,
            density_noise_std: float,
            negative_z: bool = False,
            edge_thr: float = 1.,
    ):
        #render_base init的内容被直接提到了这里
        super(RendererBase, self).__init__()
        #self._renderer = torch.nn.ModuleDict()
        self.volumetric_function = torch.nn.ModuleDict()

         # Init the EA raymarcher used by both passes.
        # self.raymarcher = SDFNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        #for render_pass in render_passes: #没有先粗后精， 而是用sdf周围密度采样

        self.raysampler = BatchRaysampler(
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


        # Encoder
        self.encoder = ENCODERS.build(encoder_config)
        print(f'Latent Code Encoder: {type(self.encoder).__name__}')

        # Encoder2
        self.style_encoder = ENCODERS.build({'type':'style_encoder'})
        print(f'Latent Code Encoder: {type(self.style_encoder).__name__}')

        # Transform
        if sequences is not None:
            self.transforms = TransformDict(sequences)
            self.deviations = DeviationDict(sequences,0.640)
        else:
            print('No transforms are registered, since sequences is None.')
        
        # Deformable Field, no registration to self.modules
        self.deformer = HyperDeformationField(**deformer_config)

        self.shape_pro = nn.Linear(function_config.color_code_dim * 2,function_config.color_code_dim)

        # Instantiate the fine/coarse NeuralRadianceField module.
        self.volumetric_function = IMPLICIT_FUNCTIONS.build(function_config)

        # Set deformable field of implicit functions
        self.volumetric_function.set_deformable_field(self.deformer)

        # Other parameters
        self._pairs_per_image = pairs_per_image
        self._epsilon = epsilon
        self._weighted_mse = weighted_mse
        self._mask_thr = mask_thr
        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self._n_rays_per_image = n_rays_per_image
        #self.near = min_depth
        #self.far = max_depth
        self.n_samples = n_pts_per_ray
        self.up_sample_steps = up_sample_steps
        self.n_importance = n_pts_importance
        #init_val = 550.
        #self.deviation_network = SingleVarianceNetwork(init_val=init_val).to(torch.device('cuda'))

        print(f'Weighted MSE: {self._weighted_mse}.')

    def load_transforms(self, state_dict: collections.OrderedDict):
        """
        Load transforms of each sequence
        :param state_dict:
        :return:
        """
        prefix = 'transforms'

        state_dict = collections.OrderedDict(
            [
                (f'{prefix}.{name}', param)
                for name, param in state_dict.items()
            ]
        )

        self.transforms.load_pretrained_transforms(state_dict)

    def load_weights(self, state_dict: dict):
        """
        Load hyper and encoder1 and template weight
        :param state_dict:
        :return:
        """
        state_dict = select_template_weights_hyper(state_dict=state_dict)
        # Load Weights only from the fine function
        with torch.no_grad():
            for module_name, module in [('mlp_xyz', self.volumetric_function.mlp_xyz),
                                        ('density_layer', self.volumetric_function.density_layer),
                                        ('mlp_diffuse', self.volumetric_function.mlp_diffuse),
                                        ('diffuse_layer', self.volumetric_function.diffuse_layer),]:
                for name, param in module.named_parameters():
                    weight_name = f'volumetric_function.{module_name}.{name}'
                    param.data.copy_(state_dict[weight_name])
            for module_name, module in [('encoder',self.encoder),
                                        ('deviations',self.deviations),
                                        ('deformer',self.deformer)]:
                for name, param in module.named_parameters():
                    weight_name = f'{module_name}.{name}'
                    param.data.copy_(state_dict[weight_name])

    def template_parameters(self):
        """
        An iterator that yield parameters in template modules
        :return:
        """
        #if 'Res' in self.volumetric_function.__class__.__name__ or 'Cat' in self.volumetric_function.__class__.__name__:
        if True:
            for module in [self.volumetric_function.mlp_xyz,
                            self.volumetric_function.density_layer]:
                    yield from module.parameters()
        else:
            for module in [getattr(self.volumetric_function,"lin" + str(l)) for l in range(0,9)]:
                yield from module.parameters()

    def transform_parameters(self):
        """
        An iterator that yield parameters in transform dict
        :return:
        """
        return self.transforms.parameters()

    def deviation_parameters(self):
        return self.deviations.parameters()

    def shapepro_parameters(self):
        return self.shape_pro.parameters()

    def encoder_parameters(self):
        """
        Return parameters of encoder
        :return:
        """
        return self.encoder.parameters()
    
    def encoder2_parameters(self):
        """
        Return parameters of encoder
        :return:
        """
        return self.style_encoder.parameters()

    def hyper_color_parameters(self):
        """
        Return parameters of hyper network
        :return:
        """
        #for func in self.volumetric_function:
        yield from self.volumetric_function.hyper_parameters()
    
    def hyper_shape_parameters(self):
        """
        Return parameters of hyper network
        :return:
        """
        #for func in self.volumetric_function:
        yield from self.deformer.hyper_parameters()

    def rest_parameters(self):
        """
        An iterator that yield parameters beyond other modules
        :return:
        """
        template_parameters = set(self.template_parameters())
        transform_parameters = set(self.transform_parameters())
        shapepro_parameters = set(self.shapepro_parameters())
        encoder_parameters = set(self.encoder_parameters())
        encoder2_parameters = set(self.encoder2_parameters())
        hyper_color_parameters = set(self.hyper_color_parameters())
        hyper_shape_parameters = set(self.hyper_shape_parameters())
        deviation_parameters = set(self.deviation_parameters())
        for param in self.parameters():
            if (param not in template_parameters) and (param not in transform_parameters) and \
                    (param not in encoder_parameters) and (param not in encoder2_parameters) and (param not in shapepro_parameters) and \
                          (param not in hyper_color_parameters) and (param not in hyper_shape_parameters) and (param not in deviation_parameters):
                yield param

    def _get_transform_code(self, sequence_name: str):
        # (1, d)
        if sequence_name is not None:
            return self.transforms(sequence_name)
        else:
            return None
    
    def _get_deviation(self, sequence_name: str):
        # (1, d)
        if sequence_name is not None:
            return self.deviations(sequence_name)
        else:
            return None

    def encode_codes(self, source_images: torch.Tensor):
        shape_code, color_code = self.encoder(source_images)

        return shape_code, color_code

    def produce_parameters(self, shape_code: torch.Tensor, color_code: torch.Tensor, style_code: torch.Tensor):
        parameters = []
        parameters.extend(self.deformer.produce_parameters(shape_code))#shape
        #for func in self.volumetric_function.values():
        hybrid_code = self.shape_pro(torch.cat((color_code,style_code),dim=1))
        parameters.extend(self.volumetric_function.produce_parameters(hybrid_code))#color
        #parameters.extend(self.volumetric_function.produce_parameters(color_code))
        return parameters

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            transform_code: torch.Tensor,
            deviation: torch.Tensor,
            enable_deformation: bool,
            enable_specular: bool,
            chunk_idx: int,
            ray_sampler_config: str,
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
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # Store points deformation, position and reflectance
        points_deformation = {}
        points_position = {}
        points_specular = {}

        # First evaluate the coarse rendering pass, then the fine one.
        (features, weights, gradient, gradient_error), ray_bundle_out = self._renderer(
            cameras=target_camera,

            chunksize=self._chunk_size_test,
            chunk_idx=chunk_idx,
            density_noise_std=(self._density_noise_std if self.valida else 0.0),
            camera_hash=None,
            ray_sampler_config=ray_sampler_config,

            transform_code=transform_code,
            deviation=deviation,
            enable_deformation=enable_deformation,
            enable_specular=enable_specular,

            fg_probability=target_mask,
            fg_thr=self._mask_thr,

            valid = self.valida
        )

        # store
        if self.training:
            points_specular = self.volumetric_function.latest_points_specular
            points_deformation = self.volumetric_function.latest_points_deformation
            points_position = self.volumetric_function.latest_points_position

        if target_image is not None:
            # Sample the ground truth images at the xy locations of the
            # rendering ray pixels.
            with torch.no_grad():
                rgb_gt = sample_images_at_mc_locs(
                    target_image.permute(0, 2, 3, 1).contiguous(),
                    ray_bundle_out.xys,
                )
        else:
            rgb_gt = None

        if target_fg_probability is not None:
            with torch.no_grad():
                fg_gt = sample_images_at_mc_locs(
                    target_fg_probability.permute(0, 2, 3, 1).contiguous(),
                    ray_bundle_out.xys
                )  # (n, i, 1)
        else:
            fg_gt = None

        rgb = features[..., : 3]
        depth = features[..., 3: 4]
        #normal = features[..., 4: 7]
        normal = torch.sum(gradient * weights[:, :, :, None], dim=-2, keepdim=True)
        
        if self.valida:
            rgb_fine = rgb
            depth_fine = depth
            weights_fine = weights.sum(dim=-1)
        else:
            rgb_fine = rgb.detach()
            depth_fine = depth.detach()
            # Sum of weights along a ray, (n, i)
            weights_fine = weights.sum(dim=-1).detach()

        out = {"rgb_fine": rgb_fine, "rgb_gt": rgb_gt, 'normal': normal}

        out.update({
            'depth_fine': depth_fine,
            'fg_gt': fg_gt,
            'weights_fine': weights_fine,
            'gradient_error': gradient_error
        })

        del rgb,weights,ray_bundle_out

        if self.training:  # when training
            out['points_deformation'] = points_deformation
            out['points_position'] = points_position
            out['points_specular'] = points_specular

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
        
        scale = 8
        ray_bundle = ray_bundle._replace(lengths=ray_bundle.lengths / scale)
        ray_bundle = ray_bundle._replace(origins=ray_bundle.origins / scale)

        if kwargs['transform_code'] is not None:
            #ray_bundle = ray_bundle
            ray_bundle = self.volumetric_function.transformer(ray_bundle, kwargs['transform_code'])
        else:
            assert not self.training

        pts = ray_bundle_variables_to_ray_points(ray_bundle[0],ray_bundle[1],ray_bundle[2])
        rays_o = ray_bundle.origins
        z_vals = ray_bundle.lengths
        rays_d = ray_bundle.directions

        if self.n_importance > 0:
            with torch.no_grad():
                sdf = self.volumetric_function.get_sdf_withpts(pts, enable_deformation=True)
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

        '''import random
        index1 = torch.LongTensor(random.sample(range(0,40), 1))
        index2 = torch.LongTensor(random.sample(range(40,ray_bundle[2].shape[2]), 20))
        index = torch.cat((index1,index2))
        if kwargs['chunk_idx'] == 0:
            pts_plt = ray_bundle_variables_to_ray_points(ray_bundle[0][0,::60,],ray_bundle[1][0,::60,],torch.index_select(ray_bundle[2][0,::60,],1,index.cuda()))
            np.save(('plt.npy'),pts_plt.cpu().numpy())
        else:
            pts_plt = ray_bundle_variables_to_ray_points(ray_bundle[0][0,::60,],ray_bundle[1][0,::60,],torch.index_select(ray_bundle[2][0,::60,],1,index.cuda()))
            plt=np.load('plt.npy')
            plt_new=np.concatenate((plt,pts_plt.cpu().numpy()),axis=0)
            np.save(('plt.npy'),plt_new)'''
        #np.save(('plt_%s.npy'%kwargs['chunk_idx']),pts_plt.cpu().numpy())
        rays_sdf, rays_features, gradients = self.volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )
        # ray_sdf - minibatch x ... x n_pts_per_ray x sdf_dim (1)
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim (3)
    
        batch_size, chunk, n_samples, _ = rays_sdf.shape 
        #get inv
        inv_s = kwargs['deviation'][:, :1].clip(1e-6, 1e6)
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
        relax_inside_sphere = (pts_norm < torch.linalg.norm(torch.tensor([1.5,1.5,1.5]),ord=2)).float().detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, chunk, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[..., :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (rays_features * weights[..., None]).sum(dim=-2)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, chunk, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        # images - minibatch x ... x (feature_dim + opacity_dim)

        return (color, weights, gradients.reshape(batch_size, chunk, n_samples, 3), gradient_error), ray_bundle

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
        new_pts = ray_bundle_variables_to_ray_points(New_RayBundle[0],New_RayBundle[1],New_RayBundle[2])

        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.volumetric_function.get_sdf_withpts(new_pts, enable_deformation=True)
            new_sdf = new_sdf.squeeze(-1)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            for i in range(batch_size):
                xx = torch.arange(chunk)[..., None].expand(chunk, n_samples + n_importance).reshape(-1)
                index_i = index[i].reshape(-1)
                sdf[i] = sdf[i][(xx, index_i)].reshape(chunk, n_samples + n_importance)

        return z_vals, sdf

    def compute_point_density(
            self,
            target_points: torch.Tensor,
            source_image: torch.Tensor,
            enable_deformation: bool = True,
            chunk_size: int = 9600
    ):
        """
        Compute density of given points
        :param target_points: (1, p, 3)
        :param source_image: (n, c, h, w)
        :param enable_deformation:
        :param chunk_size:
        :return:
        """
        # get codes
        shape_latent_code, _ = self.encode_codes(source_image)
        # produce parameters
        self.deformer.produce_parameters(shape_latent_code)
        # only use fine model
        fine_model = self.volumetric_function

        # inference
        chunks = []
        n_points = target_points.shape[1]
        for i in range(0, n_points, chunk_size):
            points = target_points[:, i: min(i + chunk_size, n_points), :]
            chunks.append(
                fine_model.compute_density(
                    points,
                    enable_deformation
                )
            )
        return torch.cat(
            chunks,
            dim=1
        )

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            source_image: torch.Tensor,
            sequence_name: str,
            valida: bool,
            enable_deformation: bool = True,
            enable_specular: bool = True,
            enable_trans: bool = True,
            style_image: torch.Tensor = None,
            transform_code: torch.Tensor = None,
            epoch: int =0,
            produce_parameters: bool = True,
            ray_sampler_config: str = None,
    ) -> Tuple[dict, dict]:
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
            source_image: images from source view
            sequence_name:
            enable_deformation:
            enable_specular:
            shape_code: offered when testing
            color_code: offered when testing
            transform_code: offered when testing
            produce_parameters: whether to generate new weights
        """
        batch_size = target_camera.R.shape[0]
        self.valida = valida

        # get rigid transform code
        if enable_trans:
            rigid_latent_code = (self._get_transform_code(sequence_name) if transform_code is None else transform_code)
        else:
            rigid_latent_code = None
        # assert (not self.training) or (rigid_code is not None), 'rigid_code can not be None when training.'

        #gei deviation
        deviation = self._get_deviation(sequence_name)

        # get latent code
        if enable_deformation:
            shape_latent_code, color_latent_code = self.encode_codes(source_image)
        else:
            assert not self.training
            shape_latent_code = None
            color_latent_code = None
        
        # get style code
        with torch.no_grad():
            style_code = self.style_encoder(style_image)

        # generate weights
        hyper_parameters = []
        if produce_parameters and shape_latent_code is not None:
            hyper_parameters = self.produce_parameters(shape_latent_code.clone().detach(), color_latent_code.clone().detach(), style_code)
        else:
            assert not self.training

        if not self.valida:
            # Full evaluation pass.
            n_chunks = self.raysampler.get_n_chunks(
                self._chunk_size_test,
                batch_size,
            )
        else:
            # Trainning batch ray sampling.
            n_chunks = 1

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                target_camera=target_camera,
                target_image=target_image,
                target_fg_probability=target_fg_probability,
                target_mask=target_mask,
                transform_code=rigid_latent_code,
                deviation=deviation,
                enable_deformation=enable_deformation,
                enable_specular=enable_specular,
                chunk_idx=chunk_idx,
                ray_sampler_config=ray_sampler_config,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.valida:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_gt")
            }

            k = "depth_fine"
            out.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 1)
                if chunk_outputs[0][k] is not None
                else None
            })

        else:
            out = chunk_outputs[0]
            

        # Calc the psnr metrics.
        # None
        metrics = {}

        if self.valida:
            # mse loss
            # None
            content_out = out['rgb_gt']
            style_out = out['rgb_fine']

            # Style and Content loss
            content_loss, style_loss = self.style_encoder.cal_style_loss(content_out,style_out,style_image,int(self._n_rays_per_image * 1.2),self._n_rays_per_image)

            metrics['content_loss'] = content_loss
            if epoch < 50:
                metrics['style_loss'] = 0.
            else:
                metrics['style_loss'] = style_loss

            rgb_loss = img2mse(style_out, content_out)

            metrics['rgb_loss'] = rgb_loss

            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, int(self._n_rays_per_image * 1.2), self._n_rays_per_image , 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_gt")
            }
            # Calc points deformation regularization
            '''points_deformation = out['points_deformation']
            points_position = out['points_position']
            for render_pass in ('fine', ):  # compute deformation loss only for fine pass
                metrics[f'deformation_reg_{render_pass}'] = calc_l2_reg(points_deformation)
                metrics[f'deformation_loss_{render_pass}'] = calc_deformation_loss(points_deformation,
                                                                                   points_position,
                                                                                   self._pairs_per_image,
                                                                                   self._epsilon)

            # Calc mask reg
            # None

            # Calc points specular regularization
            # points_specular = out['points_specular']
            for render_pass in ('fine', ):
                metrics[f'specular_reg_{render_pass}'] = 0  # calc_l2_reg(points_specular[render_pass])

            # Calc l1 norm of generated weights
            param_norm = 0.0
            for w in hyper_parameters:
                param_norm += weight_norm_l2(w)
            metrics['weights_norm'] = param_norm'''

        return out, metrics

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.volumetric_function.get_sdf_withpts(pts,enable_deformation=False))

class DeviationDict(nn.Module):
    """
    A parameter dict that only stores the deviation
    """

    def __init__(self, sequence_names: List[str], init_val: float):
        super(DeviationDict, self).__init__()

        '''init_parameter = torch.tensor(
            [init_val], dtype=torch.float, requires_grad=False
        )'''

        self.deviations = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(init_val), requires_grad=True)
                for name in sequence_names
            }
        )

        '''with torch.no_grad():
            for param in self.deviations.parameters():
                param.data.copy_(init_parameter)'''

    def forward(self, sequence_name: str):
        return torch.ones([1, 1]).cuda() * torch.exp(self.deviations[sequence_name] * 10.0)

    @property
    def keys(self):
        return self.deviations.keys()

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    #pts = pts[torch.linalg.norm(pts,ord=2,dim=-1) < 1]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


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