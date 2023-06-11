"""
Renderer for basemlp
"""
import collections
from typing import Tuple, List

import torch
from pytorch3d.renderer.cameras import CamerasBase

from loss_functions import calc_weights_reg, calc_mse_prob, weight_norm_l2
from modules.encoder import ENCODERS
from nerf.utils import calc_psnr, sample_images_at_mc_locs, calc_mse
from .renderer_base import RendererBase


class BaseRenderer(RendererBase):
    """
    Model the density, diffused color and aleatoric uncertainty
    """

    def __init__(
            self,
            image_size: Tuple[int, int],
            n_pts_per_ray: int,
            n_pts_per_ray_fine: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            stratified: bool,
            stratified_test: bool,
            chunk_size_test: int,
            function_config: dict,
            encoder_config: dict,
            mask_thr: float,
            weighted_mse: bool,
            density_noise_std: float
    ):
        super().__init__(
            image_size=image_size,
            n_pts_per_ray=n_pts_per_ray,
            n_pts_per_ray_fine=n_pts_per_ray_fine,
            n_rays_per_image=n_rays_per_image,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified=stratified,
            stratified_test=stratified_test,
            chunk_size_test=chunk_size_test,
            implicit_function_config=function_config,
            mask_thr=mask_thr,
            density_noise_std=density_noise_std
        )

        # Encoder
        self.encoder = ENCODERS.build(encoder_config)
        print(f'Latent Code Encoder: {type(self.encoder).__name__}')

        # # Transform
        # if sequences is not None:
        #     self.transforms = TransformDict(sequences)
        # else:
        #     print('No transforms are registered, since sequences is None.')

        # # Deformable Field, no registration to self.modules
        # self.deformer = HyperDeformationField(**deformer_config)
        #
        # # Set deformable field of implicit functions
        # for renderer_pass in ['coarse', 'fine']:
        #     self._implicit_function[renderer_pass].set_deformable_field(self.deformer)

        # Other parameters
        # self._pairs_per_image = pairs_per_image
        # self._epsilon = epsilon
        self._weighted_mse = weighted_mse

        print(f'Weighted MSE: {self._weighted_mse}.')

    def load_transforms(self, state_dict: collections.OrderedDict):
        """
        Load transforms of each sequence
        :param state_dict:
        :return:
        """
        pass

    def load_template_weights(self, state_dict: dict):
        """
        Only suit for template_mlp
        :param state_dict:
        :return:
        """
        pass

    def template_parameters(self):
        """
        An iterator that yield parameters in template modules
        :return:
        """
        pass

    def transform_parameters(self):
        """
        An iterator that yield parameters in transform dict
        :return:
        """
        # return self.transforms.parameters()
        pass

    def encoder_parameters(self):
        """
        Return parameters of encoder
        :return:
        """
        return self.encoder.parameters()

    def hyper_parameters(self):
        """
        Return parameters of hyper network
        :return:
        """
        for func in self._implicit_function.values():
            yield from func.hyper_parameters()
        # yield from self.deformer.hyper_parameters()

    def rest_parameters(self):
        """
        An iterator that yield parameters beyond other modules
        :return:
        """
        encoder_parameters = set(self.encoder_parameters())
        hyper_parameters = set(self.hyper_parameters())
        for param in self.parameters():
            if (param not in encoder_parameters) and (param not in hyper_parameters):
                yield param

    def encode_codes(self, source_images: torch.Tensor):
        shape_code, color_code = self.encoder(source_images)

        return shape_code, color_code

    def produce_parameters(self, shape_code: torch.Tensor, color_code: torch.Tensor):
        parameters = []
        for func in self._implicit_function.values():
            parameters.extend(func.produce_parameters(shape_code, color_code))
        return parameters

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            transform_code: torch.Tensor,
            enable_deformation: bool,
            enable_specular: bool,
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
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (features, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=target_camera,
                volumetric_function=self._implicit_function[renderer_pass],

                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=None,

                transform_code=transform_code,
                enable_deformation=enable_deformation,
                enable_specular=enable_specular,

                fg_probability=target_mask,
                fg_thr=self._mask_thr
            )

            if renderer_pass == "coarse":
                rgb_coarse = features[..., : 3]   # (n, i, 3)
                depth_coarse = features[..., 3: 4]  # (n, i, 1)

                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights

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

                # Sample the ground truth fg_probability
                if target_fg_probability is not None:
                    with torch.no_grad():
                        fg_gt = sample_images_at_mc_locs(
                            target_fg_probability.permute(0, 2, 3, 1).contiguous(),
                            ray_bundle_out.xys
                        )  # (n, i, 1)
                else:
                    fg_gt = None

                # Sum of weights along a ray, (n, i)
                weights_coarse = weights.sum(dim=-1)

            elif renderer_pass == "fine":
                rgb_fine = features[..., : 3]   # (n, i, 3)
                depth_fine = features[..., 3: 4]  # (n, i, 1)

                # Sum of weights along a ray, (n, i)
                weights_fine = weights.sum(dim=-1)

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        out = {"rgb_fine": rgb_fine, "rgb_coarse": rgb_coarse, "rgb_gt": rgb_gt}

        out.update({
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine,
            'fg_gt': fg_gt,
            'weights_coarse': weights_coarse,
            'weights_fine': weights_fine
        })

        return out

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            source_image: torch.Tensor,
            sequence_name: str,
            enable_deformation: bool = True,
            enable_specular: bool = True,
            shape_code: torch.Tensor = None,
            color_code: torch.Tensor = None,
            transform_code: torch.Tensor = None,
            produce_parameters: bool = True
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

        # # get rigid transform code
        # rigid_latent_code = (self._get_transform_code(sequence_name) if transform_code is None else transform_code)
        # # assert (not self.training) or (rigid_code is not None), 'rigid_code can not be None when training.'
        rigid_latent_code = transform_code

        # get latent code
        if enable_deformation:
            if (shape_code is not None) or (color_code is not None):
                assert (shape_code is not None) and (color_code is not None), \
                    f'Shape and color code must be provided together.'
                shape_latent_code, color_latent_code = shape_code, color_code
            else:
                shape_latent_code, color_latent_code = self.encode_codes(source_image)
        else:
            assert not self.training
            shape_latent_code = None
            color_latent_code = None

        # generate weights
        hyper_parameters = []
        if produce_parameters and shape_latent_code is not None:
            hyper_parameters = self.produce_parameters(shape_latent_code, color_latent_code)
        else:
            assert not self.training

        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                batch_size,
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
                transform_code=rigid_latent_code,
                enable_deformation=enable_deformation,
                enable_specular=enable_specular,
                chunk_idx=chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }

            out.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 1)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("depth_fine", "depth_coarse")
            })
        else:
            out = chunk_outputs[0]

        # Calc the psnr metrics.
        metrics = {}
        if target_image is not None:
            with torch.no_grad():
                for render_pass in ("coarse", "fine"):
                    metrics[f"psnr_{render_pass}"] = calc_psnr(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                        target_mask.permute(0, 2, 3, 1).contiguous() if not self.training else None
                    )

        if self.training:
            for render_pass in ('coarse', 'fine'):
                # mse loss
                if self._weighted_mse:
                    metrics[f'mse_{render_pass}'] = calc_mse_prob(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                        out['fg_gt']
                    )
                else:
                    metrics[f'mse_{render_pass}'] = calc_mse(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                        None
                    )

            # Calc points deformation regularization
            for render_pass in ('fine', ):  # compute deformation loss only for fine pass
                metrics[f'deformation_reg_{render_pass}'] = 0.0
                metrics[f'deformation_loss_{render_pass}'] = 0.0

            # Calc mask reg
            for render_pass in ("coarse", "fine"):
                metrics[f'weights_reg_{render_pass}'] = calc_weights_reg(
                    out[f'weights_{render_pass}'],
                    out['fg_gt'].squeeze(-1)
                )

            # Calc points specular regularization
            # points_specular = out['points_specular']
            for render_pass in ('coarse', 'fine'):
                metrics[f'specular_reg_{render_pass}'] = 0  # calc_l2_reg(points_specular[render_pass])

            # Calc l1 norm of generated weights
            param_norm = 0.0
            for w in hyper_parameters:
                param_norm += weight_norm_l2(w)
            metrics['weights_norm'] = param_norm

        return out, metrics
