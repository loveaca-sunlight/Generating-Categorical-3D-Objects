import collections
from typing import Tuple, List

import torch
from pytorch3d.renderer.cameras import CamerasBase

from loss_functions import calc_l2_reg, calc_weights_reg, calc_deformation_loss, calc_mse_prob
from modules.defnerf import DeformableField
from modules.encoder import ENCODERS
from modules.multiplex import TransformDict
from nerf.utils import calc_psnr, sample_images_at_mc_locs, calc_mse
from .renderer_base import RendererBase
from .util import select_template_weights


class ASRenderer(RendererBase):
    """
    Model the density, diffused color and aleatoric uncertainty
    """

    def __init__(
            self,
            sequences: List[str],
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
            deformable_config: dict,
            encoder_config: dict,
            mask_thr: float,
            pairs_per_image: int,
            epsilon: float,
            weighted_mse: bool,
            density_noise_std: float,
            calibration_mode: bool = False
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

        self._as_mode = function_config.type.split('_')[0]
        assert self._as_mode in ['as0', 'as1', 'as2']

        # Encoder
        self.encoder = ENCODERS.build(encoder_config)
        print(f'Latent Code Encoder: {type(self.encoder).__name__}')

        # Transform
        full_transform = True
        if sequences is not None:
            self.transforms = TransformDict(sequences, full_transform)
        else:
            print('No transforms are registered, since sequences is None.')

        if self._as_mode in ['as1', 'as2']:
            # Deformable Field, no registration to self.modules
            deformer = DeformableField(
                **deformable_config
            )

            # Set deformable field of implicit functions
            for renderer_pass in ['coarse', 'fine']:
                self._implicit_function[renderer_pass].set_deformable_field(deformer)

        # Other parameters
        self._pairs_per_image = pairs_per_image
        self._epsilon = epsilon
        self._weighted_mse = weighted_mse
        self._calibration_mode = calibration_mode

        print(f'Weighted MSE: {self._weighted_mse}, AS Mode: {self._as_mode}, Calibration Mode: '
              f'{self._calibration_mode}.')

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

    def load_template_weights(self, state_dict: dict):
        """
        Only suit for template_mlp
        :param state_dict:
        :return:
        """
        if self._as_mode == 'as0':
            print('No template to load.')
            return None

        # Select Weights
        weights = select_template_weights(state_dict, fine_only=True)

        # Load Weights only from the fine function
        with torch.no_grad():
            for render_pass in ['coarse', 'fine']:
                for module_name, module in [('mlp_xyz', self._implicit_function[render_pass].mlp_xyz),
                                            ('density_layer', self._implicit_function[render_pass].density_layer)]:
                    for name, param in module.named_parameters():
                        param.data.copy_(weights[f'_implicit_function.fine.{module_name}.{name}'])

    def template_parameters(self):
        """
        An iterator that yield parameters in template modules
        :return:
        """
        if self._as_mode in ['as1', 'as2']:
            for render_pass in ['coarse', 'fine']:
                for module in [self._implicit_function[render_pass].mlp_xyz,
                               self._implicit_function[render_pass].density_layer]:
                    yield from module.parameters()
        else:
            return []

    def transform_parameters(self):
        """
        An iterator that yield parameters in transform dict
        :return:
        """
        return self.transforms.parameters()

    def encoder_parameters(self):
        """
        Return parameters of encoder
        :return:
        """
        return self.encoder.parameters()

    def rest_parameters(self):
        """
        An iterator that yield parameters beyond other modules
        :return:
        """
        template_parameters = set(self.template_parameters())
        transform_parameters = set(self.transform_parameters())
        encoder_parameters = set(self.encoder_parameters())
        for param in self.parameters():
            if (param not in template_parameters) and (param not in transform_parameters) and \
                    (param not in encoder_parameters):
                yield param

    def _get_transform_code(self, sequence_name: str):
        # (1, d)
        if sequence_name is not None:
            return self.transforms(sequence_name)
        else:
            return None

    def encode_codes(self, source_images: torch.Tensor):
        shape_code, color_code = self.encoder(source_images)

        return shape_code, color_code

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            shape_latent_code: torch.Tensor,
            color_latent_code: torch.Tensor,
            transform_code: torch.Tensor,
            enable_deformation: bool,
            chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            shape_latent_code:
            color_latent_code:
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

                shape_latent_code=shape_latent_code,
                color_latent_code=color_latent_code,
                transform_code=transform_code,
                enable_deformation=enable_deformation,

                fg_probability=target_mask,
                fg_thr=self._mask_thr
            )

            # store
            if self.training:
                if renderer_pass == 'fine':  # only record when renderer_pass is 'fine'
                    points_deformation[renderer_pass] = self._implicit_function[renderer_pass].latest_points_deformation
                    points_position[renderer_pass] = self._implicit_function[renderer_pass].latest_points_position

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

        if self.training:  # when training
            out['points_deformation'] = points_deformation
            out['points_position'] = points_position

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
            shape_code: torch.Tensor = None,
            color_code: torch.Tensor = None,
            transform_code: torch.Tensor = None
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
            shape_code: offered when testing
            color_code: offered when testing
            transform_code: offered when testing
        """
        batch_size = target_camera.R.shape[0]

        # get rigid transform code
        if self._as_mode == 'as1':
            rigid_latent_code = (self._get_transform_code(sequence_name) if transform_code is None else transform_code)
            # assert (not self.training) or (rigid_code is not None), 'rigid_code can not be None when training.'
        elif self._calibration_mode:
            assert transform_code is not None
            rigid_latent_code = transform_code
        else:
            rigid_latent_code = None

        # get latent code
        if enable_deformation:
            if (shape_code is not None) or (color_code is not None):
                assert (shape_code is not None) and (color_code is not None), \
                    f'Shape and color code must be provided together.'
                shape_latent_code, color_latent_code = shape_code, color_code
            else:
                shape_latent_code, color_latent_code = self.encode_codes(source_image)
        else:
            shape_latent_code = None
            color_latent_code = None

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
                shape_latent_code=shape_latent_code,
                color_latent_code=color_latent_code,
                transform_code=rigid_latent_code,
                enable_deformation=enable_deformation,
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
            if self._as_mode in ['as1', 'as2']:
                points_deformation = out['points_deformation']
                points_position = out['points_position']
                for render_pass in ('fine', ):  # compute deformation loss only for fine pass
                    metrics[f'deformation_reg_{render_pass}'] = calc_l2_reg(points_deformation[render_pass])
                    metrics[f'deformation_loss_{render_pass}'] = calc_deformation_loss(points_deformation[render_pass],
                                                                                       points_position[render_pass],
                                                                                       self._pairs_per_image,
                                                                                       self._epsilon)
            else:
                for render_pass in ('fine', ):  # compute deformation loss only for fine pass
                    metrics[f'deformation_reg_{render_pass}'] = 0.0
                    metrics[f'deformation_loss_{render_pass}'] = 0.0

            # Calc mask reg
            for render_pass in ("coarse", "fine"):
                metrics[f'weights_reg_{render_pass}'] = calc_weights_reg(
                    out[f'weights_{render_pass}'],
                    out['fg_gt'].squeeze(-1)
                )

        return out, metrics
