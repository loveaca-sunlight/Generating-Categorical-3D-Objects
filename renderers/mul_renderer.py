import typing

import torch
from pytorch3d.renderer.cameras import CamerasBase

from loss_functions import calc_l2_reg, calc_weights_reg, calc_deformation_loss
from modules.defnerf import DeformableField
from modules.embedding import LatentCodeEmbeddings
from modules.multiplex import TransformMultiplexDict
from nerf.utils import calc_mse, calc_psnr, sample_images_at_mc_locs
from .renderer_base import RendererBase
from .util import get_name_idx_mapping, select_template_weights


class MulRenderer(RendererBase):
    def __init__(
            self,
            sequences: typing.List[str],
            image_size: typing.Tuple[int, int],
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
            mask_thr: float,
            pairs_per_image: int,
            epsilon: float,
            max_code_norm: float,
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

        # Latent Code Embeddings
        if len(set(sequences)) != len(sequences):
            raise ValueError(f'There are duplicates among sequences.')
        name_idx_mapping = get_name_idx_mapping(sequences)
        # (shape, color)
        embedding_dims = {
            'shape': function_config.get('shape_code_dim'),
            'color': function_config.get('color_code_dim')
        }
        self.latent_embedding = LatentCodeEmbeddings(
            n_embeddings=2,
            embedding_dims=embedding_dims,
            name_idx_mapping=name_idx_mapping,
            max_norm=max_code_norm
        )

        # Rigid Transform Multiplexes
        self.transform_dict = TransformMultiplexDict(sequences)

        # Deformable Field
        self.deformer = DeformableField(
            **deformable_config
        )

        # Set deformable field of implicit functions
        for renderer_pass in ['coarse', 'fine']:
            self._implicit_function[renderer_pass].set_deformable_field(self.deformer)

        self._pairs_per_image = pairs_per_image
        self._epsilon = epsilon

    def load_template_weights(self, state_dict: dict):
        """
        Only suit for template_mlp
        :param state_dict:
        :return:
        """
        # Select Weights
        weights = select_template_weights(state_dict)

        # Load Weights
        with torch.no_grad():
            for render_pass in ['coarse', 'fine']:
                for module_name, module in [('mlp_xyz', self._implicit_function[render_pass].mlp_xyz),
                                            ('density_layer', self._implicit_function[render_pass].density_layer)]:
                    for name, param in module.named_parameters():
                        param.data.copy_(weights[f'_implicit_function.{render_pass}.{module_name}.{name}'])

    def template_parameters(self):
        """
        An iterator that yield parameters in template modules
        :return:
        """
        for render_pass in ['coarse', 'fine']:
            for module in [self._implicit_function[render_pass].mlp_xyz,
                           self._implicit_function[render_pass].density_layer]:
                for param in module.parameters():
                    yield param

    def transform_parameters(self):
        """
        An iterator that yield parameters in transform dict
        :return:
        """
        return self.transform_dict.parameters()

    def non_template_parameters(self):
        """
        An iterator that yield parameters beyond template modules
        :return:
        """
        template_parameters = set(self.template_parameters())
        transform_parameters = set(self.transform_parameters())
        for param in self.parameters():
            if (param not in template_parameters) and (param not in transform_parameters):
                yield param

    def _get_transform_code(self, sequence_name: str):
        # (m, d)
        return self.transform_dict(sequence_name)

    def _get_latent_code(self, shape_sequence_name: str, color_sequence_name: str):
        query = [shape_sequence_name] \
            if shape_sequence_name == color_sequence_name else \
            {
                'shape': [shape_sequence_name],
                'color': [color_sequence_name]
            }
        embeddings = self.latent_embedding(query)
        shape_outputs, color_outputs = embeddings['shape'], embeddings['color']

        return shape_outputs, color_outputs

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            shape_latent_code: torch.Tensor,
            color_latent_code: torch.Tensor,
            transform_codes: torch.Tensor,
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

        # Store points deformation and position
        points_deformation = {}
        points_position = {}

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
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
                transform_codes=transform_codes,
                enable_deformation=enable_deformation,

                fg_probability=target_mask,
                fg_thr=self._mask_thr
            )

            # store
            if self.training:
                # (m * n, i, p, 3)
                points_deformation[renderer_pass] = self._implicit_function[renderer_pass].latest_points_deformation
                points_position[renderer_pass] = self._implicit_function[renderer_pass].latest_points_position

            n_multiplexes = self.transform_dict.num_multiplexes

            if renderer_pass == "coarse":
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights.view(n_multiplexes, -1, *weights.shape[1:]).mean(dim=0)  # use mean weights TODO may use weights from the best transform
                if target_image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    rgb_gt = sample_images_at_mc_locs(
                        target_image.permute(0, 2, 3, 1).contiguous(),
                        ray_bundle_out.xys,
                    ).repeat(n_multiplexes, 1, 1)  # (m * n, i, 3)
                else:
                    rgb_gt = None

                # Sample the ground truth fg_probability
                if target_fg_probability is not None:
                    fg_gt = sample_images_at_mc_locs(
                        target_fg_probability.permute(0, 2, 3, 1).contiguous(),
                        ray_bundle_out.xys
                    ).repeat(n_multiplexes, 1, 1)  # (m * n, i, 1)
                else:
                    fg_gt = None

                # Sum of weights along a ray, (m * n, i)
                weights_coarse = weights.sum(dim=-1)

                # render depth
                depth_coarse = torch.sum(weights * ray_bundle_out.lengths.repeat(n_multiplexes, 1, 1),
                                         dim=-1, keepdim=True)  # (m * n, i, 1)

            elif renderer_pass == "fine":
                rgb_fine = rgb

                # Sum of weights along a ray, (m * n, i)
                weights_fine = weights.sum(dim=-1)

                # render depth
                depth_fine = torch.sum(weights * ray_bundle_out.lengths.repeat(n_multiplexes, 1, 1),
                                       dim=-1, keepdim=True)  # (m * n, i, 1)

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

        if self.training:
            for render_pass in ('coarse', 'fine'):
                out[f'points_deformation_{render_pass}'] = points_deformation[renderer_pass]
                out[f'points_position_{render_pass}'] = points_position[renderer_pass]

        return out

    def update_multiplex_score(self, sequence_name: str, scores: torch.Tensor):
        self.transform_dict.update_scores(sequence_name, scores)

    def forward(
            self,
            target_camera: CamerasBase,
            target_image: torch.Tensor,
            target_fg_probability: torch.Tensor,
            target_mask: torch.Tensor,
            shape_sequence_name: str,
            color_sequence_name: str = None,
            enable_deformation: bool = True
    ):
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
            shape_sequence_name: sequence name to provide shape embeddings
            color_sequence_name: sequence name to provide color embeddings
            enable_deformation:
        """
        batch_size = target_camera.R.shape[0]
        n_multiplexes = self.transform_dict.num_multiplexes

        # get rigid transform codes
        rigid_codes = self._get_transform_code(shape_sequence_name)

        # get latent code
        shape_latent_code, color_latent_code = self._get_latent_code(
            shape_sequence_name,
            shape_sequence_name if color_sequence_name is None else color_sequence_name,
        )

        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
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
                shape_latent_code=shape_latent_code,
                color_latent_code=color_latent_code,
                transform_codes=rigid_codes,
                enable_deformation=enable_deformation,
                chunk_idx=chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            outs = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }

            outs.update({
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(batch_size, *self._image_size, 1)  # n_multiplexes = 1 when self.training == false
                if chunk_outputs[0][k] is not None
                else None
                for k in ("depth_fine", "depth_coarse")
            })
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
        for out in outs:
            # Calc the error metrics.
            metrics = {}
            if target_image is not None:
                for render_pass in ("coarse", "fine"):
                    for metric_name, metric_fun in zip(
                            ("mse", "psnr"), (calc_mse, calc_psnr)
                    ):
                        metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                            out["rgb_" + render_pass][..., :3],
                            out["rgb_gt"][..., :3],
                            target_mask.permute(0, 2, 3, 1).contiguous() if not self.training else None
                        )

            if self.training:
                # Calc points deformation regularization
                for render_pass in ('coarse', 'fine'):
                    metrics[f'deformation_reg_{render_pass}'] = calc_l2_reg(out[f'points_deformation_{render_pass}'])
                    metrics[f'deformation_loss_{render_pass}'] = \
                        calc_deformation_loss(out[f'points_deformation_{render_pass}'],
                                              out[f'points_position_{render_pass}'],
                                              self._pairs_per_image,
                                              self._epsilon)

                # Calc weights reg
                fg_probability = out['fg_gt']
                for render_pass in ("coarse", "fine"):
                    metrics[f'weights_reg_{render_pass}'] = calc_weights_reg(out[f'weights_{render_pass}'],
                                                                             fg_probability.squeeze(-1))

            # append to list
            multiplex_metrics.append(metrics)

        # each element stores all outputs and metrics
        outs = outs[0] if len(outs) == 1 else outs
        multiplex_metrics = multiplex_metrics[0] if len(multiplex_metrics) == 1 else multiplex_metrics

        return outs, multiplex_metrics
