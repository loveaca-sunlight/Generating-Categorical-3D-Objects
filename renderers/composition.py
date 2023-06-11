"""
A renderer that composite several implicit functions to render multiple objects
"""
import collections
from typing import Tuple, List

import torch
from pytorch3d.renderer import ImplicitRenderer
from pytorch3d.renderer.cameras import CamerasBase

from modules.encoder import ENCODERS
from nerf.raymarcher import EmissionAbsorptionNeRFRaymarcher
from nerf.raysampler import NeRFRaysampler
from render_functions import ComMlp
from .util import select_function_weights, select_encoder_weights


class CompositionRenderer(torch.nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int],
            n_pts_per_ray: int,
            n_rays_per_image: int,
            min_depth: float,
            max_depth: float,
            chunk_size_test: int,
            scene_function_config: dict,
            implicit_function_configs: List[dict],
            deformable_configs: List[dict],
            encoder_configs: List[dict]
    ):
        super(CompositionRenderer, self).__init__()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        # Initialize the coarse raysampler.
        raysampler = NeRFRaysampler(
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth,
            stratified=True,
            stratified_test=False,
            n_rays_per_image=n_rays_per_image,
            image_height=image_height,
            image_width=image_width,
        )

        # define renderer
        self._renderer = ImplicitRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        )

        # only define coarse functions
        self._implicit_function = ComMlp(scene_function_config, implicit_function_configs, deformable_configs)

        # define encoders
        encoders = [
            ENCODERS.build(config)
            for config in encoder_configs
        ]
        self._encoders = torch.nn.ModuleList(encoders)

        # save other parameters
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size

    def load_weights(self, scene_weights: dict, state_dicts: List[dict]):
        assert len(state_dicts) == len(self._encoders), \
            f'The number of state_dicts ({len(state_dicts)}) must be equal to the number of encoders ' \
            f'({len(self._encoders)}).'

        # select scene weights
        if scene_weights is not None:
            prefix = 'model._implicit_function.fine.'
            scene_weights = collections.OrderedDict(
                (k[len(prefix):], scene_weights[k])
                for k in filter(lambda x: x.startswith(prefix), scene_weights.keys())
            )

        # load weights for implicit functions
        function_state_dicts = [
            select_function_weights(state_dict, fine_only=True)
            for state_dict in state_dicts
        ]

        self._implicit_function.load_weights(scene_weights, function_state_dicts)

        # load weights for encoders
        for weights, encoder in zip(state_dicts, self._encoders):
            selected_weights = select_encoder_weights(weights)
            encoder.load_state_dict(selected_weights, strict=True)

    def _process_ray_chunk(
            self,
            target_camera: CamerasBase,
            object_codes: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
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
        # only coarse pass
        (features, _), _ = self._renderer(
            cameras=target_camera,
            volumetric_function=self._implicit_function,

            chunksize=self._chunk_size_test,
            chunk_idx=chunk_idx,
            density_noise_std=0.0,
            camera_hash=None,

            object_codes=object_codes,
            enable_deformation=enable_deformation,
            enable_specular=enable_specular
        )

        out = {
            'rgb_fine': features[..., : 3],
            'depth_fine': features[..., 3: 4]
        }

        return out

    def forward(
            self,
            target_camera: CamerasBase,
            object_images: List[List[torch.Tensor]],
            object_transforms: List[List[torch.Tensor]],
            enable_deformation: bool = True,
            enable_specular: bool = True,
    ) -> dict:
        """
        Render composition
        """
        batch_size = target_camera.R.shape[0]

        latent_codes = []
        for image_chunk, transform_chunk, encoder in zip(object_images, object_transforms, self._encoders):
            chunk_data = []
            for image, transform in zip(image_chunk, transform_chunk):
                shape_code, color_code = encoder(image)
                chunk_data.append((shape_code, color_code, transform))
            latent_codes.append(chunk_data)

        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer.raysampler.get_n_chunks(
                self._chunk_size_test,
                batch_size,
            )
        else:
            # must be val mode
            raise Exception('Model must be in evaluation mode.')

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                target_camera=target_camera,
                object_codes=latent_codes,
                enable_deformation=enable_deformation,
                enable_specular=enable_specular,
                chunk_idx=chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        # For a full render pass concatenate the output chunks,
        # and reshape to image size.
        out = {
            k: torch.cat(
                [ch_o[k] for ch_o in chunk_outputs],
                dim=1,
            ).view(-1, *self._image_size, 3)
            if chunk_outputs[0][k] is not None
            else None
            for k in ('rgb_fine',)
        }
        out.update({
            k: torch.cat(
                [ch_o[k] for ch_o in chunk_outputs],
                dim=1,
            ).view(batch_size, *self._image_size, 1)
            if chunk_outputs[0][k] is not None
            else None
            for k in ("depth_fine",)
        })

        return out
