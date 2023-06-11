import torch

from torch.utils.data import Dataset
from models.util import choose_views
from dataclasses import dataclass, fields
from pytorch3d.renderer.cameras import PerspectiveCameras


@dataclass
class LLFFFrameData:
    image: torch.Tensor
    camera: PerspectiveCameras
    bounds: torch.Tensor

    def to(self, *args, **kwargs):
        new_params = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (torch.Tensor, PerspectiveCameras)):
                new_params[f.name] = value.to(*args, **kwargs)
            else:
                new_params[f.name] = value
        return type(self)(**new_params)

    @classmethod
    def collate(cls, batch):
        images = torch.cat(
                [b.image for b in batch],
                dim=0
            )
        cameras = PerspectiveCameras(
                R=torch.cat([c.camera.R for c in batch], dim=0),
                T=torch.cat([c.camera.T for c in batch], dim=0),
                focal_length=torch.cat([c.camera.focal_length for c in batch], dim=0),
                principal_point=torch.cat([c.camera.principal_point for c in batch], dim=0),
            )
        bounds = torch.cat(
            [b.bounds for b in batch],
            dim=0
        )
        return cls(
            image=images,
            camera=cameras,
            bounds=bounds
        )


class LLFFDataset(Dataset):
    """
    A dataset that support llff-format dataset.
    """
    def __init__(self, data_path: str):
        """
        Initialize
        :param data_path: path of data file
        """
        # load to memory
        data = torch.load(data_path, map_location='cpu')
        self._images = data['images']
        self._cameras = data['cameras']
        self._bounds = data['bounds']
        assert self._images.shape[0] == self._cameras.R.shape[0]

        # show message
        print(f'LLFF data has been loaded from {data_path}.')

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        camera, image, bound = choose_views([idx], self._cameras, self._images, self._bounds)
        return LLFFFrameData(
            image=image,
            camera=camera,
            bounds=bound
        )
