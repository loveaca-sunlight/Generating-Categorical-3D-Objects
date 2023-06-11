import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.cameras import CamerasBase
from .utils import repeat_first_dim
from ..encoder import resnet18_wo_bn, resnet34_wo_bn


class PixelEncoder(nn.Module):
    def __init__(self,
                 n_layers: int,
                 latent_dim: int = 512,
                 ):
        """
        Initialize
        :param n_layers: layers of resnet encoder
        :param latent_dim: dimensions of latent code
        """
        super(PixelEncoder, self).__init__()

        self.latent_dim = latent_dim

        assert latent_dim == 512

        backbone = {
            18: resnet18_wo_bn,
            34: resnet34_wo_bn
        }[n_layers](dim_in=3, norm=False)

        self.block1 = nn.Sequential(backbone.conv1, backbone.relu)
        self.block2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.block3 = backbone.layer2
        self.block4 = backbone.layer3
        # self.block5 = backbone.layer4

        # num_features = np.array([64, 64, 128, 256, 512])
        # if n_layers == 50:
        #     num_features[1:] *= 4

        # self.project = nn.Conv2d(int(np.sum(num_features)), latent_dim, kernel_size=(1, 1))
        # _xavier_init(self.project)

        self.source_latent_code = None
        self.source_camera = None

    def encode(self, source_image: torch.Tensor, source_camera: CamerasBase):
        """
        Encode image to features, call this first
        :param source_image: input image, (n, c, h, w)
        :param source_camera: camera in source view
        :return:
        """
        n, _, h, w = source_image.shape
        features = []

        # 1
        out = self.block1(source_image)
        features.append(out)
        # 2
        out = self.block2(out)
        features.append(out)
        # 3
        out = self.block3(out)
        features.append(out)
        # 4
        out = self.block4(out)
        features.append(out)
        # # 5
        # out = self.block5(out)
        # features.append(out)

        features = [
            F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True)
            for feature in features
        ]
        out = torch.cat(features, dim=1)  # (n, c, h, w), n is the number of source views

        # out = self.project(out)

        self.source_latent_code = out
        self.source_camera = source_camera

    def index(self, uv: torch.Tensor):
        """
        Fetch features corresponding to image coordinates
        Adapted for multi-view
        :param uv: image coordinates, (b, i, p, 2)
        :return:
        """
        # check
        assert (self.source_latent_code is not None) and (self.source_camera is not None), \
            f'Please encode the image first.'

        # b - batch size, n - number of source views
        n, c, h, w = self.source_latent_code.shape
        b = uv.shape[0] // n

        # repeat dims of latent code to (b * n, i, p, d)
        latent_code = repeat_first_dim(self.source_latent_code, dim=0, n_repeats=b).view(b * n, c, h, w)

        # (b * n, d, i, p)
        samples = F.grid_sample(
            latent_code,
            uv,
            align_corners=True,
            mode='bilinear',
            padding_mode='border',
        )
        # (b * n, i, p, d)
        return samples.permute(0, 2, 3, 1).contiguous()

    def forward(self, points: torch.Tensor):
        """
        Fetch features corresponding to source cameras.
        Note, encode first
        :param points: points in world coordinate, (n, i, p, 3)
        :return:
        """
        b_n, i, p, _ = points.shape

        # # back-project points to ndc coordinates and invert direction. (b * n, i, p, 2)
        # uv = -1.0 * self.source_camera.transform_points_ndc(points.view(b_n, i * p, 3),
        #                                                     eps=1.0e-8)[..., :2].view(b_n, i, p, 2)

        # fetch features, (b * n, i, p, d), n is number of source views
        return self.index(points)
