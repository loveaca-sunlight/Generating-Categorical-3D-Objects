import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.renderer.cameras import CamerasBase
from .utils import repeat_first_dim


def conv(dim_in: int, dim_out: int, kernel_size: int):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size, 2, padding=(kernel_size // 2)),
        nn.ReLU(True)
    )


def conv_t(dim_in: int, dim_out: int):
    return nn.Sequential(
        nn.ConvTranspose2d(dim_in, dim_out, 3, 2, padding=1, output_padding=1),
        nn.ReLU(True)
    )


class LLFFUnet(nn.Module):
    def __init__(self, dim_in: int):
        super(LLFFUnet, self).__init__()

        # layers
        self.conv1 = conv(dim_in, 64, 7)
        self.conv2 = conv(64, 128, 3)
        self.conv3 = conv(128, 256, 3)
        self.conv4 = conv(256, 512, 3)

        self.linear = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            conv(512, 512, 3)
        )
        self.fc = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(True)
        )

        self.conv_t4 = conv_t(512, 256)
        self.conv_t3 = conv_t(512, 128)
        self.conv_t2 = conv_t(256, 64)
        self.conv_t1 = conv_t(128, 128)

        self.source_latent_code = None
        self.source_camera = None

        self.latent_dim = 128

    def encode(self, source_image: torch.Tensor, source_camera: CamerasBase):
        x_64 = self.conv1(source_image)
        x_128 = self.conv2(x_64)
        x_256 = self.conv3(x_128)
        x_512 = self.conv4(x_256)

        n, _, h, w = x_512.shape
        globl = self.linear(x_512)
        globl = torch.mean(globl, dim=(2, 3), keepdim=True)
        globl = globl.repeat(1, 1, h, w)
        globl = torch.cat([globl, x_512], dim=1)
        globl = self.fc(globl)

        x2_256 = self.conv_t4(globl)
        x2_128 = self.conv_t3(torch.cat([x2_256, x_256], dim=1))
        x2_64 = self.conv_t2(torch.cat([x2_128, x_128], dim=1))
        local = self.conv_t1(torch.cat([x2_64, x_64], dim=1))

        self.source_latent_code = local
        self.source_camera = source_camera

    def index(self, uv: torch.Tensor):
        """
        Fetch features corresponding to image coordinates
        Adapted for multi-view
        :param uv: image coordinates, (b * n, i, p, 2)
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
            mode='nearest',
            padding_mode='zeros',
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

        # fetch features, (b * n, i, p, d), n is number of source views
        return self.index(points)

