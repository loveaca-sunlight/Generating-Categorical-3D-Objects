from typing import List, Optional

import torch

from .sparse_conv import SparseConv2d, SparseReLU, SparseMaxPool2d, TensorWithMask, SparseGroupNorm


_NUM_GROUP = 16


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> SparseConv2d:
    """3x3 convolution with padding"""
    return SparseConv2d(in_planes, out_planes, 3, stride)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> SparseConv2d:
    """1x1 convolution"""
    return SparseConv2d(in_planes, out_planes, 1, stride)


class _BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        norm: bool = False
    ) -> None:
        super(_BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = SparseGroupNorm(num_groups=_NUM_GROUP, num_channels=planes) if norm else torch.nn.Identity()
        self.relu = SparseReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = SparseGroupNorm(num_groups=_NUM_GROUP, num_channels=planes) if norm else torch.nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: TensorWithMask) -> TensorWithMask:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(torch.nn.Module):

    def __init__(
        self,
        layers: List[int],
        dim_in: int,
        norm: bool
    ) -> None:
        super(_ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self.norm = norm

        self.conv1 = SparseConv2d(dim_in, self.inplanes, kernel_size=7, stride=2)
        self.norm1 = SparseGroupNorm(num_groups=_NUM_GROUP, num_channels=self.inplanes) if norm else torch.nn.Identity()
        self.relu = SparseReLU(inplace=True)

        self.maxpool = SparseMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> torch.nn.Sequential:
        block = _BasicBlock

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                SparseGroupNorm(num_groups=_NUM_GROUP,
                                num_channels=planes * block.expansion) if self.norm else torch.nn.Identity()
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm=self.norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=self.norm))

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x: TensorWithMask) -> TensorWithMask:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: TensorWithMask) -> TensorWithMask:
        return self._forward_impl(x)


def sparse_resnet18(dim_in: int, norm: bool):
    return _ResNet(
        layers=[2, 2, 2, 2],
        dim_in=dim_in,
        norm=norm
    )


def sparse_resnet34(dim_in: int, norm: bool):
    return _ResNet(
        layers=[3, 4, 6, 3],
        dim_in=dim_in,
        norm=norm
    )
