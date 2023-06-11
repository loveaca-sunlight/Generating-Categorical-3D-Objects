import warnings
from typing import List, Optional

import torch.nn as nn
from torch import Tensor
import torch


_NUM_GROUPS = 16


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias=True) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm: bool = False
    ) -> None:
        super(_BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = nn.GroupNorm(_NUM_GROUPS, planes) if norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = nn.GroupNorm(_NUM_GROUPS, planes) if norm else nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class _ResNet(nn.Module):

    def __init__(
        self,
        layers: List[int],
        dim_in: int,
        norm: bool
    ) -> None:
        super(_ResNet, self).__init__()

        if norm:
            warnings.warn('The norm is not recommended.')

        self.inplanes = 64
        self.dilation = 1
        self._norm = norm

        self.conv1 = nn.Conv2d(dim_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.norm1 = nn.GroupNorm(_NUM_GROUPS, self.inplanes) if self._norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        block = _BasicBlock

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, bias=False),
                nn.GroupNorm(_NUM_GROUPS, planes * block.expansion) if self._norm else nn.Identity()
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm=self._norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=self._norm))

        return nn.Sequential(*layers)

    def encode_with_intermediate(self, input: Tensor):
        results = [input]
        x = self.conv1(input)
        x = self.norm1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        for i in range(4):
            func = getattr(self, 'layer{:d}'.format(i + 1))
            x = func(x)
            results.append(x)
        return results[1:]

    def get_content_feat(self, input: Tensor):
        return self._forward_impl(input)


    def get_style_feat(self, input):
        style_feats = self.encode_with_intermediate(input)
        out_mean = []
        out_std = []
        out_mean_std = []
        for style_feat in style_feats:
            style_feat_mean, style_feat_std, style_feat_mean_std = self.calc_feat_mean_std(style_feat)
            out_mean.append(style_feat_mean)
            out_std.append(style_feat_std)
            out_mean_std.append(style_feat_mean_std)
        return style_feats, torch.cat(out_mean_std, dim=-1)
    
    def calc_feat_mean_std(self, input, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = input.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = input.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std, torch.cat([feat_mean, feat_std], dim = 1)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)# 64 64 64
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)# 64 32 32 

        x = self.layer1(x)
        x = self.layer2(x)#128 16 16
        x = self.layer3(x)#256 8 8 
        x = self.layer4(x)#512 4 4

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18_wo_bn(dim_in: int, norm: bool):
    return _ResNet(
        layers=[2, 2, 2, 2],
        dim_in=dim_in,
        norm=norm
    )


def resnet34_wo_bn(dim_in: int, norm: bool):
    return _ResNet(
        layers=[3, 4, 6, 3],
        dim_in=dim_in,
        norm=norm
    )
