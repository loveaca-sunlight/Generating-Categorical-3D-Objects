import torch.nn as nn
import torch
import torch.nn.functional as F


class TensorWithMask:
    def __init__(self, data: torch.Tensor, mask: torch.Tensor):
        self.data = data  # (n, c, h, w)
        self.mask = mask  # (n, 1, h, w)

    def __add__(self, other):
        x1, m1 = self.data, self.mask
        x2, m2 = other.data, other.mask

        return TensorWithMask(
            ((x1 * m1) + (x2 * m2)) / (m1 + m2 + 1.0e-8),
            torch.maximum(m1, m2)
        )

    def __repr__(self):
        return f'data: {self.data}, mask: {self.mask}.'


class SparseConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False
        )

        self.bias = nn.Parameter(
            torch.zeros(out_channels),
            requires_grad=True
        )

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False
        )

        kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False
        )

        self.max_pool = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, inputs: TensorWithMask):
        x, mask = inputs.data, inputs.mask

        x = x * mask
        x = self.conv(x)
        normalizer = 1.0 / (self.sparsity(mask) + 1e-8)
        x = x * normalizer + self.bias.view(1, -1, 1, 1)

        mask = self.max_pool(mask)

        return TensorWithMask(
            x, mask
        )


class SparseReLU(nn.Module):
    def __init__(self, inplace: bool):
        super(SparseReLU, self).__init__()

        self.inplace = inplace

    def forward(self, inputs):
        return TensorWithMask(
            F.relu(inputs.data, inplace=self.inplace),
            inputs.mask
        )


class SparseMaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(SparseMaxPool2d, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, inputs):
        return TensorWithMask(
            self.pool(inputs.data),
            self.pool(inputs.mask)
        )


class SparseGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super(SparseGroupNorm, self).__init__()

        self.norm = nn.GroupNorm(num_groups, num_channels, eps, affine)

    def forward(self, inputs):
        return TensorWithMask(
            self.norm(inputs.data),
            inputs.mask
        )
