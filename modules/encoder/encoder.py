import typing
import warnings

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ..resnet_backbone import resnet18_backbone, resnet34_backbone, resnet50_backbone
from nerf.implicit_function import _xavier_init
from torchvision.models import vgg11, vgg13, vgg16
from mmcv import Registry
from .sparse_resnet import sparse_resnet18, sparse_resnet34, TensorWithMask
from .resnet_wo_bn import resnet18_wo_bn, resnet34_wo_bn
from .ViT.vit import Block


ENCODERS = Registry('encoders')

_LEAKY_RELU_ALPHA = 1 / 5.5


def _remove_batch_norm(module: nn.Module):
    """
    Replace batch norm as identity
    :param module:
    :return:
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = nn.Identity()
    for name, child in module.named_children():
        module_output.add_module(name, _remove_batch_norm(child))
    del module
    return module_output


def _prepare_encoder(name: str, pretrained: bool, dim_in: int = 3):
    model, n_layers = name.split('_')
    n_layers = int(n_layers)

    if model == 'resnet':
        if n_layers == 18:
            backbone = resnet18_backbone(pretrained)
        elif n_layers == 34:
            backbone = resnet34_backbone(pretrained)
        elif n_layers == 50:
            backbone = resnet50_backbone(pretrained)
        else:
            raise ValueError(f'Unknown layers: {n_layers}.')

        # remove bn
        backbone = _remove_batch_norm(backbone)

        # input channels
        if dim_in != 3:
            backbone.conv1 = nn.Conv2d(dim_in, 64, kernel_size=7, stride=2, padding=3, bias=False)

        feature_dim = {
            18: 512,
            34: 512,
            50: 1024
        }[n_layers]

        encoder = nn.Sequential(
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            nn.Sequential(backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
    elif model == 'vgg':
        vgg = {
            11: vgg11,
            13: vgg13,
            16: vgg16
        }[n_layers](pretrained=pretrained)

        feature_dim = 512

        encoder = vgg.features
    else:
        raise f'Unknown model: {model}.'

    return encoder, feature_dim


def _normalize(tensor: torch.Tensor, max_norm: torch.Tensor):
    norm = torch.norm(tensor, dim=-1, keepdim=True)  # (n, 1)
    norm = torch.maximum(norm, max_norm.expand_as(norm))
    return tensor / norm


class EncoderBase(nn.Module):
    """
    Base class for encoders
    """
    def __init__(self, encoder_name: str, pretrained: bool = False):
        super(EncoderBase, self).__init__()

        self.encoder, self._feature_dim = _prepare_encoder(encoder_name, pretrained)
        print(f'Using encoder: {encoder_name}.')


@ENCODERS.register_module(name='code_encoder_v1')
class LatentCodeEncoder(EncoderBase):
    def __init__(self, encoder_name: str, shape_code_dim: int, color_code_dim: int,
                 clamp_norm: bool = True, reduction: bool = True, pretrained: bool = True):
        super().__init__(encoder_name, pretrained)

        self.shape_project = nn.Conv2d(self._feature_dim, shape_code_dim, 1)
        _xavier_init(self.shape_project)
        self.color_project = nn.Conv2d(self._feature_dim, color_code_dim, 1)
        _xavier_init(self.color_project)

        self._clamp_norm = clamp_norm
        if self._clamp_norm:
            self.register_buffer('_max_shape_norm', torch.tensor([math.sqrt(shape_code_dim)], dtype=torch.float))
            self.register_buffer('_max_color_norm', torch.tensor([math.sqrt(color_code_dim)], dtype=torch.float))
            print(f'Set the max norm of shape and color to {self._max_shape_norm.item()} '
                  f'and {self._max_color_norm.item()}, respectively.')

        self._reduction = reduction

        # not the currently used component
        warnings.warn(f'{type(self).__name__} is not a currently used components.')

    def forward(self, images):
        """
        Compute latent code of input image x
        :param images:
        :return:
        """
        feature = self.encoder(images)

        # （n, c)
        shape_latent_code = self.shape_project(feature).mean(-1).mean(-1)
        color_latent_code = self.color_project(feature).mean(-1).mean(-1)

        # check norm
        if self._clamp_norm:
            shape_latent_code = _normalize(shape_latent_code, self._max_shape_norm)
            color_latent_code = _normalize(color_latent_code, self._max_color_norm)

        # reduction, (1, c)
        if self._reduction:
            shape_latent_code = torch.mean(shape_latent_code, dim=0, keepdim=True)
            color_latent_code = torch.mean(color_latent_code, dim=0, keepdim=True)

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='code_encoder_v2')
class LatentCodeEncoderV2(EncoderBase):
    def __init__(self, encoder_name: str, shape_code_dim: int, color_code_dim: int,
                 clamp_norm: bool = True, reduction: bool = True, pretrained: bool = True):
        super().__init__(encoder_name, pretrained)

        self.shape_project = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self._feature_dim, shape_code_dim)
        )
        self.color_project = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self._feature_dim, color_code_dim)
        )

        self._clamp_norm = clamp_norm
        if self._clamp_norm:
            self.register_buffer('_max_shape_norm', torch.tensor([math.sqrt(shape_code_dim)], dtype=torch.float))
            self.register_buffer('_max_color_norm', torch.tensor([math.sqrt(color_code_dim)], dtype=torch.float))
            print(f'Set the max norm of shape and color to {self._max_shape_norm.item():.04f} '
                  f'and {self._max_color_norm.item():.04f}, respectively.')

        self._reduction = reduction

        self.init_layers()

    def init_layers(self):
        for linear in (self.shape_project, self.color_project):
            for module in linear.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(0.0, 0.02)
                    module.bias.data.zero_()

    def forward(self, images):
        """
        Compute latent code of input image x
        :param images:
        :return:
        """
        feature = self.encoder(images).mean(dim=(2, 3))  # (n, c)

        # （n, c)
        shape_latent_code = self.shape_project(feature)
        color_latent_code = self.color_project(feature)

        # check norm
        if self._clamp_norm:
            shape_latent_code = _normalize(shape_latent_code, self._max_shape_norm)
            color_latent_code = _normalize(color_latent_code, self._max_color_norm)

        # reduction, (1, c)
        if self._reduction:
            shape_latent_code = torch.mean(shape_latent_code, dim=0, keepdim=True)
            color_latent_code = torch.mean(color_latent_code, dim=0, keepdim=True)

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='code_encoder_v3')
class LatentCodeEncoderV3(EncoderBase):
    """
    Encoder with Attention
    """
    def __init__(self, encoder_name: str, shape_code_dim: int, color_code_dim: int,
                 clamp_norm: bool = True, reduction: bool = None, pretrained: bool = True):
        """
        Initialize
        :param encoder_name:
        :param shape_code_dim:
        :param color_code_dim:
        :param clamp_norm:
        :param reduction: must be true
        :param pretrained:
        """
        super().__init__(encoder_name, pretrained)

        assert reduction is None, 'reduction must be unset in this version.'

        # shape
        self.shape_branch = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.shape_attention = nn.Sequential(
            nn.Linear(self._feature_dim, shape_code_dim),
            nn.Tanh()  # scale the attention value to (-1, 1)
        )

        # color
        self.color_branch = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)
        self.color_attention = nn.Sequential(
            nn.Linear(self._feature_dim, color_code_dim),
            nn.Tanh()
        )

        self._clamp_norm = clamp_norm
        if self._clamp_norm:
            self.register_buffer('_max_shape_norm', torch.tensor([math.sqrt(shape_code_dim)], dtype=torch.float))
            self.register_buffer('_max_color_norm', torch.tensor([math.sqrt(color_code_dim)], dtype=torch.float))
            print(f'Set the max norm of shape and color to {self._max_shape_norm.item():.04f} '
                  f'and {self._max_color_norm.item():.04f}, respectively.')

        self.init_layers()

        # not the currently used component
        warnings.warn(f'{type(self).__name__} is not a currently used components.')

    def init_layers(self):
        for linear in (self.shape_branch, self.shape_project, self.shape_attention, self.color_branch,
                       self.color_project, self.color_attention):
            for module in linear.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(0.0, 0.02)
                    module.bias.data.zero_()

    def forward(self, images):
        """
        Compute latent code of input image x
        :param images:
        :return:
        """
        feature = self.encoder(images).mean(dim=(2, 3))  # (n, c)

        # （n, c)
        shape_features = self.shape_branch(feature)
        color_features = self.color_branch(feature)

        # original code, (n, c)
        shape_latent_code = self.shape_project(shape_features)
        color_latent_code = self.color_project(color_features)

        # check code norm
        if self._clamp_norm:
            shape_latent_code = _normalize(shape_latent_code, self._max_shape_norm)
            color_latent_code = _normalize(color_latent_code, self._max_color_norm)

        # attention value, (n, c)
        shape_attention = self.shape_attention(shape_features)
        color_attention = self.color_attention(color_features)
        # softmax, (n, c)
        shape_attention = F.softmax(shape_attention, dim=0)
        color_attention = F.softmax(color_attention, dim=0)

        # reduction with attention, (1, c)
        shape_latent_code = torch.sum(shape_latent_code * shape_attention, dim=0, keepdim=True)  # (1, c)
        color_latent_code = torch.sum(color_latent_code * color_attention, dim=0, keepdim=True)  # (1, c)

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='simple_encoder')
class SimpleEncoder(nn.Module):
    """
    Simple encoder that encode input images to color and shape latent code.
    Using output scale to limit the norm of output latent code.
    """
    def __init__(self, n_layers: int, shape_code_dim: int, color_code_dim: int):
        super(SimpleEncoder, self).__init__()

        self.encoder = {
            18: resnet18_wo_bn,
            34: resnet34_wo_bn
        }[n_layers](dim_in=4, norm=False)

        self._feature_dim = 512

        # intermediate layers
        self.intermediate_shape = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        self.intermediate_color = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        # projection layers
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.intermediate_shape, self.intermediate_color):
            for linear in module.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.kaiming_uniform_(linear.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in',
                                             nonlinearity='leaky_relu')
        for linear in (self.shape_project, self.color_project):
            nn.init.normal_(linear.weight.data, mean=0.0, std=0.01)

    def forward(self, images):
        """
        Compute latent code of input image x
        :param images:
        :return:
        """
        feature = self.encoder(images).mean(dim=(2, 3))  # (n, c)

        # （n, c)
        shape_latent_code = self.shape_project(self.intermediate_shape(feature))
        color_latent_code = self.color_project(self.intermediate_color(feature))

        # reduction, (1, c)
        shape_latent_code = torch.mean(shape_latent_code, dim=0, keepdim=True)
        color_latent_code = torch.mean(color_latent_code, dim=0, keepdim=True)

        return shape_latent_code, color_latent_code


class CameraEncoder(EncoderBase):
    def __init__(self, encoder_name: str, pretrained: bool = False):
        super(CameraEncoder, self).__init__(encoder_name, pretrained)

        self.k_project = nn.Linear(2 + 2, self._feature_dim)
        self.ext_project = nn.Sequential(
            nn.Linear(self._feature_dim * 2, self._feature_dim),
            nn.Linear(self._feature_dim, 6)
        )

    def forward(self, images: torch.Tensor, focal_length: torch.Tensor, principal_point: torch.Tensor):
        """
        Compute camera extrinsics
        :param images: (n, c, h, w)
        :param focal_length: (n, 2)
        :param principal_point: (n, 2)
        :return:
        """
        image_feature = self.encoder(images).mean(-1).mean(-1)  # (n, c)

        cam_k = torch.cat([focal_length * 0.1, principal_point], dim=1)  # (n, 4)
        cam_feature = self.k_project(cam_k)  # (n, c)

        extrinsics = self.ext_project(
            torch.cat([image_feature, cam_feature], dim=1)
        )  # (n, 6)

        return extrinsics


@ENCODERS.register_module(name='new_encoder')
class NewEncoder(EncoderBase):
    """
    Simple encoder that encode input images to color and shape latent code.
    For input images with size (128, 128).
    """
    def __init__(self, encoder_net: str, shape_code_dim: int, color_code_dim: int):
        super(NewEncoder, self).__init__(encoder_net, pretrained=False)

        self.shape_output = nn.Sequential(
            nn.Linear(self._feature_dim, shape_code_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(shape_code_dim, shape_code_dim),
            nn.ReLU(inplace=True)  # try to use non-negative code
        )
        self.color_output = nn.Sequential(
            nn.Linear(self._feature_dim, color_code_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(color_code_dim, color_code_dim),
            nn.ReLU(inplace=True)  # try to use non-negative code
        )

        self.shape_coef = nn.Parameter(torch.zeros(1, 1), requires_grad=True)
        self.color_coef = nn.Parameter(torch.zeros(1, 1), requires_grad=True)

        self.init_layers()

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, images):
        """
        Compute latent code of input image x
        :param images:
        :return:
        """
        feature = self.encoder(images).mean(dim=(2, 3))  # (n, c)

        # (n, c)
        shape_feature = self.shape_output(feature)
        color_feature = self.color_output(feature)

        # reduction and fusion
        shape_mean = torch.mean(shape_feature, dim=0, keepdim=True)  # (1, c)
        shape_max = torch.max(shape_feature, dim=0, keepdim=True)[0]  # (1, c)
        shape_coef = torch.sigmoid(self.shape_coef)
        shape_latent_code = shape_coef * shape_mean + (1.0 - shape_coef) * shape_max

        color_mean = torch.mean(color_feature, dim=0, keepdim=True)
        color_max = torch.max(color_feature, dim=0, keepdim=True)[0]
        color_coef = torch.sigmoid(self.color_coef)
        color_latent_code = color_coef * color_mean + (1.0 - color_coef) * color_max

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='sparse_encoder')
class SparseEncoder(nn.Module):
    def __init__(self, n_layers: int, shape_code_dim: int, color_code_dim: int, attention: bool, norm: bool):
        super(SparseEncoder, self).__init__()

        self.encoder = {
            18: sparse_resnet18,
            34: sparse_resnet34
        }[n_layers](dim_in=4, norm=norm)

        self._feature_dim = 512
        self._attention = attention

        # intermediate layers
        self.intermediate_shape = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.GroupNorm(32, self._feature_dim) if norm else nn.Identity(),
            nn.ELU(inplace=True)
        )
        self.intermediate_color = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.GroupNorm(32, self._feature_dim) if norm else nn.Identity(),
            nn.ELU(inplace=True)
        )
        # projection layers
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)
        # attention layers
        if self._attention:
            self.shape_attn = nn.Linear(self._feature_dim, shape_code_dim)
            self.color_attn = nn.Linear(self._feature_dim, color_code_dim)
        # init
        self.init_linear()

    def init_linear(self):
        for linear in (self.shape_project, self.color_project):
            linear.weight.data.normal_(0.0, 0.01)
            linear.bias.data.zero_()

    def forward(self, inputs: TensorWithMask) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        enc_out = self.encoder(inputs)
        enc_data, enc_mask = enc_out.data, enc_out.mask  # (n, c, h, w), (n, 1, h, w)

        # encoder features
        enc_feats = (enc_data * enc_mask).sum(dim=(2, 3)) / (enc_mask.sum(dim=(2, 3)) + 1.0e-6)  # (n, c)

        # intermediate features
        shape_feats = self.intermediate_shape(enc_feats)
        color_feats = self.intermediate_color(enc_feats)

        # codes
        shape_codes = self.shape_project(shape_feats)  # (n, cs)
        color_codes = self.color_project(color_feats)  # (n, cc)

        # attention
        if self._attention:
            n = shape_codes.shape[0]
            if n > 1:
                shape_attn = self.shape_attn(shape_feats)
                color_attn = self.color_attn(color_feats)
                # softmax
                shape_attn = torch.softmax(shape_attn, dim=0)
                color_attn = torch.softmax(color_attn, dim=0)
                # sum
                shape_codes = (shape_codes * shape_attn).sum(dim=0, keepdim=True)
                color_codes = (color_codes * color_attn).sum(dim=0, keepdim=True)
        else:
            shape_codes = shape_codes.mean(dim=0, keepdim=True)
            color_codes = color_codes.mean(dim=0, keepdim=True)

        # (1, cs), (1, cc)
        return shape_codes, color_codes


@ENCODERS.register_module(name='attn_encoder')
class AttentionEncoder(nn.Module):
    def __init__(self, n_layers: int, shape_code_dim: int, color_code_dim: int):
        super(AttentionEncoder, self).__init__()

        self.encoder = {
            18: resnet18_wo_bn,
            34: resnet34_wo_bn
        }[n_layers](dim_in=4, norm=False)

        self._feature_dim = 512

        # intermediate layers
        self.intermediate_shape = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        self.intermediate_color = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        # projection layers
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)
        # attention layers
        self.shape_attention = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_attention = nn.Linear(self._feature_dim, color_code_dim)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.intermediate_shape, self.intermediate_color):
            for linear in module.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.kaiming_uniform_(linear.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in',
                                             nonlinearity='leaky_relu')
        for linear in (self.shape_project, self.color_project, self.shape_attention, self.color_attention):
            nn.init.normal_(linear.weight.data, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor):
        enc_feat = self.encoder(x).mean(dim=(2, 3))  # (n, c)

        shape_feat = self.intermediate_shape(enc_feat)  # (n, c)
        color_feat = self.intermediate_color(enc_feat)

        shape_code = self.shape_project(shape_feat)  # (n, d)
        color_code = self.color_project(color_feat)

        n = x.shape[0]
        if n > 1:  # compute attention only when n > 1
            shape_attention = self.shape_attention(shape_feat)
            color_attention = self.color_attention(color_feat)

            shape_attention = torch.softmax(shape_attention, dim=0)
            color_attention = torch.softmax(color_attention, dim=0)

            shape_latent_code = (shape_code * shape_attention).sum(dim=0, keepdim=True)  # (1, d)
            color_latent_code = (color_code * color_attention).sum(dim=0, keepdim=True)
        else:
            shape_latent_code = shape_code.mean(dim=0, keepdim=True)
            color_latent_code = color_code.mean(dim=0, keepdim=True)

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='conf_encoder')
class ConfidenceEncoder(nn.Module):
    def __init__(self, n_layers: int, shape_code_dim: int, color_code_dim: int):
        super(ConfidenceEncoder, self).__init__()

        self.encoder = {
            18: resnet18_wo_bn,
            34: resnet34_wo_bn
        }[n_layers](dim_in=4, norm=False)

        self._feature_dim = 512

        # intermediate layers
        self.intermediate_shape = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        self.intermediate_color = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        # projection layers
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)
        # attention layers
        self.shape_confidence = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_confidence = nn.Linear(self._feature_dim, color_code_dim)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.intermediate_shape, self.intermediate_color):
            for linear in module.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.kaiming_uniform_(linear.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in',
                                             nonlinearity='leaky_relu')
        for linear in (self.shape_project, self.color_project, self.shape_confidence, self.color_confidence):
            nn.init.normal_(linear.weight.data, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, return_conf: bool = False):
        enc_feat = self.encoder(x).mean(dim=(2, 3))  # (n, c)
        # features
        shape_feat = self.intermediate_shape(enc_feat)  # (n, c)
        color_feat = self.intermediate_color(enc_feat)
        # original codes
        shape_code = self.shape_project(shape_feat)  # (n, d)
        color_code = self.color_project(color_feat)
        # confidence, (n, d)
        shape_confidence = torch.sigmoid(self.shape_confidence(shape_feat))
        color_confidence = torch.sigmoid(self.color_confidence(color_feat))
        # softmax, (n, d)
        shape_softmax = torch.softmax(shape_confidence, dim=0)
        color_softmax = torch.softmax(color_confidence, dim=0)
        # fuse, (1, d)
        shape_latent_code = (shape_code * shape_softmax).sum(dim=0, keepdim=True)
        color_latent_code = (color_code * color_softmax).sum(dim=0, keepdim=True)

        if return_conf:
            return shape_latent_code, color_latent_code, shape_confidence, color_confidence
        else:
            return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='token_encoder')
class TokenEncoder(nn.Module):
    def __init__(
            self,
            n_layers: int,
            shape_code_dim: int,
            color_code_dim: int,
            num_heads: int,
            attention_dropout_rate: float,
            mlp_dim: int,
            dropout_rate: float,
            ffn_norm: bool = True
    ):
        super(TokenEncoder, self).__init__()

        # encoder
        self.encoder = {
            18: resnet18_wo_bn,
            34: resnet34_wo_bn
        }[n_layers](dim_in=4, norm=False)

        self._feature_dim = 512

        # vit relevant
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._feature_dim))
        self.vit_block = Block(self._feature_dim, num_heads, attention_dropout_rate, mlp_dim, dropout_rate, ffn_norm)

        # projection layers
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        for linear in (self.shape_project, self.color_project):
            nn.init.normal_(linear.weight.data, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor):
        enc_feat = self.encoder(x).mean(dim=(2, 3)).unsqueeze(0)  # (1, n, c)

        # vit
        tokens = torch.cat([self.cls_token, enc_feat], dim=1)  # (1, 1 + n, c)
        vit_feat = self.vit_block(tokens)[0][:, 0, :]

        # project
        shape_latent_code = self.shape_project(vit_feat)
        color_latent_code = self.color_project(vit_feat)

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='mean_encoder')
class MeanEncoder(nn.Module):
    def __init__(self, n_layers: int, shape_code_dim: int, color_code_dim: int, dim_in: int = 4):
        super(MeanEncoder, self).__init__()

        self.encoder = {
            18: resnet18_wo_bn,
            34: resnet34_wo_bn
        }[n_layers](dim_in=dim_in, norm=False)

        self._feature_dim = 512

        # intermediate layers
        self.intermediate_shape = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        self.intermediate_color = nn.Sequential(
            nn.Linear(self._feature_dim, self._feature_dim),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True)
        )
        # projection layers
        self.shape_project = nn.Linear(self._feature_dim, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim, color_code_dim)

        # reset parameters
        self.reset_parameters()
    
    def cal_style_loss(self, content, style, style_ori, n_rays):
        batch = content.shape[0]
        content = content.transpose(2, 1).reshape(batch, 4, n_rays, n_rays)
        style = style.transpose(2, 1).reshape(batch, 4, n_rays, n_rays)
        content_fea = self.encoder.get_content_feat(content)
        style_fea = self.encoder.get_content_feat(style)
        content_loss = F.mse_loss(content_fea, style_fea)

        output_style_feats, output_style_feat_mean_std = self.encoder.get_style_feat(style)
        style_feats, style_feat_mean_std = self.encoder.get_style_feat(style_ori.expand(batch, 4, 128, 128))
        style_loss = F.mse_loss(output_style_feat_mean_std, style_feat_mean_std)
        #style_loss = F.mse_loss(output_style_feats, style_feats)

        return content_loss, style_loss

    def reset_parameters(self):
        for module in (self.intermediate_shape, self.intermediate_color):
            for linear in module.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.kaiming_uniform_(linear.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in',
                                             nonlinearity='leaky_relu')
        for linear in (self.shape_project, self.color_project):
            nn.init.normal_(linear.weight.data, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor):
        tmp_feat = self.encoder(x)
        enc_feat = tmp_feat.mean(dim=(2, 3))  # (n, c)
        # features
        shape_feat = self.intermediate_shape(enc_feat)  # (n, c)
        color_feat = self.intermediate_color(enc_feat)
        # original codes
        shape_code = self.shape_project(shape_feat)  # (n, d)
        color_code = self.color_project(color_feat)
        # fuse, (1, d)
        shape_latent_code = shape_code.mean(dim=0, keepdim=True)
        color_latent_code = color_code.mean(dim=0, keepdim=True)

        return shape_latent_code, color_latent_code#, tmp_feat

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True)  # relu5-4
)

vgg4 = nn.Sequential(
    nn.Conv2d(4, 4, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(4, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True)  # relu5-4
)

fc_encoder = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024)
)

from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

@ENCODERS.register_module(name='vgg_encoder')
class VGGEncoder(nn.Module):
    def __init__(self, shape_code_dim: int, color_code_dim: int, dim_in: int = 4):
        super(VGGEncoder, self).__init__()

        enc_layers = list(vgg4.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        self._feature_dim = 512

        # intermediate layers
        self.intermediate_shape = nn.Sequential(
            nn.Linear(self._feature_dim*2, self._feature_dim*2),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True),
        )
        self.intermediate_color = nn.Sequential(
            nn.Linear(self._feature_dim*2, self._feature_dim*2),
            nn.LeakyReLU(_LEAKY_RELU_ALPHA, True),
        )
        # projection layers
        self.shape_project = nn.Linear(self._feature_dim*2, shape_code_dim)
        self.color_project = nn.Linear(self._feature_dim*2, color_code_dim)

        # reset parameters
        self.reset_parameters()
    
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def get_content_feat(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

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
    
    def cal_style_loss(self, content, style, style_ori, n_rays):
        batch = content.shape[0]
        content = content.transpose(2, 1).reshape(batch, 4, n_rays, n_rays)
        style = style.transpose(2, 1).reshape(batch, 4, n_rays, n_rays)
        content_fea = self.get_content_feat(content)
        style_fea = self.get_content_feat(style)
        content_loss = F.mse_loss(content_fea, style_fea)

        output_style_feats, output_style_feat_mean_std = self.get_style_feat(style)
        style_feats, style_feat_mean_std = self.get_style_feat(style_ori.expand(batch, 4, 128, 128))
        style_loss = F.mse_loss(output_style_feat_mean_std, style_feat_mean_std)
        #style_loss = F.mse_loss(output_style_feats, style_feats)

        return content_loss, style_loss
    
    def calc_feat_mean_std(self, input, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = input.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = input.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std, torch.cat([feat_mean, feat_std], dim = 1)

    def reset_parameters(self):
        for module in (self.intermediate_shape, self.intermediate_color):
            for linear in module.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.kaiming_uniform_(linear.weight.data, a=_LEAKY_RELU_ALPHA, mode='fan_in',
                                             nonlinearity='leaky_relu')
        for linear in (self.shape_project, self.color_project):
            nn.init.normal_(linear.weight.data, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor):
        style_feats = self.encode_with_intermediate(x)
        _, _, style_feat_mean_std = self.calc_feat_mean_std(style_feats[-1])

        # features
        shape_feat = self.intermediate_shape(style_feat_mean_std)  # (n, c)
        color_feat = self.intermediate_color(style_feat_mean_std)
        # original codes
        shape_code = self.shape_project(shape_feat)  # (n, d)
        color_code = self.color_project(color_feat)
        # fuse, (1, d)
        shape_latent_code = shape_code.mean(dim=0, keepdim=True)
        color_latent_code = color_code.mean(dim=0, keepdim=True)

        return shape_latent_code, color_latent_code


@ENCODERS.register_module(name='style_encoder')
class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        
        vgg.load_state_dict(torch.load('VAE/vgg_normalised.pth'))
        fc_encoder.load_state_dict(torch.load('VAE/fc_encoder_iter_160000.pth'))

        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.fc_encoder = fc_encoder

        self._feature_dim = 512

        # intermediate layers
        self.fc_encoder = fc_encoder

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
        self.fc_encoder.apply(weights_init_kaiming)
    
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def get_content_feat(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

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
    
    def cal_style_loss(self, content, style, style_ori, n_rays_y, n_rays_x, trans=True):
        batch = content.shape[0]
        if trans:
            content = content.transpose(2, 1).reshape(batch, 3, n_rays_y, n_rays_x)
            style = style.transpose(2, 1).reshape(batch, 3, n_rays_y, n_rays_x)
        content_fea = self.get_content_feat(content)
        style_fea = self.get_content_feat(style)
        content_loss = F.mse_loss(content_fea, style_fea)

        output_style_feats, output_style_feat_mean_std = self.get_style_feat(style)
        style_feats, style_feat_mean_std = self.get_style_feat(style_ori.expand(batch, 3, 128, 128))
        style_loss = F.mse_loss(output_style_feat_mean_std, style_feat_mean_std)
        #style_loss = F.mse_loss(output_style_feats, style_feats)

        return content_loss, style_loss
    
    def calc_feat_mean_std(self, input, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = input.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = input.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std, torch.cat([feat_mean, feat_std], dim = 1)

    def _calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def adaptive_instance_normalization(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = self._calc_mean_std(style_feat)
        content_mean, content_std = self._calc_mean_std(content_feat)
        
        content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        content_feat = content_feat * style_std.expand(size) + style_mean.expand(size)
        
        return content_feat

    def forward(self, x: torch.Tensor):
        style_feats = self.encode_with_intermediate(x)
        _, _, style_feat_mean_std = self.calc_feat_mean_std(style_feats[-1])

        intermediate = self.fc_encoder(style_feat_mean_std)
        intermediate_mean = intermediate[:, :512]

        return intermediate_mean#, style_feats[-1]

@ENCODERS.register_module(name='mask_encoder')
class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        
        vgg.load_state_dict(torch.load('VAE/vgg_normalised.pth'))
        fc_encoder.load_state_dict(torch.load('VAE/fc_encoder_iter_160000.pth'))

        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.fc_encoder = fc_encoder

        self._feature_dim = 512

        # intermediate layers
        self.fc_encoder = fc_encoder

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
        self.fc_encoder.apply(weights_init_kaiming)
                
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        #print("Encoder Using: Mask Encoder")
    
    def get_pool_mask(self, mask):
        results = [mask]
        for i in range(3):
            result = self.maxpool(results[-1])
            results.append(result)
        return results
    
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def get_content_feat(self, input, mask = None):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        if mask != None:
            input = input * mask
        return input

    def get_style_feat(self, input, mask = None):
        style_feats = self.encode_with_intermediate(input)
        out_mean = []
        out_std = []
        out_mean_std = []
        for (idx,style_feat) in enumerate (style_feats):
            size = style_feat.size()
            N, C = size[:2]
            if mask != None:
                style_feat_mask = torch.masked_select(style_feat,mask[idx].bool()).reshape(N, C, -1)
                if style_feat_mask.size()[2] <= 1: #会触发NaN的情况提前退出
                    return style_feat, None
                style_feat_mean, style_feat_std, style_feat_mean_std = self.calc_feat_mean_std(style_feat_mask)
            else:
                style_feat_mean, style_feat_std, style_feat_mean_std = self.calc_feat_mean_std(style_feat)
            out_mean.append(style_feat_mean)
            out_std.append(style_feat_std)
            out_mean_std.append(style_feat_mean_std)
        return style_feats, torch.cat(out_mean_std, dim=-1)
    
    def cal_style_loss(self, content, style, style_ori, n_rays_y, n_rays_x, mask, trans=True):
        
        batch = content.shape[0]
        if trans:
            mask = ((mask.transpose(2, 1).reshape(batch, 1, n_rays_y, n_rays_x)) > 0.5).float()
            content = content.transpose(2, 1).reshape(batch, 3, n_rays_y, n_rays_x)
            style = style.transpose(2, 1).reshape(batch, 3, n_rays_y, n_rays_x)
        mask_feats = self.get_pool_mask(mask)
        content_fea = self.get_content_feat(content, mask_feats[3])
        style_fea = self.get_content_feat(style, mask_feats[3])
        content_loss = F.mse_loss(content_fea, style_fea)

        output_style_feats, output_style_feat_mean_std = self.get_style_feat(style, mask_feats)
        style_feats, style_feat_mean_std = self.get_style_feat(style_ori.expand(batch, 3, 128, 128))
        if output_style_feat_mean_std == None:
            style_loss = 0
        else:
            style_loss = F.mse_loss(output_style_feat_mean_std, style_feat_mean_std)
        #style_loss = F.mse_loss(output_style_feats, style_feats)
        if style_loss!=style_loss or content_loss!=content_loss:
            content_loss = 0

        return content_loss, style_loss
    
    def calc_feat_mean_std(self, input, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = input.size()
        assert (len(size) == 4 or len(size) == 3)
        N, C = size[:2]
        feat_var = input.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std, torch.cat([feat_mean, feat_std], dim = 1)

    def forward(self, x: torch.Tensor):
        style_feats = self.encode_with_intermediate(x)
        _, _, style_feat_mean_std = self.calc_feat_mean_std(style_feats[-1])

        intermediate = self.fc_encoder(style_feat_mean_std)
        intermediate_mean = intermediate[:, :512]

        return intermediate_mean#, style_feats[-1]