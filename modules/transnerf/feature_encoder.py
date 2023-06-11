import torch
import torch.nn as nn

from ..resnet_backbone import resnet18_backbone, resnet34_backbone, resnet50_backbone
from nerf.implicit_function import _xavier_init
from torchvision.models import vgg11, vgg13, vgg16
import warnings


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


def _prepare_encoder(name: str, pretrained: bool):
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


class LatentCodeEncoder(nn.Module):
    def __init__(self, encoder_name: str, shape_code_dim: int, color_code_dim: int, single_input: bool,
                 reduction: bool = True, pretrained: bool = True):
        super().__init__()

        encoder, feature_dim = _prepare_encoder(encoder_name, pretrained)

        self.encoder = encoder
        print(f'Using encoder: {encoder_name}.')

        self.shape_project = nn.Conv2d(feature_dim, shape_code_dim, 1)
        _xavier_init(self.shape_project)
        self.color_project = nn.Conv2d(feature_dim, color_code_dim, 1)
        _xavier_init(self.color_project)

        self._single_input = single_input
        self._reduction = reduction

        warnings.warn('Consider to use encoder in modules.encoder directory.')

    def forward(self, images):
        """
        Compute latent code of input image x
        :param images:
        :return:
        """
        if self._single_input:
            assert images.shape[0] == 1, f'Can only input one image when single_input==True, ' \
                                         f'but given {images.shape[0]}.'

        feature = self.encoder(images)

        # （n, c)
        shape_latent_code = self.shape_project(feature).mean(-1).mean(-1)
        color_latent_code = self.color_project(feature).mean(-1).mean(-1)

        # reduction, (1, c)
        if (not self._single_input) and self._reduction:
            shape_latent_code = torch.mean(shape_latent_code, dim=0, keepdim=True)
            color_latent_code = torch.mean(color_latent_code, dim=0, keepdim=True)

        return shape_latent_code, color_latent_code
