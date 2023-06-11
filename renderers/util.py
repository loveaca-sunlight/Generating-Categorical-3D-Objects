import torch
import random

from tools import select_cameras
from pytorch3d.renderer.cameras import PerspectiveCameras
from typing import List

TINY_NUMBER = 1e-6

def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)

def get_name_idx_mapping(names: List[str]):
    return {name: idx for idx, name in enumerate(names)}


def zero_param(param: torch.nn.Parameter):
    param.data.zero_()


def set_transform_code(param: torch.nn.Parameter, full_transform: bool, rigid_scale: float = 1.0):
    if full_transform:
        data = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float, device=param.device) / rigid_scale
        param.data.copy_(data)
    else:
        zero_param(param)


def select_template_weights(state_dict: dict, fine_only: bool = False):
    """
    Select weights w.r.t template from given state_dict. Only suit for template_mlp
    :param state_dict:
    :param fine_only:
    :return:
    """
    function_name = '_implicit_function.'
    selected_weights = {}

    # Select Weights
    for name, param in state_dict.items():
        if function_name in name:
            for render_pass in (('fine',) if fine_only else ('coarse', 'fine')):
                if f'{function_name}{render_pass}.' in name:
                    for module in ['mlp_xyz', 'density_layer']:
                        path = f'{function_name}{render_pass}.{module}.'
                        if path in name:
                            selected_weights[name[name.index(path):]] = param

    return selected_weights

def select_template_weights_r(state_dict: dict, fine_only: bool = False):
    """
    Select weights w.r.t template from given state_dict. Only suit for template_mlp
    :param state_dict:
    :param fine_only:
    :return:
    """
    function_name = 'volumetric_function'
    selected_weights = {}

    # Select Weights
    for name, param in state_dict.items():
        if function_name in name:
            #for render_pass in (('fine',) if fine_only else ('coarse', 'fine')):
            for module in ['mlp_xyz', 'density_layer']:
                path = f'{function_name}.{module}.'
                if path in name:
                    selected_weights[name[name.index(path):]] = param

    return selected_weights

def select_template_weights_hyper(state_dict: dict, fine_only: bool = False):
    """
    Select weights w.r.t template from given state_dict. Only suit for template_mlp
    :param state_dict:
    :param fine_only:
    :return:
    """
    function_name = 'volumetric_function'
    encoder_name = 'model.encoder'#保证不被其他encoder顶替出错了
    deviations_name = 'deviations'
    deformer_name = 'model.deformer'
    transform_name = 'transforms'
    selected_weights = {}

    # Select Weights
    for name, param in state_dict.items():
        if function_name in name:
            #for render_pass in (('fine',) if fine_only else ('coarse', 'fine')):
            for module in ['mlp_xyz', 'density_layer', 'diffuse_layer', 'mlp_diffuse']:
                path = f'{function_name}.{module}.'
                if path in name:
                    selected_weights[name[name.index(path):]] = param  #去一下model
        if encoder_name in name:
            selected_weights[name[name.index('encoder'):]] = param
        if deformer_name in name:
            selected_weights[name[name.index('deformer'):]] = param
        if deviations_name in name:
            selected_weights[name[name.index('deviations'):]] = param
        if transform_name in name:
            selected_weights[name[name.index('transforms'):]] = param


    return selected_weights

def select_template_weights_s(state_dict: dict, fine_only: bool = False):
    """
    Select weights w.r.t template from given state_dict. Only suit for template_mlp
    :param state_dict:
    :param fine_only:
    :return:
    """
    function_name = 'volumetric_function.'
    selected_weights = {}
    lin='lin'
    # Select Weights
    for name, param in state_dict.items():
        if function_name in name:
            if f'{function_name}{lin}' in name:
                for i in range(0,9):
                    path = f'{function_name}{lin+str(i)}.'
                    if path in name:
                        selected_weights[name[name.index(path):]] = param

    return selected_weights


def select_function_weights(state_dict: dict, fine_only: bool = False):
    """
    Select weights of implicit function from given state_dict
    :param state_dict:
    :param fine_only:
    :return:
    """
    function_name = '_implicit_function.'
    selected_weights = {}

    # Select Weights
    for name, param in state_dict.items():
        if function_name in name:
            for render_pass in (('fine',) if fine_only else ('coarse', 'fine')):
                path = f'{function_name}{render_pass}.'
                if path in name:
                    selected_weights[name[(name.index(path) + len(path)):]] = param

    return selected_weights


def select_encoder_weights(state_dict: dict):
    """
    Select weights of encoder from given state_dict
    :param state_dict:
    :return:
    """
    function_name = 'encoder.'
    selected_weights = {}

    # Select Weights
    for name, param in state_dict.items():
        if function_name in name:
            selected_weights[name[(name.index(function_name) + len(function_name)):]] = param

    return selected_weights


def compute_view_directions(cameras: PerspectiveCameras):
    """
    Return view direction of each camera
    :param cameras:
    :return:
    """
    points_ndc = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 2]
        ], dtype=torch.float, device=cameras.device
    ).view(1, 2, 3)

    points_world = cameras.unproject_points(points_ndc, world_coordinates=True)  # (n, 2, 3)

    directions = points_world[:, 1, :] - points_world[:, 0, :]  # norm is unchanged after R and T, (n, 3)

    return directions
