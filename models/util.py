from typing import List
import warnings

import torch
from pytorch3d.renderer.cameras import CamerasBase

from tools import select_cameras
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, InterpolationMode
from modules.encoder.sparse_conv import TensorWithMask

import numpy as np
from PIL import Image
import glob
import os

def find_files(dir, exts):
    # types should be ['*.png', '*.jpg']
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
    if len(files_grabbed) > 0:
        files_grabbed = sorted(files_grabbed)
    return files_grabbed

def mask_image(image: torch.Tensor, fg_probability: torch.Tensor, threshold: float, background = 0.0):
    """
    Mask an image with given mask
    :param image: (n, c, h, w)
    :param fg_probability: (n, 1, h, w)
    :param threshold:
    :return:
    """
    clamped_mask = fg_probability.clone()
    clamped_mask[clamped_mask < threshold] = background
    return image * clamped_mask


def choose_views(ids: List[int], *choose_from):
    """
    Choose view data specified by ids
    :param ids:
    :param choose_from:
    :return:
    """
    outputs = []
    for data in choose_from:
        if isinstance(data, CamerasBase):
            outputs.append(
                select_cameras(data, ids)
            )
        elif isinstance(data, torch.Tensor):
            # assert data.dim() == 4, f'Expect input tensor has four dimensions (n, c, h, w), but given {data.dim()}.'
            outputs.append(
                data[ids]
                # data[ids, :, :, :]
            )
        else:
            raise TypeError(f'Unsupported type: {type(data).__name__}.')

    return outputs[0] if len(outputs) == 1 else outputs


def mask_images_background(images: torch.Tensor, ids: List[int], thr: float = 0.01):
    """
    Mask the background of images in place.
    :param images: images of shape (N, H, W, C)
    :param ids:
    :param thr:
    :return:
    """
    selected_images = images[ids, :, :, :]
    fg_probability = (torch.mean(selected_images, dim=-1, keepdim=True) > thr).float()  # (n, h, w, 1)
    images[ids, :, :, :] = fg_probability * selected_images + (1.0 - fg_probability) * 0.8


def select_key_points(kp_dict: dict, keywords: List[str], stack_dim: int = 1):
    """
    Select key points according to given keywords
    :param kp_dict: [str, (n, 3)]
    :param keywords:
    :param stack_dim:
    :return:
    """
    sel_kps = []
    sel_ids = []

    for idx, key in enumerate(keywords):
        if key in kp_dict:
            sel_kps.append(kp_dict[key])
            sel_ids.append(idx)

    assert len(sel_ids) > 0, 'No key point is selected.'

    return torch.stack(sel_kps, dim=stack_dim), sel_ids


def augment_source_images(src_img: torch.Tensor):
    affine = RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    )
    flip = RandomHorizontalFlip()

    transforms = Compose([affine, flip])

    return transforms(src_img)


def prepare_sparse_inputs(src_data: torch.Tensor, src_mask: torch.Tensor, mask_thr: float):
    mask = (src_mask > mask_thr).float()

    return TensorWithMask(
        src_data,
        mask
    )

def load_image(path):
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im

def load_mask(path):
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)
    mask = mask.astype(np.float32) / 255.0
    return mask[None]  # fake feature channel

def get_bbox_from_mask(mask, thr, decrease_quant=0.05):
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0

def _get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]

def get_clamp_bbox(bbox, box_crop_context=0.0, impath=""):
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        warnings.warn(f"squashed image {impath}!!")
        return None

    bbox[2:] = torch.clamp(bbox[2:], 2)
    bbox[2:] += bbox[0:2] + 1  # convert to [xmin, ymin, xmax, ymax]
    # +1 because upper bound is not inclusive

    return bbox

def crop_around_box(tensor, bbox, impath=""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox[[0, 2]] = torch.clamp(bbox[[0, 2]], 0.0, tensor.shape[-1]) #限制上下限
    bbox[[1, 3]] = torch.clamp(bbox[[1, 3]], 0.0, tensor.shape[-2])
    bbox = bbox.round().long()
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"

    return tensor

def crop_around_box(tensor, bbox, impath=""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox[[0, 2]] = torch.clamp(bbox[[0, 2]], 0.0, tensor.shape[-1]) #限制上下限
    bbox[[1, 3]] = torch.clamp(bbox[[1, 3]], 0.0, tensor.shape[-2])
    bbox = bbox.round().long()
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"

    return tensor

def resize_image(image, shape=None, mode="bilinear"):
    
    if shape == None:
        # skip the resizing
        imre_ = torch.from_numpy(image)
        return imre_, 1.0, torch.ones_like(imre_[:1])
    # takes numpy array, returns pytorch tensor
    _, image_height, image_width = shape
    minscale = min(
        image_height / image.shape[-2],
        image_width / image.shape[-1],
    )
    imre = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None],
        scale_factor=minscale,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        recompute_scale_factor=True,
    )[0]
    imre_ = torch.zeros(image.shape[0], image_height, image_width)
    imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
    mask = torch.zeros(1, image_height, image_width)
    mask[:, 0 : imre.shape[1] - 1, 0 : imre.shape[2] - 1] = 1.0
    return imre_, minscale, mask