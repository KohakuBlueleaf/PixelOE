from functools import cache

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from . import env
from ..logger import logger


@cache
def compile_warning_log_once():
    logger.warning(
        "Torch compile is not enabled. "
        "This may result in large vram usage and slow performance."
    )


def compile_wrapper(func):
    compiled = torch.compile(
        func,
        dynamic=True,
        options={
            "shape_padding": True,
        },
    )

    def runner(*args, **kwargs):
        if env.TORCH_COMPILE:
            return compiled(*args, **kwargs)
        else:
            compile_warning_log_once()
            return func(*args, **kwargs)

    return runner


def to_numpy(tensor):
    """
    Convert a torch.Tensor [B,C,H,W] with range [0..1]
    back to a NumPy HWC image [0..255].
    """
    return list(
        (tensor.float().permute(0, 2, 3, 1).cpu().numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def pre_resize(
    img_pil,
    target_size=128,
    patch_size=8,
):
    if isiterable(target_size):
        in_w = target_size[0] * patch_size
        in_h = target_size[1] * patch_size
    else:
        W, H = img_pil.size
        ratio = W / H
        in_h = (target_size**2 / ratio) ** 0.5
        in_w = int(in_h * ratio) * patch_size
        in_h = int(in_h) * patch_size
    img_pil = img_pil.resize((in_w, in_h), Image.Resampling.BICUBIC)
    img_t = to_tensor(img_pil)
    return img_t[None]


@compile_wrapper
def batched_kmeans_iter(datas, centroids, cs=None):
    """
    datas: (B, N, C)
    centroids: (B, K, C)
    cs: (B, N)
    """
    K = centroids.shape[1]
    if cs is None:
        cs = torch.arange(K, device=datas.device)
    dists = (datas - centroids.unsqueeze(1)).pow(2).sum(dim=-1)
    labels = dists.argmin(dim=-1).unsqueeze(-1).repeat(1, 1, K)
    masks = labels == cs
    mask_valid = masks.sum(dim=1) > 0
    cluster_sums = torch.sum(masks.unsqueeze(-1) * datas, dim=1)
    cluster_means = cluster_sums / masks.sum(dim=1)[:, :, None]
    new_centroids = torch.where(mask_valid[:, :, None], cluster_means, centroids)
    diff = torch.max((new_centroids - centroids).abs())
    return new_centroids, diff
