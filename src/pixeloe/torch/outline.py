import torch
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from .env import TORCH_COMPILE
from .minmax import dilate_cont, erode_cont, KERNELS


@torch.compile(disable=not TORCH_COMPILE)
def local_stat(tensor, kernel, stride, stat="median"):
    B, C, H, W = tensor.shape
    patches = F.unfold(tensor, kernel_size=kernel, stride=stride, padding=kernel // 2)
    if stat == "median":
        vals = patches.median(dim=1, keepdims=True).values.repeat(1, patches.size(1), 1)
    elif stat == "max":
        vals = patches.max(dim=1, keepdims=True).values.repeat(1, patches.size(1), 1)
    elif stat == "min":
        vals = patches.min(dim=1, keepdims=True).values.repeat(1, patches.size(1), 1)
    out = F.fold(
        vals,
        output_size=(H, W),
        kernel_size=kernel,
        stride=stride,
        padding=kernel // 2,
    )
    return out / ((kernel // stride) ** 2 + 1e-8)


@torch.compile(disable=not TORCH_COMPILE)
def expansion_weight(img, k=16, stride=4, avg_scale=10, dist_scale=3):
    """
    Compute a weight matrix for outline expansion.
    """
    lab = rgb_to_lab(img)  # [B,3,H,W]
    L = lab[:, 0:1] / 100  # [B,1,H,W]

    L_med = local_stat(L, k * 2, stride, stat="median")
    L_min = local_stat(L, k, stride, stat="min")
    L_max = local_stat(L, k, stride, stat="max")

    bright_dist = L_max - L_med
    dark_dist = L_med - L_min

    weight = (L_med - 0.5) * avg_scale - (bright_dist - dark_dist) * dist_scale
    weight = torch.sigmoid(weight)

    weight = (weight - weight.amin()) / (weight.amax() - weight.amin() + 1e-8)
    return weight  # shape [B, 1, H,W]


def outline_expansion(
    img, erode_iters=2, dilate_iters=2, k=16, avg_scale=10, dist_scale=3
):
    """
    Perform contrast-aware outline expansion on an image.
    """
    w = expansion_weight(img, k, k // 2, avg_scale, dist_scale)

    e = erode_cont(img, KERNELS[erode_iters], 1)
    d = dilate_cont(img, KERNELS[dilate_iters], 1)

    out = e * w + d * (1.0 - w)

    oc_iter = max(erode_iters - 2, dilate_iters - 2, 1)

    out = erode_cont(out, KERNELS[oc_iter], 1)
    out = dilate_cont(out, KERNELS[oc_iter], 2)
    out = erode_cont(out, KERNELS[oc_iter], 1)

    return out, w
