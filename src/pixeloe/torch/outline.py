import torch
import torch.nn.functional as F

from .lab import rgb_to_lab
from .utils import compile_wrapper
from .minmax import dilate_cont, erode_cont, KERNELS


@compile_wrapper
def local_stat(tensor, kernel, stride, stat="median"):
    B, C, H, W = tensor.shape
    patches = F.unfold(tensor, kernel_size=kernel, stride=stride, padding=kernel // 2)
    if stat == "median":
        vals = patches.median(dim=1, keepdims=True).values.repeat(1, patches.size(1), 1)
    elif stat == "max":
        vals = patches.max(dim=1, keepdims=True).values.repeat(1, patches.size(1), 1)
    elif stat == "min":
        vals = patches.min(dim=1, keepdims=True).values.repeat(1, patches.size(1), 1)
    else:
        raise ValueError(f"Unsupported local stat: {stat}")
    div = F.fold(
        torch.ones_like(vals),
        output_size=(H, W),
        kernel_size=kernel,
        stride=stride,
        padding=kernel // 2,
    )
    out = F.fold(
        vals,
        output_size=(H, W),
        kernel_size=kernel,
        stride=stride,
        padding=kernel // 2,
    )
    return out / (div + 1e-8)


def normalize_weight(weight, mode="global"):
    if mode == "none":
        return weight
    if mode == "global":
        return (weight - weight.amin()) / (weight.amax() - weight.amin() + 1e-8)
    if mode == "per_image":
        reduce_dims = tuple(range(1, weight.ndim))
        minv = weight.amin(dim=reduce_dims, keepdim=True)
        maxv = weight.amax(dim=reduce_dims, keepdim=True)
        return (weight - minv) / (maxv - minv + 1e-8)
    raise ValueError(f"Unsupported weight normalization mode: {mode}")


@compile_wrapper
def outline_weight_stats(img, k=16, stride=4):
    lab = rgb_to_lab(img)  # [B,3,H,W]
    luminance = lab[:, 0:1] / 100  # [B,1,H,W]

    l_med = local_stat(luminance, k * 2, stride, stat="median")
    l_min = local_stat(luminance, k, stride, stat="min")
    l_max = local_stat(luminance, k, stride, stat="max")

    bright_dist = l_max - l_med
    dark_dist = l_med - l_min
    local_contrast = bright_dist + dark_dist

    return l_med, l_min, l_max, bright_dist, dark_dist, local_contrast


def polarity_score(l_med, bright_dist, dark_dist, avg_scale=10, dist_scale=3):
    background_polarity = l_med - 0.5
    detail_polarity = dark_dist - bright_dist
    return background_polarity * avg_scale + detail_polarity * dist_scale


@compile_wrapper
def current_weight_mapping(
    l_med,
    l_min,
    l_max,
    bright_dist,
    dark_dist,
    local_contrast,
    avg_scale=10,
    dist_scale=3,
    normalize="global",
):
    weight = torch.sigmoid(
        polarity_score(l_med, bright_dist, dark_dist, avg_scale, dist_scale)
    )
    return normalize_weight(weight, normalize)


@compile_wrapper
def contrast_ratio_weight_mapping(
    l_med,
    l_min,
    l_max,
    bright_dist,
    dark_dist,
    local_contrast,
    avg_scale=10,
    dist_scale=3,
    normalize="global",
):
    calc_dtype = (
        torch.float32 if local_contrast.dtype == torch.float16 else local_contrast.dtype
    )
    asymmetry = (bright_dist.to(calc_dtype) - dark_dist.to(calc_dtype)) / (
        local_contrast.to(calc_dtype) + 1e-8
    )
    weight = (l_med.to(calc_dtype) - 0.5) * avg_scale - asymmetry * dist_scale
    weight = torch.sigmoid(weight)
    return normalize_weight(weight, normalize).to(l_med.dtype)


@compile_wrapper
def contrast_gated_weight_mapping(
    l_med,
    l_min,
    l_max,
    bright_dist,
    dark_dist,
    local_contrast,
    avg_scale=10,
    dist_scale=3,
    normalize="global",
):
    weight = torch.sigmoid(
        polarity_score(l_med, bright_dist, dark_dist, avg_scale, dist_scale)
    )
    contrast_gate = torch.sigmoid((local_contrast - local_contrast.mean()) * dist_scale)
    weight = 0.5 + (weight - 0.5) * contrast_gate
    return normalize_weight(weight, normalize)


_WEIGHT_MAPPINGS = {
    "current": current_weight_mapping,
    "polarity": current_weight_mapping,
    "contrast_ratio": contrast_ratio_weight_mapping,
    "contrast_gated": contrast_gated_weight_mapping,
}


@compile_wrapper
def expansion_weight(
    img,
    k=16,
    stride=4,
    avg_scale=10,
    dist_scale=3,
    mapping="current",
    normalize="global",
):
    """
    Compute a weight matrix for outline expansion.
    """
    stats = outline_weight_stats(img, k, stride)
    try:
        mapping_func = _WEIGHT_MAPPINGS[mapping]
    except KeyError as exc:
        raise ValueError(f"Unsupported outline weight mapping: {mapping}") from exc
    return mapping_func(*stats, avg_scale, dist_scale, normalize)  # shape [B, 1, H,W]


def outline_expansion(
    img,
    erode_iters=2,
    dilate_iters=2,
    k=16,
    avg_scale=10,
    dist_scale=3,
    weight_mapping="current",
    weight_normalize="global",
):
    """
    Perform contrast-aware outline expansion on an image.
    """
    w = expansion_weight(
        img,
        k,
        k // 2,
        avg_scale,
        dist_scale,
        mapping=weight_mapping,
        normalize=weight_normalize,
    )

    e = erode_cont(img, KERNELS[erode_iters].to(img), 1)
    d = dilate_cont(img, KERNELS[dilate_iters].to(img), 1)

    out = e * w + d * (1.0 - w)

    oc_iter = max(erode_iters - 1, dilate_iters - 1, 1)

    out = erode_cont(out, KERNELS[oc_iter].to(img), 1)
    out = dilate_cont(out, KERNELS[oc_iter].to(img), 2)
    out = erode_cont(out, KERNELS[oc_iter].to(img), 1)

    return out, w
