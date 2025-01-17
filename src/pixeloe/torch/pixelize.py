import torch
import torch.nn.functional as F

from .outline import outline_expansion
from .color import match_color, quantize_and_dither
from .downscale.contrast_based import contrast_downscale

from .downscale.k_centroid import k_centroid_downscale_torch


def pixelize(
    img_t,
    target_size=256,
    patch_size=6,
    thickness=3,
    mode="contrast",
    do_color_match=True,
    do_quant=False,
    num_centroids=32,
    quant_mode="ordered",
):
    """
    Main pipeline: pixelize an image using PyTorch.
        img_t: Input RGB image tensor [B,C,H,W] with range [0..1]
    """

    if thickness > 0:
        expanded, _ = outline_expansion(img_t, thickness, thickness, patch_size)
    else:
        expanded = img_t

    if do_color_match:
        expanded = match_color(expanded, img_t)

    h, w = expanded.shape[2], expanded.shape[3]
    ratio = w / h
    out_h = int((target_size**2 / ratio) ** 0.5)
    out_w = int(out_h * ratio)

    if mode == "contrast":
        down = contrast_downscale(expanded, patch_size)
    elif mode == "k_centroid":
        down = k_centroid_downscale_torch(expanded, target_size, 2)
    else:
        down = F.interpolate(expanded, size=(out_h, out_w), mode="nearest-exact")

    if do_quant:
        down_final = quantize_and_dither(down, num_centroids=num_centroids, dither_method=quant_mode)
        down_final = match_color(down_final, down)
    else:
        down_final = down

    out_pixel = F.interpolate(down_final, scale_factor=patch_size, mode="nearest-exact")

    return out_pixel
