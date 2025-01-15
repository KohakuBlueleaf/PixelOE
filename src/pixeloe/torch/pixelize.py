import torch.nn.functional as F
from .outline import outline_expansion
from .downscale.contrast_based import contrast_downscale
from .color import match_color, color_quantization_kmeans, quantize_and_dither


def pixelize_pytorch(
    img_t,
    target_size=256,
    patch_size=6,
    thickness=3,
    mode="contrast",
    do_color_match=True,
    do_quant=False,
    K=32,
    quant_mode="ordered",
):
    """
    Main pipeline: pixelize an image using PyTorch.
    """

    if thickness > 0:
        expanded, w = outline_expansion(img_t, thickness, thickness, patch_size)
    else:
        expanded = img_t

    if do_color_match:
        expanded = match_color(expanded[None], img_t[None])[0]

    H, W = expanded.shape[1], expanded.shape[2]
    ratio = W / H
    out_h = int((target_size**2 / ratio) ** 0.5)
    out_w = int(out_h * ratio)

    if mode == "contrast":
        down = contrast_downscale(expanded, patch_size)
    else:
        down = F.interpolate(
            expanded.unsqueeze(0), size=(out_h, out_w), mode="nearest"
        )[0]

    if do_quant:
        down_q = quantize_and_dither(down, K=K, dither_method=quant_mode)
        down_final = match_color(down_q[None], down[None])[0]
    else:
        down_final = down

    out_pixel = F.interpolate(
        down_final.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0]

    return out_pixel
