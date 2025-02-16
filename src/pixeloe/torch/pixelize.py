import torch
import torch.nn.functional as F

from .outline import outline_expansion, expansion_weight
from .color import match_color, quantize_and_dither

from .downscale.contrast_based import contrast_downscale
from .downscale.k_centroid import k_centroid_downscale_torch
from .downscale.lanczos import lanczos_resize


def pixelize(
    img_t,
    pixel_size=6,
    thickness=3,
    mode="contrast",
    do_color_match=True,
    do_quant=False,
    num_colors=32,
    quant_mode="kmeans",
    dither_mode="ordered",
    no_post_upscale=False,
    return_intermediate=False,
):
    """
    Main pipeline: pixelize an image using PyTorch.
        img_t: Input RGB image tensor [B,C,H,W] with range [0..1]
    """
    quant_mode = quant_mode.lower()
    weighted_quant = do_quant and quant_mode in {"weighted-kmeans", "repeat-kmeans"}
    repeat_mode = quant_mode == "repeat-kmeans"
    quant_mode = quant_mode.split("-")[-1]

    h, w = img_t.shape[2], img_t.shape[3]
    out_h = h // pixel_size
    out_w = w // pixel_size
    pad_h = pixel_size - h % pixel_size
    pad_w = pixel_size - w % pixel_size
    if pad_h or pad_w:
        img_t = F.pad(
            img_t,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="replicate",
        )
        out_h += 1
        out_w += 1
    target_size = (out_h * out_w) ** 0.5

    oe_weights = None
    if thickness > 0:
        expanded, oe_weights = outline_expansion(
            img_t, thickness, thickness, pixel_size
        )
    else:
        expanded = img_t

    if weighted_quant:
        if oe_weights is None:
            weights = expansion_weight(img_t, pixel_size, pixel_size // 2)
        else:
            weights = oe_weights
        weights = torch.abs(weights * 2 - 1) * weights
        weights = F.interpolate(weights, size=(out_h, out_w), mode="bilinear")
        w_gamma = target_size / 512
        weights = weights**w_gamma
    else:
        weights = None

    if do_color_match:
        expanded = match_color(expanded, img_t)

    match mode:
        case "contrast":
            down = contrast_downscale(expanded, pixel_size)
        case "k_centroid":
            down = k_centroid_downscale_torch(expanded, pixel_size, 2)
        case "lanczos":
            down = lanczos_resize(expanded, size=(out_h, out_w))
        case mode:
            down = F.interpolate(expanded, size=(out_h, out_w), mode=mode)

    if do_quant:
        down_final = quantize_and_dither(
            down,
            weights=weights,
            num_centroids=num_colors,
            quant_mode=quant_mode,
            dither_method=dither_mode,
            repeat_mode=repeat_mode,
        )
        down_final = match_color(down_final, down)
    else:
        down_final = down

    if no_post_upscale:
        out_pixel = down_final
    else:
        out_pixel = F.interpolate(down_final, scale_factor=pixel_size, mode="nearest-exact")

    if return_intermediate:
        return out_pixel, expanded, oe_weights

    return out_pixel
