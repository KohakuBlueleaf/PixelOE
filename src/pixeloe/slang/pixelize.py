"""Slang pipeline reproducing pixeloe.torch.pixelize.pixelize."""

from .ops.color import match_color
from .ops.downscale import contrast_downscale, k_centroid_downscale
from .ops.outline import expansion_weight, normalized_weight, outline_expansion
from .ops.quant import quant_weights, quantize_and_dither
from .ops.resample import (
    interpolate,
    lanczos_resize,
    pad_replicate,
    upscale_nearest_exact,
)
from .ops.sharpen import sharpen
from .runtime.registry import get_context


def pixelize(
    img_t,
    pixel_size=6,
    thickness=3,
    mode="contrast",
    sharpen_mode=None,
    sharpen_factor=0.5,
    do_color_match=True,
    do_quant=False,
    num_colors=32,
    quant_mode="kmeans",
    dither_mode="ordered",
    no_post_upscale=False,
    return_intermediate=False,
    weight_mapping="current",
    weight_normalize="global",
    *,
    backend=None,
    colorfix_blur="exact",
    blur_impl=None,
    local_stats="lattice",
    stat_padding="zero",
    context=None,
):
    """Same arguments and results as the torch pipeline; img_t [B,3,H,W] in [0, 1].

    Extra keyword-only knobs:
        local_stats    outline-weight statistics: "lattice" (reference:
                       patches at stride p/2, overlap-add averaged) or
                       "sliding" (exact per-pixel windows)
        stat_padding   sliding windows outside the image: "zero" or "replicate"
        backend        slang device type ("cuda", "d3d12", "vulkan", "cpu");
                       default follows the input tensor's device
        colorfix_blur  "exact" (reference float16 2-D kernel) or "separable"
        blur_impl      exact-blur implementation: "tiled" (GPU groupshared),
                       "sym" (symmetric fold) or "direct"; default tiled on
                       GPU backends, sym on CPU
        context        an explicit runtime.Context
    """
    ctx = context or get_context(backend, img_t)
    img = ctx.from_torch(img_t)
    out, expanded, weight = run(
        ctx,
        img,
        pixel_size=pixel_size,
        thickness=thickness,
        mode=mode,
        sharpen_mode=sharpen_mode,
        sharpen_factor=sharpen_factor,
        do_color_match=do_color_match,
        do_quant=do_quant,
        num_colors=num_colors,
        quant_mode=quant_mode,
        dither_mode=dither_mode,
        no_post_upscale=no_post_upscale,
        keep_intermediate=return_intermediate,
        weight_mapping=weight_mapping,
        weight_normalize=weight_normalize,
        colorfix_blur=colorfix_blur,
        blur_impl=blur_impl,
        stats={"local_stats": local_stats, "stat_padding": stat_padding},
    )
    ctx.release(img)
    result = ctx.to_torch(out, like=img_t)
    ctx.release(out)
    if not return_intermediate:
        return result
    expanded_t = ctx.to_torch(expanded, like=img_t)
    ctx.release(expanded)
    weight_t = None
    if weight is not None:
        b, h, w = weight.shape
        weight_t = ctx.to_torch(weight, like=img_t).reshape(b, 1, h, w)
        ctx.release(weight)
    return result, expanded_t, weight_t


def run(
    ctx,
    img,
    pixel_size,
    thickness,
    mode,
    sharpen_mode,
    sharpen_factor,
    do_color_match,
    do_quant,
    num_colors,
    quant_mode,
    dither_mode,
    no_post_upscale,
    keep_intermediate,
    weight_mapping,
    weight_normalize,
    colorfix_blur,
    blur_impl,
    stats=None,
):
    """Device-side pipeline. Returns (out, expanded or None, weight or None);
    the caller owns the returned arrays; `img` is not consumed."""
    quant_mode = quant_mode.lower()
    weighted_quant = do_quant and quant_mode in {"weighted-kmeans", "repeat-kmeans"}
    repeat_mode = quant_mode == "repeat-kmeans"
    quant_mode = quant_mode.split("-")[-1]
    p = pixel_size
    stats = stats or {}

    _, _, h, w = img.shape
    out_h = h // p
    out_w = w // p
    pad_h = p - ((h % p) or p)
    pad_w = p - ((w % p) or p)
    owned_input = False
    if pad_h or pad_w:
        img = pad_replicate(
            ctx, img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2
        )
        owned_input = True
        out_h += 1
        out_w += 1
    target_size = (out_h * out_w) ** 0.5

    oe_weight = None
    if thickness > 0:
        expanded, oe_weight = outline_expansion(
            ctx,
            img,
            thickness,
            thickness,
            p,
            weight_mapping=weight_mapping,
            weight_normalize=weight_normalize,
            **stats,
        )
    else:
        expanded = img

    if sharpen_mode in ("unsharp", "laplacian"):
        sharp = sharpen(ctx, expanded, sharpen_mode, sharpen_factor)
        if expanded is not img:
            ctx.release(expanded)
        expanded = sharp

    weights = None
    if weighted_quant:
        if oe_weight is None:
            rw = expansion_weight(
                ctx,
                img,
                p,
                p // 2,
                mapping=weight_mapping,
                normalize=weight_normalize,
                **stats,
            )
            full = normalized_weight(ctx, rw)
        else:
            full = oe_weight
        weights = quant_weights(ctx, full, (out_h, out_w), target_size / 512)
        if full is not oe_weight:
            ctx.release(full)

    if do_color_match:
        matched = match_color(ctx, expanded, img, blur=colorfix_blur, impl=blur_impl)
        if expanded is not img:
            ctx.release(expanded)
        expanded = matched

    match mode:
        case "contrast":
            down = contrast_downscale(ctx, expanded, p)
        case "k_centroid":
            down = k_centroid_downscale(ctx, expanded, p)
        case "lanczos":
            down = lanczos_resize(ctx, expanded, (out_h, out_w))
            if down is None:
                down = ctx.clone(expanded)
        case _:
            down = interpolate(ctx, expanded, (out_h, out_w), mode)

    if do_quant:
        if weights is not None and weights.shape[1] != down.shape[2] * down.shape[3]:
            # the reference sizes the weights (out_h, out_w) but block modes
            # emit (H_pad // p, W_pad // p); it fails on the mismatch as well
            raise RuntimeError(
                f"quantization weights {weights.shape} do not match the "
                f"downscaled image {down.shape}"
            )
        quant = quantize_and_dither(
            ctx,
            down,
            weights=weights,
            num_centroids=num_colors,
            quant_mode=quant_mode,
            dither_method=dither_mode.lower(),
            repeat_mode=repeat_mode,
        )
        down_final = match_color(ctx, quant, down, blur=colorfix_blur, impl=blur_impl)
        ctx.release(quant, down)
    else:
        down_final = down
    ctx.release(weights)

    if no_post_upscale:
        out = down_final
    else:
        out = upscale_nearest_exact(ctx, down_final, p)
        ctx.release(down_final)

    if keep_intermediate:
        if expanded is img:
            expanded = ctx.clone(img)
    else:
        if expanded is not img:
            ctx.release(expanded)
        expanded = None
        ctx.release(oe_weight)
        oe_weight = None
    if owned_input:
        ctx.release(img)
    return out, expanded, oe_weight
