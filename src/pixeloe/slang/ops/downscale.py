"""Block downscalers: contrast-based (Lab) and k-centroid (RGB, k = 2)."""

from .color import lab_image
from .reduce import stop_diff

CONTRAST = "downscale/contrast"
KCENTROID = "downscale/kcentroid"
KCENTROID_ATOMIC = "downscale/kcentroid_atomic"  # integer atomics: GPU only
ATOMIC_STOP = True  # GPU: k-centroid stop value by atomic max (else reduction)

KCENTROID_ITERS = 4  # max(2 * int(2 ** 0.5), 4)
# block sizes with register-resident entry points (contrast_downscale_p<P>)
CONTRAST_REGISTER_SIZES = (2, 3, 4, 5, 6, 8)


def contrast_downscale(ctx, img, p, upscale=False):
    """[B,3,H//p,W//p]; upscale=True: its nearest-exact upscale by p instead,
    or None when that fusion is unavailable for p (the caller upscales)."""
    b, _, h, w = img.shape
    out_h, out_w = h // p, w // p
    dims = {"height": h, "width": w, "out_h": out_h, "out_w": out_w}
    grid = (out_w, out_h, b)
    if p in CONTRAST_REGISTER_SIZES:  # Lab converted in the kernel
        shape = (b, 3, out_h * p, out_w * p) if upscale else (b, 3, out_h, out_w)
        dst = ctx.empty(shape)
        ctx.dispatch(
            CONTRAST,
            f"contrast_rgb_p{p}",
            grid,
            lab_img=img,
            dst=dst,
            up=1 if upscale else 0,
            **dims,
        )
        return dst
    if upscale:
        return None
    dst = ctx.empty((b, 3, out_h, out_w))
    lab = lab_image(ctx, img)
    ctx.dispatch(
        CONTRAST, "contrast_downscale", grid, lab_img=lab, dst=dst, p=p, **dims
    )
    ctx.release(lab)
    return dst


def k_centroid_downscale(ctx, img, p):
    b, _, h, w = img.shape
    out_h, out_w = h // p, w // p
    blocks = b * out_h * out_w
    grid = (out_w, out_h, b)
    dims = {"height": h, "width": w, "p": p, "out_h": out_h, "out_w": out_w}
    cent = ctx.empty((blocks, 8))
    ctx.dispatch(KCENTROID, "kc_init", grid, img=img, cent=cent, **dims)
    if ctx.backend != "cpu" and ATOMIC_STOP:  # one dispatch per iteration
        diff_bits = ctx.empty((KCENTROID_ITERS,), "uint32")
        ctx.clear_uint(diff_bits)
        for it in range(KCENTROID_ITERS):
            ctx.dispatch(
                KCENTROID_ATOMIC,
                "kc_iter_max",
                grid,
                img=img,
                cent=cent,
                diff_bits=diff_bits,
                it=it,
                **dims,
            )
        dst = ctx.empty((b, 3, out_h, out_w))
        ctx.dispatch(KCENTROID, "kc_final", grid, img=img, cent=cent, dst=dst, **dims)
        ctx.release(cent, diff_bits)
        return dst
    item_diff = ctx.empty((blocks,))
    diff = ctx.empty((KCENTROID_ITERS,))
    ctx.clear_uint(diff)
    for it in range(KCENTROID_ITERS):
        ctx.dispatch(
            KCENTROID,
            "kc_iter",
            grid,
            img=img,
            cent=cent,
            item_diff=item_diff,
            diff=diff,
            it=it,
            **dims,
        )
        stop_diff(ctx, item_diff, blocks, diff, it)
    dst = ctx.empty((b, 3, out_h, out_w))
    ctx.dispatch(KCENTROID, "kc_final", grid, img=img, cent=cent, dst=dst, **dims)
    ctx.release(cent, item_diff, diff)
    return dst
