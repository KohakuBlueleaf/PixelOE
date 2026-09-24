"""Block downscalers: contrast-based (Lab) and k-centroid (RGB, k = 2)."""

from .color import lab_image
from .reduce import stop_diff

CONTRAST = "downscale/contrast"
KCENTROID = "downscale/kcentroid"

KCENTROID_ITERS = 4  # max(2 * int(2 ** 0.5), 4)


def contrast_downscale(ctx, img, p):
    b, _, h, w = img.shape
    out_h, out_w = h // p, w // p
    lab = lab_image(ctx, img)
    dst = ctx.empty((b, 3, out_h, out_w))
    ctx.dispatch(
        CONTRAST,
        "contrast_downscale",
        (out_w, out_h, b),
        lab_img=lab,
        dst=dst,
        height=h,
        width=w,
        p=p,
        out_h=out_h,
        out_w=out_w,
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
    item_diff = ctx.empty((blocks,))
    diff = ctx.empty((KCENTROID_ITERS,))
    ctx.clear_uint(diff)
    ctx.dispatch(KCENTROID, "kc_init", grid, img=img, cent=cent, **dims)
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
