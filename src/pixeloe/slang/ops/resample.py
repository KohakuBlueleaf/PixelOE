"""Padding, torch-semantics interpolation, Lanczos, nearest-exact upscale."""

from functools import cache

import numpy as np
import torch

from ...torch.downscale.lanczos import compute_weights_and_indices

MODULE = "resample/resample"

INTERP_MODES = {
    "nearest": 0,
    "nearest-exact": 1,
    "bilinear": 2,
    "bicubic": 3,
    "area": 4,
}


def pad_replicate(ctx, src, top, bottom, left, right):
    b, c, h, w = src.shape
    out_h, out_w = h + top + bottom, w + left + right
    dst = ctx.empty((b, c, out_h, out_w))
    ctx.dispatch(
        MODULE,
        "pad_replicate",
        (out_w, out_h, b * c),
        src=src,
        dst=dst,
        in_h=h,
        in_w=w,
        out_h=out_h,
        out_w=out_w,
        top=top,
        left=left,
    )
    return dst


def interpolate(ctx, src, size, mode):
    """F.interpolate(src, size=size, mode=mode) -> new array."""
    if mode not in INTERP_MODES:
        raise ValueError(f"Unsupported interpolate mode: {mode}")
    b, c, h, w = src.shape
    out_h, out_w = size
    dst = ctx.empty((b, c, out_h, out_w))
    ctx.dispatch(
        MODULE,
        "interpolate",
        (out_w, out_h, b * c),
        src=src,
        dst=dst,
        mode=INTERP_MODES[mode],
        in_h=h,
        in_w=w,
        out_h=out_h,
        out_w=out_w,
        scale_h=np.float32(h) / np.float32(out_h),
        scale_w=np.float32(w) / np.float32(out_w),
    )
    return dst


@cache
def lanczos_csr(in_size, out_size, a=3, support_scaling=1.5):
    """Reference Lanczos weights as CSR (offsets, indices, weights)."""
    weights, idx = compute_weights_and_indices(
        in_size,
        out_size,
        out_size / in_size,
        a,
        torch.float32,
        torch.device("cpu"),
        support_scaling,
    )
    rows = idx[0].numpy().astype(np.int64)
    cols = idx[1].numpy().astype(np.uint32)
    offsets = np.zeros(out_size + 1, dtype=np.uint32)
    np.add.at(offsets, rows + 1, 1)
    offsets = np.cumsum(offsets).astype(np.uint32)
    order = np.argsort(rows, kind="stable")
    return offsets, cols[order], weights.numpy().astype(np.float32)[order]


def _csr_constants(ctx, in_size, out_size):
    offsets, cols, weights = lanczos_csr(in_size, out_size)
    key = ("lanczos", in_size, out_size)
    return (
        ctx.cached_constant(key + ("o",), lambda: offsets),
        ctx.cached_constant(key + ("i",), lambda: cols),
        ctx.cached_constant(key + ("w",), lambda: weights),
    )


def lanczos_resize(ctx, src, size):
    """Reference lanczos_resize(src, size) (a=3, support_scaling=1.5)."""
    b, c, h, w = src.shape
    out_h, out_w = size
    if (h, w) == (out_h, out_w):
        return None
    cur = src
    owned = False
    if h != out_h:
        offsets, cols, weights = _csr_constants(ctx, h, out_h)
        dst = ctx.empty((b, c, out_h, w))
        ctx.dispatch(
            MODULE,
            "csr_pass",
            (w, out_h, b * c),
            src=cur,
            dst=dst,
            offsets=offsets,
            indices=cols,
            weights=weights,
            axis=0,
            in_h=h,
            in_w=w,
            out_h=out_h,
            out_w=w,
            do_clamp=1 if w == out_w else 0,
        )
        cur, owned = dst, True
    if w != out_w:
        offsets, cols, weights = _csr_constants(ctx, w, out_w)
        dst = ctx.empty((b, c, out_h, out_w))
        ctx.dispatch(
            MODULE,
            "csr_pass",
            (out_w, out_h, b * c),
            src=cur,
            dst=dst,
            offsets=offsets,
            indices=cols,
            weights=weights,
            axis=1,
            in_h=out_h,
            in_w=w,
            out_h=out_h,
            out_w=out_w,
            do_clamp=1,
        )
        if owned:
            ctx.release(cur)
        cur = dst
    return cur


def upscale_nearest_exact(ctx, src, factor):
    """F.interpolate(src, scale_factor=factor, mode='nearest-exact')."""
    b, c, h, w = src.shape
    out_h, out_w = h * factor, w * factor
    dst = ctx.empty((b, c, out_h, out_w))
    ctx.dispatch(
        MODULE,
        "upscale_nearest_exact",
        (out_w, out_h, b * c),
        src=src,
        dst=dst,
        in_h=h,
        in_w=w,
        out_h=out_h,
        out_w=out_w,
        inv_scale=np.float32(1.0 / factor),
    )
    return dst
