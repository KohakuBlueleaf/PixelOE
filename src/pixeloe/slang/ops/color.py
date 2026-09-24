"""Colour conversion and colour matching (match_color + wavelet colour fix)."""

from functools import cache

import numpy as np
import torch

from ...torch.color import gaussian_kernel
from .reduce import lab_moments_pair

CONVERT = "color/convert"
MATCH = "color/match"
BLUR = "color/blur"
BLUR_TILED = "color/blur_tiled"  # groupshared: GPU backends only
BLUR_LOWRANK = "color/blur_lowrank"
BLUR_LOWRANK_SHARED = "color/blur_lowrank_shared"  # groupshared: GPU only

WAVELET_LEVELS = 5
BLUR_MODES = ("exact", "separable")
# exact-kernel implementations: "tiled" (groupshared, GPU only), "sym"
# (symmetric fold straight from memory), "direct" (plain 2-D loop),
# "lowrank" (rank-`blur_rank` SVD of the kernel as row/column passes)
BLUR_IMPLS = ("tiled", "sym", "direct", "lowrank")
# blur_tiled.slang entry points: radius -> (outputs per thread, thread rows)
TILED_SHAPES = {2: (4, 4), 4: (4, 4), 8: (4, 4), 16: (8, 4), 32: (8, 2)}
SYM_RADII = (2, 4, 8, 16, 32)  # blur.slang blur2d_sym_rN, 4 outputs per thread
# blur_lowrank.slang lr_rows_rN / lr_cols_rN: LOWRANK_OUT outputs per thread
LOWRANK_BLOCKED_RADII = (2, 4, 8, 16, 32)
LOWRANK_OUT = 8


def default_blur_impl(ctx):
    return "lowrank"


def luminance(ctx, img):
    """[B,3,H,W] sRGB -> [B,H,W] L / 100."""
    b, _, h, w = img.shape
    lum = ctx.empty((b, h, w))
    ctx.dispatch_flat(
        CONVERT, "luminance", b * h * w, img=img, lum=lum, pixels=b * h * w, hw=h * w
    )
    return lum


def lab_image(ctx, img):
    """[B,3,H,W] sRGB -> [B,3,H,W] Lab."""
    b, _, h, w = img.shape
    lab = ctx.empty(img.shape)
    ctx.dispatch_flat(
        CONVERT,
        "lab_image",
        b * h * w,
        img=img,
        lab_out=lab,
        pixels=b * h * w,
        hw=h * w,
    )
    return lab


@cache
def exact_blur_table(radius):
    """The reference 2-D kernel: float16 outer product / float16 sum."""
    return gaussian_kernel(radius, torch.device("cpu")).float().numpy().reshape(-1)


@cache
def separable_blur_table(radius):
    """float32 1-D Gaussian (sigma = radius) whose outer product is the
    float32 version of the reference kernel."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * radius * radius))
    return (k / k.sum()).astype(np.float32)


@cache
def lowrank_blur_tables(radius, rank):
    """[rank row vectors, rank column vectors] (float32, singular values in
    the columns) of the exact kernel's SVD, computed in float64. rank is
    clipped to the kernel size, where the sum equals the kernel."""
    k = 2 * radius + 1
    table = exact_blur_table(radius).astype(np.float64).reshape(k, k)
    u, s, vt = np.linalg.svd(table)
    rank = min(rank, k)
    rows = vt[:rank]
    cols = (u[:, :rank] * s[:rank]).T
    return np.concatenate([rows.reshape(-1), cols.reshape(-1)]).astype(np.float32)


def blur_table(ctx, kind, radius):
    factory = exact_blur_table if kind == "exact" else separable_blur_table
    return ctx.cached_constant(("blur", kind, radius), lambda: factory(radius))


def _blur_lowrank(ctx, src, dst, radius, add_src, rank, reflect):
    b, c, h, w = src.shape
    k = 2 * radius + 1
    rank = min(rank, k)
    tables = ctx.cached_constant(
        ("blur_lowrank", radius, rank), lambda: lowrank_blur_tables(radius, rank)
    )
    blocked = radius in LOWRANK_BLOCKED_RADII
    dims = {"height": h, "width": w, "reflect": reflect}
    if not blocked:
        dims["radius"] = radius
    suffix = f"_r{radius}" if blocked else ""
    out = LOWRANK_OUT if blocked else 1
    if blocked and ctx.backend != "cpu":  # groupshared streaming passes
        rows = (
            BLUR_LOWRANK_SHARED,
            f"lr_rows_shared_r{radius}",
            (-(-w // (32 * LOWRANK_OUT)) * 32, -(-h // 4) * 4),
        )
        cols = (
            BLUR_LOWRANK_SHARED,
            f"lr_cols_shared_r{radius}",
            (-(-w // 32) * 32, -(-h // (8 * LOWRANK_OUT)) * 8),
        )
    else:
        rows = (BLUR_LOWRANK, f"lr_rows{suffix}", (-(-w // out), h))
        cols = (BLUR_LOWRANK, f"lr_cols{suffix}", (w, -(-h // out)))
    tmp = ctx.empty(src.shape)
    for comp in range(rank):
        ctx.dispatch(
            rows[0],
            rows[1],
            (*rows[2], b * c),
            src=src,
            tmp=tmp,
            tables=tables,
            offset=comp * k,
            **dims,
        )
        mode = 2 if comp else (1 if add_src is not None else 0)
        ctx.dispatch(
            cols[0],
            cols[1],
            (*cols[2], b * c),
            tmp=tmp,
            add_src=add_src if add_src is not None else src,
            dst=dst,
            tables=tables,
            offset=(rank + comp) * k,
            mode=mode,
            **dims,
        )
    ctx.release(tmp)


def blur_level(ctx, src, dst, radius, add_src=None, blur="exact", impl=None, rank=1):
    """dst = add_src + blur_radius(src), reference padding rule.
    rank: components of the "lowrank" impl (2 * radius + 1 = full kernel)."""
    b, c, h, w = src.shape
    planes = b * c
    reflect = 1 if (h > radius and w > radius) else 0
    has_add = 1 if add_src is not None else 0
    add = add_src if add_src is not None else src
    impl = impl or default_blur_impl(ctx)
    if impl not in BLUR_IMPLS:
        raise ValueError(f"Unknown colour-fix blur impl: {impl}")
    if impl == "tiled" and ctx.backend == "cpu":
        raise ValueError("the tiled blur uses workgroup barriers: GPU backends only")
    if blur == "exact" and impl == "lowrank":
        _blur_lowrank(ctx, src, dst, radius, add_src, rank, reflect)
    elif blur == "exact":
        weights = blur_table(ctx, "exact", radius)
        if impl == "sym" and radius in SYM_RADII:
            ctx.dispatch(
                BLUR,
                f"blur2d_sym_r{radius}",
                (w, -(-h // 4), planes),
                src=src,
                add_src=add,
                dst=dst,
                weights=weights,
                height=h,
                width=w,
                reflect=reflect,
                has_add=has_add,
            )
        elif impl == "tiled" and radius in TILED_SHAPES:
            rows, ty = TILED_SHAPES[radius]
            ctx.dispatch(
                BLUR_TILED,
                f"blur2d_tiled_r{radius}",
                (-(-w // 32) * 32, -(-h // (rows * ty)) * ty, planes),
                src=src,
                add_src=add,
                dst=dst,
                weights=weights,
                height=h,
                width=w,
                reflect=reflect,
                has_add=has_add,
            )
        else:
            ctx.dispatch(
                BLUR,
                "blur2d",
                (w, h, planes),
                src=src,
                add_src=add,
                dst=dst,
                weights=weights,
                radius=radius,
                height=h,
                width=w,
                reflect=reflect,
                has_add=has_add,
            )
    elif blur == "separable":
        weights = blur_table(ctx, "separable", radius)
        tmp = ctx.empty(src.shape)
        ctx.dispatch(
            BLUR,
            "blur_rows",
            (w, h, planes),
            src=src,
            dst=tmp,
            weights=weights,
            radius=radius,
            height=h,
            width=w,
            reflect=reflect,
        )
        ctx.dispatch(
            BLUR,
            "blur_cols",
            (w, h, planes),
            src=tmp,
            add_src=add,
            dst=dst,
            weights=weights,
            radius=radius,
            height=h,
            width=w,
            reflect=reflect,
            has_add=has_add,
        )
        ctx.release(tmp)
    else:
        raise ValueError(f"Unknown colour-fix blur: {blur}")


def match_color(ctx, src, tgt, level=WAVELET_LEVELS, blur="exact", impl=None, rank=1):
    """Reference match_color(src, tgt) -> new [B,3,H,W] array."""
    if level < 1:
        raise ValueError("match_color needs at least one wavelet level")
    b, _, h, w = src.shape
    stats = ctx.empty((4,))
    lab_moments_pair(ctx, src, tgt, stats)
    inp = ctx.empty(src.shape)
    diff = ctx.empty(src.shape)
    ctx.dispatch_flat(
        MATCH,
        "match_apply",
        b * h * w,
        src=src,
        tgt=tgt,
        stats=stats,
        inp=inp,
        diff=diff,
        pixels=b * h * w,
        hw=h * w,
    )
    ctx.release(stats)
    out = ctx.empty(src.shape)
    cur = diff
    for i in range(1, level + 1):
        radius = 2**i
        last = i == level
        nxt = out if last else ctx.empty(src.shape)
        blur_level(
            ctx,
            cur,
            nxt,
            radius,
            add_src=inp if last else None,
            blur=blur,
            impl=impl,
            rank=rank,
        )
        ctx.release(cur)
        cur = nxt
    ctx.release(inp)
    return out
