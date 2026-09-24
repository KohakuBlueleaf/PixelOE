"""Contrast-aware outline expansion (lattice statistics, weight, morphology)."""

import numpy as np

from ...torch.minmax import KERNELS
from .color import luminance
from .reduce import seg_minmax, seg_sum

LATTICE = "outline/lattice"
WEIGHT = "outline/weight"
MORPH = "outline/morph"

SLIDING = "outline/sliding"

# window sizes with register-resident selection entry points
# (lattice_median_k<K>, sliding_median_k<K>); others use the radix select
REGISTER_MEDIAN_SIZES = (4, 6, 8, 10, 12)
# k (= pixel size) with a one-pass median + min/max entry (lattice_stats_h<k>);
# even k centres the min/max patch in the median patch on one shared lattice
FUSED_STATS_HALVES = (2, 4, 6)

MORPH_FUSED = "outline/morph_fused"  # groupshared: GPU backends only
MORPH_TILE = 32  # morph_fused.slang TILE (outputs per side, 32 x 8 threads)
MORPH_IMG_CAP = 3072  # morph_fused.slang IMG_CAP / BUF_CAP (floats)
MORPH_BUF_CAP = 2400
# (blend, open) element sizes with compile-time entries oe_morph_fused_b<B>o<O>
MORPH_FIXED_SIZES = ((3, 3), (5, 3), (5, 5), (7, 5))


def morph_fused_fits(ks_erode, ks_dilate, ks_open):
    ro = ks_open // 2
    rb = max(ks_erode, ks_dilate) // 2
    image_side = MORPH_TILE + 2 * (4 * ro + rb)
    buffer_side = MORPH_TILE + 8 * ro
    return image_side**2 <= MORPH_IMG_CAP and buffer_side**2 <= MORPH_BUF_CAP


MAPPINGS = {"current": 0, "polarity": 0, "contrast_ratio": 1, "contrast_gated": 2}
NORMALIZE = {"none": 0, "global": 1, "per_image": 2}
LOCAL_STATS = ("lattice", "sliding")
STAT_PADDING = {"zero": 0, "replicate": 1}


def structuring_element(ctx, iters):
    """KERNELS[iters] as a device constant; returns (array, size)."""
    table = KERNELS[iters].numpy().astype(np.float32)
    arr = ctx.cached_constant(("se", iters), lambda: table)
    return arr, table.shape[0]


def median_entry(prefix, ksize):
    """Register-resident entry for ksize when there is one, else the generic."""
    return f"{prefix}_k{ksize}" if ksize in REGISTER_MEDIAN_SIZES else prefix


def _lattice(ctx, lum, entry, ksize, stride, pad, outputs):
    b, h, w = lum.shape
    lat_h = (h + 2 * pad - ksize) // stride + 1
    lat_w = (w + 2 * pad - ksize) // stride + 1
    arrays = {name: ctx.empty((b, lat_h, lat_w)) for name in outputs}
    ctx.dispatch(
        LATTICE,
        entry,
        (lat_w, lat_h, b),
        lum=lum,
        height=h,
        width=w,
        ksize=ksize,
        stride=stride,
        pad=pad,
        lat_h=lat_h,
        lat_w=lat_w,
        **arrays,
    )
    return arrays, lat_h, lat_w


def _lattice_stats(ctx, img, k, stride):
    """(median lattice, min/max lattices, sizes) of the outline statistics:
    luminance, then one fused pass for k in FUSED_STATS_HALVES, else a
    median pass and a min/max pass."""
    b, _, h, w = img.shape
    lum = luminance(ctx, img)
    if k in FUSED_STATS_HALVES:
        lat_h = h // stride + 1
        lat_w = w // stride + 1
        med = {"out_stat": ctx.empty((b, lat_h, lat_w))}
        mm = {"out_min": ctx.empty((b, lat_h, lat_w))}
        mm["out_max"] = ctx.empty((b, lat_h, lat_w))
        ctx.dispatch(
            LATTICE,
            f"lattice_stats_h{k}",
            (lat_w, lat_h, b),
            lum=lum,
            out_med=med["out_stat"],
            out_min=mm["out_min"],
            out_max=mm["out_max"],
            height=h,
            width=w,
            stride=stride,
            lat_h=lat_h,
            lat_w=lat_w,
        )
        ctx.release(lum)
        return med, mm, (lat_h, lat_w), (lat_h, lat_w)
    med, med_h, med_w = _lattice(
        ctx,
        lum,
        median_entry("lattice_median", 2 * k),
        2 * k,
        stride,
        k,
        ("out_stat",),
    )
    mm, mm_h, mm_w = _lattice(
        ctx, lum, "lattice_minmax", k, stride, k // 2, ("out_min", "out_max")
    )
    ctx.release(lum)
    return med, mm, (med_h, med_w), (mm_h, mm_w)


def _lattice_weight(ctx, img, raw, lc, k, stride, mapping_id, scales):
    b, _, h, w = img.shape
    med, mm, (med_h, med_w), (mm_h, mm_w) = _lattice_stats(ctx, img, k, stride)
    ctx.dispatch(
        WEIGHT,
        "weight_raw",
        (w, h, b),
        med_stat=med["out_stat"],
        min_stat=mm["out_min"],
        max_stat=mm["out_max"],
        w_out=raw,
        lc_out=lc,
        height=h,
        width=w,
        med_lat_h=med_h,
        med_lat_w=med_w,
        med_ksize=2 * k,
        med_pad=k,
        mm_lat_h=mm_h,
        mm_lat_w=mm_w,
        mm_ksize=k,
        mm_pad=k // 2,
        stride=stride,
        mapping=mapping_id,
        **scales,
    )
    ctx.release(med["out_stat"], mm["out_min"], mm["out_max"])


def _sliding_weight(ctx, lum, raw, lc, k, pad_mode, mapping_id, scales):
    b, h, w = lum.shape
    grid = (w, h, b)
    dims = {"height": h, "width": w, "pad_mode": pad_mode}
    med = ctx.empty(lum.shape)
    entry = median_entry("sliding_median", 2 * k)
    extra = {} if entry != "sliding_median" else {"ksize": 2 * k}
    ctx.dispatch(SLIDING, entry, grid, lum=lum, out_stat=med, **dims, **extra)
    row_min, row_max = ctx.empty(lum.shape), ctx.empty(lum.shape)
    ctx.dispatch(
        SLIDING,
        "sliding_minmax_rows",
        grid,
        lum=lum,
        row_min=row_min,
        row_max=row_max,
        ksize=k,
        **dims,
    )
    mn, mx = ctx.empty(lum.shape), ctx.empty(lum.shape)
    ctx.dispatch(
        SLIDING,
        "sliding_minmax_cols",
        grid,
        row_min=row_min,
        row_max=row_max,
        out_min=mn,
        out_max=mx,
        ksize=k,
        **dims,
    )
    ctx.release(row_min, row_max)
    ctx.dispatch_flat(
        WEIGHT,
        "weight_dense",
        b * h * w,
        med=med,
        mn=mn,
        mx=mx,
        w_out=raw,
        lc_out=lc,
        count=b * h * w,
        mapping=mapping_id,
        **scales,
    )
    ctx.release(med, mn, mx)


class RawWeight:
    """Un-normalised outline weight plus what is needed to normalise it."""

    def __init__(self, ctx, raw, wmin, wmax, norm_mode):
        self.ctx = ctx
        self.raw = raw
        self.wmin = wmin
        self.wmax = wmax
        self.norm_mode = norm_mode

    def release(self):
        self.ctx.release(self.raw, self.wmin, self.wmax)


def expansion_weight(
    ctx,
    img,
    k=16,
    stride=4,
    avg_scale=10,
    dist_scale=3,
    mapping="current",
    normalize="global",
    local_stats="lattice",
    stat_padding="zero",
):
    """expansion_weight, returned un-normalised (see RawWeight).

    local_stats: "lattice" (reference: patches at `stride`, overlap-add
    averaged) or "sliding" (exact per-pixel windows; `stride` unused).
    stat_padding (sliding only): "zero" (as the lattice) or "replicate".
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"Unsupported outline weight mapping: {mapping}")
    if normalize not in NORMALIZE:
        raise ValueError(f"Unsupported weight normalization mode: {normalize}")
    if local_stats not in LOCAL_STATS:
        raise ValueError(f"Unsupported local statistics: {local_stats}")
    b, _, h, w = img.shape
    mapping_id = MAPPINGS[mapping]
    raw = ctx.empty((b, h, w))
    lc = ctx.empty((b, h, w)) if mapping_id == 2 else raw
    scales = {"avg_scale": np.float32(avg_scale), "dist_scale": np.float32(dist_scale)}
    if local_stats == "lattice":
        _lattice_weight(ctx, img, raw, lc, k, stride, mapping_id, scales)
    else:
        lum = luminance(ctx, img)
        _sliding_weight(
            ctx, lum, raw, lc, k, STAT_PADDING[stat_padding], mapping_id, scales
        )
        ctx.release(lum)

    if mapping_id == 2:
        total = b * h * w
        lc_mean = seg_sum(ctx, lc, 1, total, scale=1.0 / total)
        ctx.dispatch_flat(
            WEIGHT,
            "weight_gate",
            total,
            w=raw,
            lc=lc,
            lc_mean=lc_mean,
            count=total,
            dist_scale=np.float32(dist_scale),
        )
        ctx.release(lc, lc_mean)

    norm_mode = NORMALIZE[normalize]
    if norm_mode == 0:
        wmin = ctx.empty((1,))
        wmax = ctx.empty((1,))
    elif norm_mode == 1:
        wmin, wmax = seg_minmax(ctx, raw, 1, b * h * w)
    else:
        wmin, wmax = seg_minmax(ctx, raw, b, h * w)
    return RawWeight(ctx, raw, wmin, wmax, norm_mode)


def normalized_weight(ctx, rw):
    """Materialise the normalised weight map [B,H,W] (consumes rw)."""
    b, h, w = rw.raw.shape
    out = ctx.empty((b, h, w))
    ctx.dispatch_flat(
        MORPH,
        "normalize_map",
        b * h * w,
        w_raw=rw.raw,
        wmin=rw.wmin,
        wmax=rw.wmax,
        w_out=out,
        hw=h * w,
        count=b * h * w,
        norm_mode=rw.norm_mode,
    )
    rw.release()
    return out


def morph(ctx, src, iters, mode, do_clamp):
    """mode 'erode' / 'dilate' with KERNELS[iters] -> new array."""
    b, c, h, w = src.shape
    se, ks = structuring_element(ctx, iters)
    dst = ctx.empty(src.shape)
    ctx.dispatch(
        MORPH,
        "morph",
        (w, h, b * c),
        src=src,
        dst=dst,
        se=se,
        ks=ks,
        height=h,
        width=w,
        mode=0 if mode == "erode" else 1,
        do_clamp=1 if do_clamp else 0,
    )
    return dst


def outline_expansion(
    ctx,
    img,
    erode_iters=2,
    dilate_iters=2,
    k=16,
    avg_scale=10,
    dist_scale=3,
    weight_mapping="current",
    weight_normalize="global",
    local_stats="lattice",
    stat_padding="zero",
):
    """outline_expansion -> (expanded [B,3,H,W], weight [B,H,W])."""
    b, c, h, w = img.shape
    rw = expansion_weight(
        ctx,
        img,
        k,
        k // 2,
        avg_scale,
        dist_scale,
        mapping=weight_mapping,
        normalize=weight_normalize,
        local_stats=local_stats,
        stat_padding=stat_padding,
    )
    se_e, ks_e = structuring_element(ctx, erode_iters)
    se_d, ks_d = structuring_element(ctx, dilate_iters)
    oc_iter = max(erode_iters - 1, dilate_iters - 1, 1)
    weight = ctx.empty((b, h, w))
    se_o, ks_o = structuring_element(ctx, oc_iter)
    if ctx.backend != "cpu" and morph_fused_fits(ks_e, ks_d, ks_o):
        out = ctx.empty(img.shape)
        sizes = {"ks_erode": ks_e, "ks_dilate": ks_d, "ks_open": ks_o}
        entry = "oe_morph_fused"
        if ks_e == ks_d and (ks_e, ks_o) in MORPH_FIXED_SIZES:
            entry, sizes = f"oe_morph_fused_b{ks_e}o{ks_o}", {}
        ctx.dispatch(
            MORPH_FUSED,
            entry,
            (-(-w // MORPH_TILE) * 32, -(-h // MORPH_TILE) * 8, b * c),
            img=img,
            w_raw=rw.raw,
            wmin=rw.wmin,
            wmax=rw.wmax,
            se_erode=se_e,
            se_dilate=se_d,
            se_open=se_o,
            dst=out,
            w_out=weight,
            height=h,
            width=w,
            norm_mode=rw.norm_mode,
            **sizes,
        )
        rw.release()
        return out, weight
    blended = ctx.empty(img.shape)
    ctx.dispatch(
        MORPH,
        "oe_blend",
        (w, h, b * c),
        img=img,
        w_raw=rw.raw,
        wmin=rw.wmin,
        wmax=rw.wmax,
        se_erode=se_e,
        se_dilate=se_d,
        dst=blended,
        w_out=weight,
        ks_erode=ks_e,
        ks_dilate=ks_d,
        height=h,
        width=w,
        norm_mode=rw.norm_mode,
    )
    rw.release()

    steps = (("erode", False), ("dilate", False), ("dilate", True), ("erode", False))
    cur = blended
    for mode, do_clamp in steps:
        nxt = morph(ctx, cur, oc_iter, mode, do_clamp)
        ctx.release(cur)
        cur = nxt
    return cur, weight
