"""k-means palette quantisation (plain / weighted / repeat) and dithering."""

from functools import cache

import numpy as np
import torch

from ...torch.color import generate_bayer_matrix
from .reduce import seg_minmax, seg_sum

WEIGHTS = "quant/weights"
KMEANS = "quant/kmeans"
KMEANS_GROUP = "quant/kmeans_group"  # groupshared: GPU backends only
REPEAT = "quant/repeat"
DITHER = "dither/dither"
ED_GROUP = "dither/ed_group"  # workgroup barriers: GPU backends only

KM_CHUNK = 256  # pixels per (chunk, cluster) thread in km_partial (CPU)
KM_CHUNK_GPU = 64  # kmeans_group.slang CHUNK: pixels per km_step_group workgroup
RP_CHUNK = 128  # remainders per thread in the repeat-table selection
RADIX_SHIFTS = (28, 24, 20, 16, 12, 8, 4, 0)


def quant_weights(ctx, weight, out_size, gamma):
    """(|2w-1| w) bilinear-resized to out_size, then ** gamma -> [B, h*w]."""
    b, h, w = weight.shape
    out_h, out_w = out_size
    gamma_mode = {0.5: 1, 1.0: 2, 2.0: 3}.get(float(gamma), 0)
    dst = ctx.empty((b, out_h * out_w))
    ctx.dispatch(
        WEIGHTS,
        "quant_weights",
        (out_w, out_h, b),
        w_full=weight,
        w_out=dst,
        in_h=h,
        in_w=w,
        out_h=out_h,
        out_w=out_w,
        scale_h=np.float32(h) / np.float32(out_h),
        scale_w=np.float32(w) / np.float32(out_w),
        gamma=np.float32(gamma),
        gamma_mode=gamma_mode,
    )
    return dst


@cache
def init_table(num_centroids, c=3):
    """interp[k, c] such that centroid = interp * min + (1 - interp) * max,
    matching centroid_generator (built with torch float32 linspace)."""
    if num_centroids < 8:
        interp = torch.linspace(0, 1, num_centroids)[:, None].expand(-1, c)
    else:
        base_num = num_centroids // 4
        cent_num = num_centroids - base_num * 3
        interp_base = torch.linspace(0, 1, base_num + 1)[1:, None, None]
        interp_base = (interp_base * torch.eye(c)).reshape(-1, c)
        interp_cent = torch.linspace(0, 1, cent_num)[:, None].expand(-1, c)
        interp = torch.cat([interp_base, interp_cent], dim=0)
    return interp.contiguous().numpy().astype(np.float32)


def repeat_table(ctx, weights, count):
    """generate_repeat_table(weights, N, 4N) -> uint32 [B, N]."""
    b = weights.shape[0]
    total = b * count
    extra = np.float32(4 * count - count)
    lw = ctx.empty((b, count))
    ctx.dispatch(REPEAT, "rp_log", (total,), w=weights, lw=lw, total=total)
    _, lmax = seg_minmax(ctx, lw, b, count)
    e = ctx.empty((b, count))
    ctx.dispatch(
        REPEAT,
        "rp_shift_exp",
        (total,),
        lw=lw,
        lmax=lmax,
        e=e,
        count=count,
        total=total,
    )
    sumexp = seg_sum(ctx, e, b, count)
    repeat = ctx.empty((b, count), "uint32")
    rem = ctx.empty((b, count))
    fl = e  # reuse
    ctx.dispatch(
        REPEAT,
        "rp_counts",
        (total,),
        lw=lw,
        lmax=lmax,
        sumexp=sumexp,
        repeat=repeat,
        rem=rem,
        fl_out=fl,
        count=count,
        total=total,
        extra=extra,
    )
    fl_sum = seg_sum(ctx, fl, b, count)
    ctx.release(lw, lmax, e, sumexp)

    state = ctx.empty((b, 4), "uint32")
    ctx.dispatch(
        REPEAT,
        "rp_select_init",
        (b,),
        fl_sum=fl_sum,
        state=state,
        count=count,
        batch=b,
        extra=extra,
    )
    ctx.release(fl_sum)
    chunks = -(-count // RP_CHUNK)
    hist = ctx.empty((b * chunks, 16), "uint32")
    sel = {"count": count, "batch": b, "chunk": RP_CHUNK, "chunks": chunks}
    for shift in RADIX_SHIFTS:
        ctx.dispatch(
            REPEAT,
            "rp_hist",
            (b * chunks,),
            rem=rem,
            state=state,
            hist=hist,
            shift=shift,
            **sel,
        )
        ctx.dispatch(
            REPEAT,
            "rp_select_step",
            (b,),
            hist=hist,
            state=state,
            shift=shift,
            batch=b,
            chunks=chunks,
        )
    ctx.release(hist)
    chunk_gt = ctx.empty((b * chunks,), "uint32")
    chunk_eq = ctx.empty((b * chunks,), "uint32")
    ctx.dispatch(
        REPEAT,
        "rp_tie_count",
        (b * chunks,),
        rem=rem,
        state=state,
        chunk_gt=chunk_gt,
        chunk_eq=chunk_eq,
        **sel,
    )
    ctx.dispatch(
        REPEAT,
        "rp_tie_scan",
        (b,),
        chunk_gt=chunk_gt,
        chunk_eq=chunk_eq,
        state=state,
        batch=b,
        chunks=chunks,
    )
    ctx.dispatch(
        REPEAT,
        "rp_mark",
        (b * chunks,),
        rem=rem,
        state=state,
        tie_prefix=chunk_eq,
        repeat=repeat,
        **sel,
    )
    ctx.release(rem, state, chunk_gt, chunk_eq)
    return repeat


def kmeans(ctx, img, num_centroids, weights=None, repeat_mode=False):
    """color_quantization_kmeans -> (quantised [B,3,h,w], palette [B,K,3])."""
    b, c, h, w = img.shape
    count = h * w
    K = num_centroids
    ch_min, ch_max = seg_minmax(ctx, img, b * c, count)
    interp = ctx.cached_constant(("km_interp", K), lambda: init_table(K))
    cent = ctx.empty((b, K, 3))
    ctx.dispatch(
        KMEANS,
        "km_init",
        (b * K,),
        interp=interp,
        ch_min=ch_min,
        ch_max=ch_max,
        cent=cent,
        K=K,
        batch=b,
    )
    ctx.release(ch_min, ch_max)

    repeat = None
    weighted = 0
    if weights is not None and repeat_mode:
        repeat = repeat_table(ctx, weights, count)
    elif weights is not None:
        weighted = 1
    dummy_f = weights if weights is not None else cent
    dummy_u = repeat if repeat is not None else ctx.empty((1,), "uint32")

    iters = 2 * int(K**0.5)
    labels = ctx.empty((b, count), "uint32")
    run = ctx.empty((max(iters, 1),), "uint32")
    ctx.clear_uint(run)
    item_diff = ctx.empty((b * K,))
    chunk = KM_CHUNK if ctx.backend == "cpu" else KM_CHUNK_GPU
    chunks = -(-count // chunk)
    part = ctx.empty((b, chunks, K, 4), "int64")
    use_repeat = 1 if repeat is not None else 0
    for it in range(iters):
        if ctx.backend == "cpu":
            ctx.dispatch(
                KMEANS,
                "km_assign",
                (b * count,),
                pix=img,
                weight=dummy_f,
                cent=cent,
                labels=labels,
                item_diff=item_diff,
                run=run,
                it=it,
                K=K,
                count=count,
                batch=b,
                weighted=weighted,
            )
            ctx.dispatch(
                KMEANS,
                "km_partial",
                (b * chunks * K,),
                pix=img,
                repeat=dummy_u,
                labels=labels,
                part=part,
                run=run,
                it=it,
                K=K,
                count=count,
                batch=b,
                chunk=chunk,
                chunks=chunks,
                use_repeat=use_repeat,
            )
        else:  # assignment + member sums in one pass (km_step_group)
            ctx.dispatch(
                KMEANS_GROUP,
                "km_step_group",
                (b * chunks * KM_CHUNK_GPU,),
                pix=img,
                weight=dummy_f,
                cent=cent,
                repeat=dummy_u,
                part=part,
                item_diff=item_diff,
                run=run,
                it=it,
                K=K,
                count=count,
                batch=b,
                chunks=chunks,
                weighted=weighted,
                use_repeat=use_repeat,
            )
        if ctx.backend == "cpu":
            ctx.dispatch(
                KMEANS,
                "km_update",
                (b * K,),
                part=part,
                cent=cent,
                item_diff=item_diff,
                run=run,
                it=it,
                K=K,
                batch=b,
                chunks=chunks,
            )
        else:  # one workgroup per (b, k): exact integer sums, same result
            ctx.dispatch(
                KMEANS_GROUP,
                "km_update_group",
                (b * K * 256,),
                part=part,
                cent=cent,
                item_diff=item_diff,
                run=run,
                it=it,
                K=K,
                chunks=chunks,
            )
    quant = ctx.empty(img.shape)
    ctx.dispatch(
        KMEANS,
        "km_final",
        (b * count,),
        pix=img,
        cent=cent,
        labels=labels,
        quant=quant,
        K=K,
        count=count,
        batch=b,
    )
    ctx.release(labels, run, item_diff, part, dummy_u)
    return quant, cent


def _bayer():
    return generate_bayer_matrix(8, torch.device("cpu")).float().numpy()


def dither(ctx, img, quantized, palette, method, ed_impl=None):
    """parallel_dither_with_palette -> new array (None: use `quantized`).

    ed_impl: "group" (one GPU workgroup per image runs every step, one
    dispatch) or "steps" (two portable dispatches per row pair); default
    "group" on GPU backends, "steps" on CPU.
    """
    b, _, h, w = img.shape
    if ed_impl is None:
        ed_impl = "steps" if ctx.backend == "cpu" else "group"
    K = palette.shape[1]
    if method == "ordered":
        bayer = ctx.cached_constant(("bayer", 8), _bayer, np.float32)
        dst = ctx.empty(img.shape)
        ctx.dispatch(
            DITHER,
            "dither_ordered",
            (w, h, b),
            img=img,
            pal=palette,
            bayer=bayer,
            dst=dst,
            K=K,
            height=h,
            width=w,
        )
        return dst
    if method == "error_diffusion":
        work = ctx.empty(img.shape)
        ctx.copy(img, work)
        err = ctx.empty((b, 3, 3, w))
        if ed_impl == "group":
            ctx.dispatch(
                ED_GROUP,
                "ed_all_steps",
                (b * 256,),
                img=work,
                pal=palette,
                err=err,
                K=K,
                height=h,
                width=w,
            )
        elif ed_impl != "steps":
            raise ValueError(f"Unknown error-diffusion impl: {ed_impl}")
        steps = range(0, h - 2, 2) if ed_impl == "steps" else ()
        for y in steps:
            ctx.dispatch(
                DITHER,
                "ed_errors",
                (w, 3, b),
                img=work,
                pal=palette,
                err=err,
                y=y,
                K=K,
                height=h,
                width=w,
            )
            ctx.dispatch(
                DITHER,
                "ed_apply",
                (w, 2, b),
                img=work,
                err=err,
                y=y,
                height=h,
                width=w,
            )
        dst = ctx.empty(img.shape)
        ctx.dispatch(
            DITHER,
            "palette_map",
            (w, h, b),
            img=work,
            pal=palette,
            dst=dst,
            K=K,
            height=h,
            width=w,
        )
        ctx.release(work, err)
        return dst
    return None


def quantize_and_dither(
    ctx,
    image,
    weights=None,
    num_centroids=32,
    quant_mode="kmeans",
    dither_method="error_diffusion",
    repeat_mode=False,
):
    if quant_mode != "kmeans":
        raise ValueError(f"Invalid quantization mode: {quant_mode}")
    quantized, palette = kmeans(ctx, image, num_centroids, weights, repeat_mode)
    out = dither(ctx, image, quantized, palette, dither_method)
    ctx.release(palette)
    if out is None:
        return quantized
    ctx.release(quantized)
    return out
