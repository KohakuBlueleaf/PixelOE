"""Segmented tree reductions (deterministic, fixed order)."""

import numpy as np

MODULE = "reduce/reduce"
STOP = "reduce/stopdiff"
FAN = 64  # elements folded per thread per level
MOMENT_PIXELS = 32  # pixels (x3 Lab values) per thread on the first level


def _levels(length):
    """Chunk counts per level until one value per segment remains."""
    counts = []
    while True:
        length = -(-length // FAN)
        counts.append(length)
        if length == 1:
            return counts


def seg_minmax(ctx, src, seg_count, seg_len, src_offset=0):
    """Per-segment (min, max) of src; returns two [seg_count] arrays."""
    cur_min = cur_max = src
    length, offset = seg_len, src_offset
    owned = []
    for chunks in _levels(seg_len):
        part_min = ctx.empty((seg_count * chunks,))
        part_max = ctx.empty((seg_count * chunks,))
        ctx.dispatch(
            MODULE,
            "seg_minmax_level",
            (seg_count * chunks,),
            src_min=cur_min,
            src_max=cur_max,
            part_min=part_min,
            part_max=part_max,
            src_offset=offset,
            seg_count=seg_count,
            seg_len=length,
            chunk=FAN,
            chunks=chunks,
        )
        ctx.release(*owned)
        owned = [part_min, part_max]
        cur_min, cur_max, length, offset = part_min, part_max, chunks, 0
    return cur_min, cur_max


def seg_sum(ctx, src, seg_count, seg_len, scale=1.0, src_offset=0):
    """Per-segment sum(src) * scale; returns a [seg_count] array."""
    cur, length, offset = src, seg_len, src_offset
    levels = _levels(seg_len)
    owned = None
    for i, chunks in enumerate(levels):
        part = ctx.empty((seg_count * chunks,))
        ctx.dispatch(
            MODULE,
            "seg_sum_level",
            (seg_count * chunks,),
            src=cur,
            part=part,
            src_offset=offset,
            seg_count=seg_count,
            seg_len=length,
            chunk=FAN,
            chunks=chunks,
            scale=np.float32(scale if i == len(levels) - 1 else 1.0),
        )
        ctx.release(owned)
        owned = part
        cur, length, offset = part, chunks, 0
    return cur


def stop_diff(ctx, item_diff, items, diff, it):
    """diff[it] = max(item_diff[:items]) unless iteration `it` was stopped."""
    chunks = -(-items // 1024)
    part = ctx.empty((chunks,))
    ctx.dispatch(
        STOP,
        "diff_partial",
        (chunks,),
        item_diff=item_diff,
        part=part,
        diff=diff,
        it=it,
        items=items,
        chunk=1024,
        chunks=chunks,
    )
    ctx.dispatch(STOP, "diff_final", (1,), part=part, diff=diff, it=it, chunks=chunks)
    ctx.release(part)


def lab_moments(ctx, img, stats, slot):
    """stats[2 slot] = mean, stats[2 slot + 1] = unbiased std of Lab(img)."""
    b, _, h, w = img.shape
    pixels = b * h * w
    chunks = -(-pixels // MOMENT_PIXELS)
    part = ctx.empty((chunks * 3,))
    ctx.dispatch(
        MODULE,
        "lab_moments_partial",
        (chunks,),
        img=img,
        part=part,
        pixels=pixels,
        hw=h * w,
        chunk=MOMENT_PIXELS,
        chunks=chunks,
    )
    count = chunks
    while count > 1:
        merged = -(-count // FAN)
        nxt = ctx.empty((merged * 3,))
        ctx.dispatch(
            MODULE,
            "moments_merge",
            (merged,),
            src=part,
            part=nxt,
            count=count,
            chunk=FAN,
            chunks=merged,
        )
        ctx.release(part)
        part, count = nxt, merged
    ctx.dispatch(MODULE, "moments_finalize", (1,), part=part, stats=stats, slot=slot)
    ctx.release(part)
