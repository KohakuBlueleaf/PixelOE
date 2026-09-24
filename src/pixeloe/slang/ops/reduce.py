"""Segmented tree reductions (deterministic, fixed order)."""

import numpy as np

MODULE = "reduce/reduce"
GROUP_MODULE = "reduce/reduce_group"  # groupshared: GPU backends only
STOP = "reduce/stopdiff"
FAN = 64  # elements folded per thread per level
MOMENT_PIXELS = 32  # pixels (x3 Lab values) per thread on the first level
GROUP = 256  # reduce_group.slang workgroup size
GROUP_ITEMS = 16  # elements per thread on a GPU first level (sets the groups)
MAX_GROUPS = 1024  # first-level partials per segment (level 2: one workgroup)


def first_level_groups(length):
    return max(1, min(MAX_GROUPS, -(-length // (GROUP * GROUP_ITEMS))))


def _seg_minmax_group(ctx, src, seg_count, seg_len):
    groups = first_level_groups(seg_len)
    part_min = ctx.empty((seg_count * groups,))
    part_max = ctx.empty((seg_count * groups,))
    ctx.dispatch(
        GROUP_MODULE,
        "seg_minmax_group",
        (groups * GROUP, seg_count),
        src_min=src,
        src_max=src,
        part_min=part_min,
        part_max=part_max,
        seg_len=seg_len,
        groups=groups,
    )
    out_min, out_max = ctx.empty((seg_count,)), ctx.empty((seg_count,))
    ctx.dispatch(
        GROUP_MODULE,
        "seg_minmax_group",
        (GROUP, seg_count),
        src_min=part_min,
        src_max=part_max,
        part_min=out_min,
        part_max=out_max,
        seg_len=groups,
        groups=1,
    )
    ctx.release(part_min, part_max)
    return out_min, out_max


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
    if ctx.backend != "cpu" and src_offset == 0:
        return _seg_minmax_group(ctx, src, seg_count, seg_len)
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


def lab_moments_pair(ctx, img0, img1, stats):
    """stats = [mean, std] of Lab(img0), then of Lab(img1) (same shapes);
    std unbiased. GPU: both images in one two-level reduction."""
    if ctx.backend == "cpu":
        lab_moments(ctx, img0, stats, 0)
        lab_moments(ctx, img1, stats, 1)
        return
    b, _, h, w = img0.shape
    pixels = b * h * w
    groups = first_level_groups(pixels)
    part = ctx.empty((2 * groups * 3,))
    ctx.dispatch(
        GROUP_MODULE,
        "lab_moments_group",
        (groups * GROUP, 2),
        img0=img0,
        img1=img1,
        part=part,
        pixels=pixels,
        hw=h * w,
        groups=groups,
    )
    ctx.dispatch(
        GROUP_MODULE,
        "moments_final_group",
        (GROUP, 2),
        part=part,
        stats=stats,
        count=groups,
    )
    ctx.release(part)


def lab_moments(ctx, img, stats, slot):
    """stats[2 slot] = mean, stats[2 slot + 1] = unbiased std of Lab(img)
    (portable multi-level reduction)."""
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
    while True:  # at least one merge level: the last one writes stats
        merged = -(-count // FAN)
        nxt = ctx.empty((merged * 3,))
        ctx.dispatch(
            MODULE,
            "moments_merge",
            (merged,),
            src=part,
            part=nxt,
            stats=stats,
            count=count,
            chunk=FAN,
            chunks=merged,
            slot=slot,
        )
        ctx.release(part)
        part, count = nxt, merged
        if count == 1:
            break
    ctx.release(part)
