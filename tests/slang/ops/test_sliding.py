"""Sliding-window outline statistics vs an independent torch reference
(F.unfold with stride 1: one window per pixel, then median/min/max)."""

import pytest
import torch
import torch.nn.functional as F
from slang_helpers import all_backends, compare, dev, host, photo, shared_context

from pixeloe.slang.ops.outline import expansion_weight, normalized_weight
from pixeloe.torch.lab import rgb_to_lab
from pixeloe.torch.outline import contrast_ratio_weight_mapping, current_weight_mapping


def window_stat(lum, k, stat, padding):
    """[B,H,W] -> per-pixel stat of the k x k window at [y - k/2, y - k/2 + k)."""
    x = lum[:, None]
    lo, hi = k // 2, k - 1 - k // 2
    if padding == "replicate":
        x = F.pad(x, (lo, hi, lo, hi), mode="replicate")
    else:
        x = F.pad(x, (lo, hi, lo, hi))
    patches = F.unfold(x, k)  # [B, k*k, H*W]
    if stat == "median":
        v = patches.median(dim=1).values
    elif stat == "min":
        v = patches.min(dim=1).values
    else:
        v = patches.max(dim=1).values
    return v.reshape(lum.shape)


def reference_weight(img, k, mapping, padding):
    lum = rgb_to_lab(img)[:, 0] / 100
    med = window_stat(lum, 2 * k, "median", padding)
    mn = window_stat(lum, k, "min", padding)
    mx = window_stat(lum, k, "max", padding)
    bright, dark = mx - med, med - mn
    fn = (
        contrast_ratio_weight_mapping
        if mapping == "contrast_ratio"
        else current_weight_mapping
    )
    return fn(med, mn, mx, bright, dark, bright + dark, 10, 3, "global")


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("k", [2, 3, 4, 5])
@pytest.mark.parametrize("padding", ["zero", "replicate"])
@pytest.mark.parametrize("mapping", ["current", "contrast_ratio"])
def test_sliding_weight(backend, k, padding, mapping):
    ctx = shared_context(backend)
    img = photo(37, 45, batch=2)
    arr = dev(ctx, img)
    rw = expansion_weight(
        ctx,
        arr,
        k,
        max(k // 2, 1),
        mapping=mapping,
        local_stats="sliding",
        stat_padding=padding,
    )
    out = host(ctx, normalized_weight(ctx, rw), img)
    ctx.release(arr)
    ref = reference_weight(img, k, mapping, padding)
    assert compare(out, ref)["max"] < 5e-5


@pytest.mark.parametrize("backend", all_backends()[:1])
def test_sliding_differs_from_lattice(backend):
    """The knob is live: sliding and lattice statistics are different maps."""
    ctx = shared_context(backend)
    img = photo(48, 64)
    arr = dev(ctx, img)
    a = host(ctx, normalized_weight(ctx, expansion_weight(ctx, arr, 4, 2)), img)
    b = host(
        ctx,
        normalized_weight(ctx, expansion_weight(ctx, arr, 4, 2, local_stats="sliding")),
        img,
    )
    ctx.release(arr)
    assert compare(a, b)["mean"] > 1e-3
    assert torch.isfinite(b).all()
