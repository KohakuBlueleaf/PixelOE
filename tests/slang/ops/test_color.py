import numpy as np
import pytest
import torch
from slang_helpers import all_backends, compare, dev, host, photo, shared_context

from pixeloe.slang.ops.color import (
    WAVELET_LEVELS,
    exact_blur_table,
    lab_image,
    lowrank_blur_tables,
    luminance,
    match_color,
)
from pixeloe.torch.color import match_color as ref_match_color
from pixeloe.torch.lab import rgb_to_lab


@pytest.mark.parametrize("backend", all_backends())
def test_lab_matches_kornia(backend):
    ctx = shared_context(backend)
    img = photo(37, 53, batch=2)
    arr = dev(ctx, img)
    lab = host(ctx, lab_image(ctx, arr), img)
    lum = host(ctx, luminance(ctx, arr), img)
    ctx.release(arr)
    ref = rgb_to_lab(img)
    # Lab channels span ~[-100, 100]; pow/cbrt differ by a few ulp per API
    assert compare(lab, ref)["max"] < 2e-3
    assert compare(lum, ref[:, 0] / 100)["max"] < 2e-5


def _match(ctx, src, tgt, **kw):
    s, t = dev(ctx, src), dev(ctx, tgt)
    out = host(ctx, match_color(ctx, s, t, **kw), tgt)
    ctx.release(s, t)
    return out


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("shape", [(64, 96), (40, 33), (20, 70)])
def test_match_color_exact(backend, shape):
    ctx = shared_context(backend)
    tgt = photo(*shape, batch=2)
    src = (tgt * 0.7 + 0.2 * torch.roll(tgt, 5, dims=-1)).clamp(0, 1)
    stats = compare(_match(ctx, src, tgt), ref_match_color(src, tgt))
    assert stats["finite"]
    assert stats["max"] < 1e-4, stats


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("impl", ["tiled", "sym", "direct", "lowrank"])
def test_match_color_blur_impls(backend, impl):
    if backend == "cpu" and impl == "tiled":
        pytest.skip("workgroup-barrier kernel: GPU backends only")
    ctx = shared_context(backend)
    tgt = photo(70, 90)
    src = (tgt * 0.8 + 0.1).clamp(0, 1)
    # lowrank at full rank (clipped to 2 r + 1 per level) is the exact kernel
    kw = {"impl": impl, "rank": 65} if impl == "lowrank" else {"impl": impl}
    stats = compare(_match(ctx, src, tgt, **kw), ref_match_color(src, tgt))
    assert stats["max"] < 1e-4, stats


def lowrank_error_bound(rank):
    """Worst-case match_color deviation of a rank-`rank` blur: per level the
    l1 norm of the kernel's SVD residual times the level's input bound (the
    colour difference, |d| <= 1), summed over the chained levels."""
    total = 0.0
    for level in range(1, WAVELET_LEVELS + 1):
        r = 2**level
        k = exact_blur_table(r).astype(np.float64).reshape(2 * r + 1, 2 * r + 1)
        tables = lowrank_blur_tables(r, rank).astype(np.float64)
        n = min(rank, 2 * r + 1) * (2 * r + 1)
        rows = tables[:n].reshape(-1, 2 * r + 1)
        cols = tables[n:].reshape(-1, 2 * r + 1)
        total += np.abs(k - cols.T @ rows).sum()
    return total


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("rank", [1, 2])
def test_match_color_lowrank_within_svd_bound(backend, rank):
    ctx = shared_context(backend)
    tgt = photo(96, 128, batch=2)
    src = (tgt * 0.6 + 0.3 * tgt.flip(-1)).clamp(0, 1)
    out = _match(ctx, src, tgt, impl="lowrank", rank=rank)
    stats = compare(out, ref_match_color(src, tgt))
    bound = lowrank_error_bound(rank)
    assert bound < 1 / 255
    # 1e-4: the exact implementations' own tolerance against the reference
    assert stats["max"] < bound + 1e-4, (stats, bound)


@pytest.mark.parametrize("backend", all_backends())
def test_match_color_separable_deviation_is_bounded(backend):
    ctx = shared_context(backend)
    tgt = photo(96, 128)
    src = (tgt * 0.6 + 0.3 * tgt.flip(-1)).clamp(0, 1)
    out = _match(ctx, src, tgt, blur="separable")
    stats = compare(out, ref_match_color(src, tgt))
    # the separable kernel differs from the float16 reference kernel only by
    # float16 rounding of the taps: far below one 8-bit level
    assert stats["max"] < 1 / 255, stats
