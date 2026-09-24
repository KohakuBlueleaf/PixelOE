import pytest
import torch
import torch.nn.functional as F
from slang_helpers import all_backends, compare, dev, host, photo, shared_context

from pixeloe.slang.ops.downscale import contrast_downscale, k_centroid_downscale
from pixeloe.slang.ops.resample import (
    interpolate,
    lanczos_resize,
    pad_replicate,
    upscale_nearest_exact,
)
from pixeloe.slang.ops.sharpen import sharpen
from pixeloe.torch.downscale.contrast_based import contrast_downscale as ref_contrast
from pixeloe.torch.downscale.k_centroid import (
    batched_kmeans,
    k_centroid_downscale_torch,
    k_centroid_preprocess,
)
from pixeloe.torch.downscale.lanczos import lanczos_resize as ref_lanczos
from pixeloe.torch.sharpen.laplacian import laplacian_sharpen
from pixeloe.torch.sharpen.unsharp import unsharp_mask
from pixeloe.torch.utils import batched_kmeans_iter


def _run(ctx, fn, img, *args):
    arr = dev(ctx, img)
    out = host(ctx, fn(ctx, arr, *args), img)
    ctx.release(arr)
    return out


def assert_mostly_exact(stats, tol=1e-4, frac=2e-3):
    """Block selectors choose one of several candidate values; a float
    rounding difference in the block mean can flip a near-tie choice."""
    assert stats["finite"]
    assert stats["frac_gt_1e-4"] <= frac, stats


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("p", [2, 3, 4, 8, 16])
def test_contrast_downscale(backend, p):
    ctx = shared_context(backend)
    img = photo(p * 24, p * 31, batch=2)
    stats = compare(_run(ctx, contrast_downscale, img, p), ref_contrast(img, p))
    assert_mostly_exact(stats)


def reference_k_centroids(img, p):
    """The reference loop (batched_kmeans) re-run to also expose the last
    partition: returns final (c0, c1) per block and the exact-tie mask
    (equal partition sizes, where both SSEs are equal in exact arithmetic)."""
    b, c, h, w = img.shape
    patches, _ = k_centroid_preprocess(img, b, c, h, w, h // p, w // p)
    maxv = patches.max(dim=1, keepdim=True).values
    minv = patches.min(dim=1, keepdim=True).values
    interp = torch.linspace(0, 1, 2, device=img.device)[None, :, None]
    cent = interp * minv + (1 - interp) * maxv
    data = patches.unsqueeze(2)
    for _ in range(4):
        labels = ((data - cent.unsqueeze(1)) ** 2).sum(-1).argmin(-1)
        cent, diff = batched_kmeans_iter(data, cent)
        if diff < 1 / 256:
            break
    n1 = labels.sum(1)
    tie = n1 * 2 == labels.shape[1]
    assert torch.allclose(cent, batched_kmeans(patches, 2))
    return cent, tie


def _blocks(t):
    return t.permute(0, 2, 3, 1).reshape(-1, 3)


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("p", [2, 4, 6, 8])
def test_k_centroid(backend, p):
    """On exact SSE ties (equal final cluster sizes) the kernel takes c0,
    where the reference's float sums pick arbitrarily. Elsewhere blocks match
    except where a near-equidistant pixel flips an assignment and the
    trajectory diverges: the reference's own CPU and CUDA runs diverge on a
    few such blocks, and the kernel may differ on no more than that."""
    ctx = shared_context(backend)
    img = photo(p * 20, p * 27, batch=2)
    got = _blocks(_run(ctx, k_centroid_downscale, img, p))
    want = _blocks(k_centroid_downscale_torch(img, p, 2))
    cent, tie = reference_k_centroids(img, p)
    mismatch = ((got - want).abs().amax(1) > 1e-5) & ~tie
    tie_err = (got[tie] - cent[tie, 0]).abs().amax(1)
    want_cpu = _blocks(k_centroid_downscale_torch(img.cpu(), p, 2)).to(want)
    ref_spread = (((want_cpu - want).abs().amax(1) > 1e-5) & ~tie).sum().item()
    assert mismatch.sum().item() <= max(2 * ref_spread, 3), ref_spread
    assert (tie_err > 1e-5).sum().item() <= max(2 * ref_spread, 3), ref_spread


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize(
    "mode", ["nearest", "nearest-exact", "bilinear", "bicubic", "area"]
)
@pytest.mark.parametrize("size", [(24, 31), (25, 32), (13, 50)])
def test_interpolate(backend, mode, size):
    ctx = shared_context(backend)
    img = photo(96, 124, batch=2)
    out = _run(ctx, interpolate, img, size, mode)
    ref = F.interpolate(img, size=size, mode=mode)
    assert compare(out, ref)["max"] < 1e-5


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("size", [(24, 31), (25, 32), (96, 40)])
def test_lanczos(backend, size):
    ctx = shared_context(backend)
    img = photo(96, 124, batch=2)
    out = _run(ctx, lanczos_resize, img, size)
    assert compare(out, ref_lanczos(img, size))["max"] < 1e-5


@pytest.mark.parametrize("backend", all_backends())
def test_pad_and_upscale(backend):
    ctx = shared_context(backend)
    img = photo(13, 17, batch=2)
    out = _run(ctx, pad_replicate, img, 1, 2, 0, 3)
    ref = F.pad(img, (0, 3, 1, 2), mode="replicate")
    assert compare(out, ref)["max"] == 0
    for factor in (2, 3, 7):
        up = _run(ctx, upscale_nearest_exact, img, factor)
        ref = F.interpolate(img, scale_factor=factor, mode="nearest-exact")
        assert compare(up, ref)["max"] == 0


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("mode", ["unsharp", "laplacian"])
def test_sharpen(backend, mode):
    ctx = shared_context(backend)
    img = photo(33, 47, batch=2)
    out = _run(ctx, sharpen, img, mode, 0.5)
    if mode == "unsharp":
        ref = unsharp_mask(img, kernel_size=3, sigma=1.0, amount=0.5)
    else:
        ref = laplacian_sharpen(img, amount=0.5)
    stats = compare(out, ref)
    # the unsharp threshold (|mask| < 0.1) is a hard switch: allow near-ties
    assert stats["frac_gt_1e-4"] <= 1e-3 and torch.isfinite(out).all(), stats
