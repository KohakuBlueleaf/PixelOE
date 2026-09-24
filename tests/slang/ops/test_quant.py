import pytest
import torch
from slang_helpers import all_backends, compare, dev, host, photo, shared_context

from pixeloe.slang.ops.quant import dither, kmeans, quant_weights, repeat_table
from pixeloe.torch.color import (
    color_quantization_kmeans,
    parallel_dither_with_palette,
)
from pixeloe.torch.outline import expansion_weight as ref_expansion_weight
from pixeloe.torch.utils import generate_repeat_table


def _weights(img, out_size, gamma):
    w = ref_expansion_weight(img, 4, 2)
    w = torch.abs(w * 2 - 1) * w
    w = torch.nn.functional.interpolate(w, size=out_size, mode="bilinear")
    return w**gamma


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("gamma", [0.5, 0.37, 1.0])
def test_quant_weights(backend, gamma):
    ctx = shared_context(backend)
    img = photo(64, 80, batch=2)
    full = ref_expansion_weight(img, 4, 2)
    arr = dev(ctx, full[:, 0].contiguous())
    out = host(ctx, quant_weights(ctx, arr, (16, 20), gamma), img)
    ctx.release(arr)
    ref = _weights(img, (16, 20), gamma).reshape(2, -1)
    assert compare(out, ref)["max"] < 1e-5


@pytest.mark.parametrize("backend", all_backends())
def test_repeat_table(backend):
    ctx = shared_context(backend)
    img = photo(128, 160, batch=2)
    w = _weights(img, (32, 40), 0.5).reshape(2, -1).contiguous()
    arr = dev(ctx, w)
    got = host(ctx, repeat_table(ctx, arr, w.shape[1]), w).long()
    ctx.release(arr)
    ref = generate_repeat_table(w.float(), w.shape[1], w.shape[1] * 4)
    assert torch.equal(got.sum(1).cpu(), ref.sum(1).cpu())
    # only tie-breaking among equal remainders may differ
    assert (got.cpu() != ref.cpu()).float().mean() < 5e-3


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("K", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("variant", ["plain", "weighted", "repeat"])
def test_kmeans(backend, K, variant):
    ctx = shared_context(backend)
    img = photo(48, 64, batch=2)
    weights = None
    if variant != "plain":
        weights = _weights(photo(192, 256, batch=2), (48, 64), 0.5).reshape(2, -1)
    arr = dev(ctx, img)
    warr = dev(ctx, weights.contiguous()) if weights is not None else None
    quant, pal = kmeans(ctx, arr, K, warr, repeat_mode=variant == "repeat")
    quant_t = host(ctx, quant, img)
    pal_t = host(ctx, pal, img)
    ctx.release(arr, warr)
    kw = {"weights": weights, "repeat_mode": variant == "repeat"}
    ref_q, ref_pal, _ = color_quantization_kmeans(img, K, **kw)
    cpu_kw = dict(kw, weights=weights.cpu() if weights is not None else None)
    cpu_q, cpu_pal, _ = color_quantization_kmeans(img.cpu(), K, **cpu_kw)
    pal_stats = compare(pal_t, ref_pal)
    q_stats = compare(quant_t, ref_q)
    # iterated means drift with float summation order; the reference drifts
    # the same way between its CPU and CUDA runs
    pal_spread = compare(ref_pal, cpu_pal)
    q_spread = compare(ref_q, cpu_q)
    pal_limit = max(1e-4, 4 * pal_spread["max"])
    q_limit = max(1e-3, 4 * q_spread["frac_gt_1e-4"])
    if variant == "repeat":
        # repeat counts may also differ on remainder ties (see test_repeat_table)
        pal_limit = max(pal_limit, 2e-2)
        q_limit = max(q_limit, 2e-2)
    assert pal_stats["max"] < pal_limit, (pal_stats, pal_spread)
    assert q_stats["frac_gt_1e-4"] < q_limit, (q_stats, q_spread)


DITHER_CASES = [
    ("ordered", None),
    ("error_diffusion", "steps"),
    ("error_diffusion", "group"),
]


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("method,ed_impl", DITHER_CASES)
@pytest.mark.parametrize("K", [4, 32])
def test_dither(backend, method, ed_impl, K):
    if backend == "cpu" and ed_impl == "group":
        pytest.skip("workgroup-barrier kernel: GPU backends only")
    ctx = shared_context(backend)
    img = photo(40, 52, batch=2)
    ref_q, ref_pal, _ = color_quantization_kmeans(img, K)
    arr = dev(ctx, img)
    pal = dev(ctx, ref_pal.contiguous())
    q = dev(ctx, ref_q.contiguous())
    out = host(ctx, dither(ctx, arr, q, pal, method, ed_impl=ed_impl), img)
    ctx.release(arr, pal, q)
    ref = parallel_dither_with_palette(img, ref_q, ref_pal, method=method)
    stats = compare(out, ref)
    assert stats["frac_gt_1e-4"] < 2e-3, stats
