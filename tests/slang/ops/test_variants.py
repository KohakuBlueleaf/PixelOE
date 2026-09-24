"""Specialised kernels against the generic kernels they replace, on the same
device input: register-resident selection, fused morphology, blocked and
shared-memory low-rank blur passes."""

import numpy as np
import pytest
from slang_helpers import all_backends, compare, dev, host, photo, shared_context

import pixeloe.slang.ops.color as color
import pixeloe.slang.ops.outline as outline
import pixeloe.slang.ops.quant as quant
from pixeloe.slang.ops import downscale


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("p", downscale.CONTRAST_REGISTER_SIZES)
def test_contrast_register_entry_matches_generic(backend, p):
    ctx = shared_context(backend)
    img = photo(p * 13, p * 17, batch=2)
    arr = dev(ctx, img)
    lab = color.lab_image(ctx, arr)
    shape = (2, 3, 13, 17)
    dims = {"height": p * 13, "width": p * 17, "out_h": 13, "out_w": 17}
    outs = []
    for entry, extra in (
        ("contrast_downscale", {"p": p}),
        (f"contrast_downscale_p{p}", {"up": 0}),
    ):
        dst = ctx.empty(shape)
        ctx.dispatch(
            downscale.CONTRAST,
            entry,
            (17, 13, 2),
            lab_img=lab,
            dst=dst,
            **dims,
            **extra,
        )
        outs.append(host(ctx, dst, img))
    ctx.release(arr, lab)
    assert compare(outs[0], outs[1])["max"] == 0.0


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("k", outline.REGISTER_MEDIAN_SIZES)
def test_lattice_median_register_entry_matches_generic(backend, k):
    ctx = shared_context(backend)
    img = photo(45, 61, batch=2)
    arr = dev(ctx, img)
    lum = color.luminance(ctx, arr)
    stride, pad = max(k // 4, 1), k // 2
    a, _, _ = outline._lattice(
        ctx, lum, "lattice_median", k, stride, pad, ("out_stat",)
    )
    b, _, _ = outline._lattice(
        ctx, lum, f"lattice_median_k{k}", k, stride, pad, ("out_stat",)
    )
    ref = ctx.download(a["out_stat"])
    got = ctx.download(b["out_stat"])
    ctx.release(arr, lum, a["out_stat"], b["out_stat"])
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("backend", [b for b in all_backends() if b != "cpu"])
@pytest.mark.parametrize("thickness", [1, 2, 3, 4, 6])
def test_fused_morph_matches_pass_chain(backend, thickness, monkeypatch):
    ctx = shared_context(backend)
    img = photo(70, 83, batch=2)
    arr = dev(ctx, img)
    fused = outline.outline_expansion(ctx, arr, thickness, thickness, 4)
    monkeypatch.setattr(outline, "morph_fused_fits", lambda *_: False)
    chain = outline.outline_expansion(ctx, arr, thickness, thickness, 4)
    got = [host(ctx, a, img) for a in fused]
    ref = [host(ctx, a, img) for a in chain]
    ctx.release(arr)
    # a few ulp: DXC contracts v - k + 1 differently in the fused kernel
    assert compare(got[0], ref[0])["max"] <= 1e-6
    assert compare(got[1], ref[1])["max"] <= 1e-6


@pytest.mark.parametrize("backend", [b for b in all_backends() if b != "cpu"])
@pytest.mark.parametrize("shape", [(2, 5), (3, 7), (41, 57), (64, 96), (27, 480)])
def test_error_diffusion_shared_matches_global(backend, shape, monkeypatch):
    ctx = shared_context(backend)
    img = photo(*shape, batch=2)
    arr = dev(ctx, img)
    palette = quant.kmeans(ctx, arr, 16)[1]
    shared = host(ctx, quant.dither(ctx, arr, None, palette, "error_diffusion"), img)
    monkeypatch.setattr(quant, "ED_SHARED_CAP", 0)
    ref = host(ctx, quant.dither(ctx, arr, None, palette, "error_diffusion"), img)
    ctx.release(arr, palette)
    assert compare(shared, ref)["max"] == 0.0


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("rank", [1, 3])
def test_lowrank_specialised_passes_match_generic(backend, rank, monkeypatch):
    ctx = shared_context(backend)
    tgt = photo(67, 131, batch=2)
    src = (tgt * 0.6 + 0.3 * tgt.flip(-1)).clamp(0, 1)
    s, t = dev(ctx, src), dev(ctx, tgt)
    fast = host(ctx, color.match_color(ctx, s, t, impl="lowrank", rank=rank), tgt)
    monkeypatch.setattr(color, "LOWRANK_BLOCKED_RADII", ())
    generic = host(ctx, color.match_color(ctx, s, t, impl="lowrank", rank=rank), tgt)
    ctx.release(s, t)
    assert compare(fast, generic)["max"] < 1e-6
