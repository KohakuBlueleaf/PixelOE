import itertools
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from slang_helpers import all_backends, compare, photo, shared_context

from pixeloe.slang.pixelize import pixelize
from pixeloe.torch.pixelize import pixelize as ref_pixelize

sys.path.insert(0, str(Path(__file__).parent / "ops"))
from test_downscale import reference_k_centroids

MODES = ["contrast", "k_centroid", "lanczos", "nearest", "bilinear", "bicubic", "area"]

BASE_CASES = [
    {"mode": m, "thickness": t, "do_color_match": cm}
    for m, t, cm in itertools.product(MODES, [0, 3], [True, False])
]
EXTRA_CASES = [
    {"sharpen_mode": "unsharp"},
    {"sharpen_mode": "laplacian"},
    {"weight_mapping": "contrast_ratio"},
    {"weight_mapping": "contrast_gated", "weight_normalize": "per_image"},
    {"thickness": 1},
    {"thickness": 6},
    {"pixel_size": 3},
    {"pixel_size": 8},
    {"do_quant": True, "num_colors": 16, "dither_mode": "none"},
    {"do_quant": True, "num_colors": 32, "dither_mode": "ordered"},
    {"do_quant": True, "num_colors": 8, "dither_mode": "error_diffusion"},
    {"do_quant": True, "quant_mode": "weighted-kmeans", "dither_mode": "none"},
    {
        "do_quant": True,
        "quant_mode": "weighted-kmeans",
        "thickness": 0,
        "dither_mode": "none",
    },
    {"do_quant": True, "quant_mode": "repeat-kmeans", "dither_mode": "ordered"},
    {"no_post_upscale": True},
]


def k_centroid_expected(img, kw):
    """The reference pipeline with the exact k-centroid tie rule: blocks whose
    final clusters have equal sizes take c0 (see kcentroid.slang)."""
    ref_kw = dict(kw, no_post_upscale=True, return_intermediate=True)
    down, expanded, _ = ref_pixelize(img, **ref_kw)
    p = kw["pixel_size"]
    cent, tie = reference_k_centroids(expanded, p)
    b, c, oh, ow = down.shape
    blocks = down.permute(0, 2, 3, 1).reshape(-1, c).clone()
    blocks[tie] = cent[tie, 0]
    down = blocks.reshape(b, oh, ow, c).permute(0, 3, 1, 2)
    if kw.get("no_post_upscale"):
        return down
    return F.interpolate(down, scale_factor=p, mode="nearest-exact")


def _check(backend, shape, case, batch=1):
    ctx = shared_context(backend)
    img = photo(*shape, batch=batch)
    kw = {"pixel_size": 4, "thickness": 3}
    kw.update(case)
    out = pixelize(img, context=ctx, **kw)
    if kw.get("mode") == "k_centroid" and not kw.get("do_quant"):
        ref = k_centroid_expected(img, kw)
    else:
        ref = ref_pixelize(img, **kw)
    stats = compare(out, ref)
    assert stats["finite"]
    loose = kw.get("quant_mode") == "repeat-kmeans"
    limit = 3e-2 if loose else 1e-2
    assert stats["frac_gt_1_255"] < limit, (case, stats)
    return stats


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("case", BASE_CASES + EXTRA_CASES, ids=str)
def test_pixelize_matches_reference(backend, case):
    _check(backend, (96, 128), case)


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("shape", [(98, 131), (97, 128), (96, 130)])
def test_pixelize_padding(backend, shape):
    _check(backend, shape, {})


@pytest.mark.parametrize("backend", all_backends())
def test_pixelize_batch(backend):
    _check(backend, (64, 80), {"do_quant": True, "num_colors": 16}, batch=3)


@pytest.mark.parametrize("backend", all_backends())
def test_pixelize_fp16_input_returns_fp16(backend):
    ctx = shared_context(backend)
    img = photo(64, 64)
    out = pixelize(img.half(), pixel_size=4, thickness=2, context=ctx)
    ref = pixelize(img, pixel_size=4, thickness=2, context=ctx)
    assert out.dtype == torch.float16
    assert compare(out, ref)["frac_gt_1_255"] < 1e-2


@pytest.mark.parametrize("backend", all_backends())
def test_pixelize_intermediates(backend):
    ctx = shared_context(backend)
    img = photo(64, 96, batch=2)
    out, expanded, weight = pixelize(
        img, pixel_size=4, thickness=3, return_intermediate=True, context=ctx
    )
    r_out, r_exp, r_w = ref_pixelize(
        img, pixel_size=4, thickness=3, return_intermediate=True
    )
    assert compare(weight, r_w)["max"] < 2e-5
    assert compare(expanded, r_exp)["max"] < 2e-4
    assert compare(out, r_out)["frac_gt_1_255"] < 1e-2
