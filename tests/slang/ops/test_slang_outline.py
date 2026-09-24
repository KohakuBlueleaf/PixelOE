import pytest
from slang_helpers import (
    all_backends,
    compare,
    dev,
    host,
    photo,
    reference_spread,
    shared_context,
)

from pixeloe.slang.ops.outline import (
    expansion_weight,
    morph,
    normalized_weight,
    outline_expansion,
)
from pixeloe.torch.minmax import KERNELS, dilate_cont, erode_cont
from pixeloe.torch.outline import expansion_weight as ref_expansion_weight
from pixeloe.torch.outline import outline_expansion as ref_outline_expansion

MAPPINGS = ["current", "contrast_ratio", "contrast_gated"]
NORMS = ["global", "per_image", "none"]


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("normalize", NORMS)
@pytest.mark.parametrize("k,shape", [(4, (48, 64)), (5, (45, 35)), (2, (20, 18))])
def test_expansion_weight(backend, mapping, normalize, k, shape):
    ctx = shared_context(backend)
    img = photo(*shape, batch=2)
    arr = dev(ctx, img)
    rw = expansion_weight(ctx, arr, k, k // 2, mapping=mapping, normalize=normalize)
    out = host(ctx, normalized_weight(ctx, rw), img)
    ctx.release(arr)
    ref = ref_expansion_weight(img, k, k // 2, mapping=mapping, normalize=normalize)
    stats = compare(out, ref[:, 0])
    # contrast_ratio divides by the local contrast: ill-conditioned in flat
    # regions, so the reference's own CPU/CUDA spread sets the scale
    spread = reference_spread(
        ref_expansion_weight, img, k, k // 2, mapping=mapping, normalize=normalize
    )
    limit = max(2e-5, 4 * spread["max"]) if spread else 2e-5
    assert stats["max"] < limit, (stats, spread)


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("iters", sorted(KERNELS))
@pytest.mark.parametrize("mode", ["erode", "dilate"])
def test_morph(backend, iters, mode):
    ctx = shared_context(backend)
    img = photo(29, 41, batch=2)
    arr = dev(ctx, img)
    out = host(ctx, morph(ctx, arr, iters, mode, mode == "dilate"), img)
    ctx.release(arr)
    ref_fn = dilate_cont if mode == "dilate" else erode_cont
    ref = ref_fn(img, KERNELS[iters].to(img), 1)
    assert compare(out, ref)["max"] < 1e-6


@pytest.mark.parametrize("backend", all_backends())
@pytest.mark.parametrize("thickness", sorted(KERNELS))
@pytest.mark.parametrize("mapping", MAPPINGS)
def test_outline_expansion(backend, thickness, mapping):
    ctx = shared_context(backend)
    img = photo(56, 72, batch=2)
    arr = dev(ctx, img)
    out, w = outline_expansion(
        ctx, arr, thickness, thickness, 4, weight_mapping=mapping
    )
    out_t = host(ctx, out, img)
    w_t = host(ctx, w, img)
    ctx.release(arr)
    ref_out, ref_w = ref_outline_expansion(
        img, thickness, thickness, 4, weight_mapping=mapping
    )
    spread = reference_spread(
        ref_outline_expansion, img, thickness, thickness, 4, weight_mapping=mapping
    )
    limit = max(5e-5, 4 * spread["max"]) if spread else 5e-5
    assert compare(w_t, ref_w[:, 0])["max"] < limit
    stats = compare(out_t, ref_out)
    assert stats["max"] < limit, (stats, spread)
