import pytest
import torch

import pixeloe.torch.env as pixeloe_env
from pixeloe.torch.outline import expansion_weight, outline_expansion


@pytest.fixture(autouse=True)
def disable_compile():
    old = pixeloe_env.TORCH_COMPILE
    pixeloe_env.TORCH_COMPILE = False
    yield
    pixeloe_env.TORCH_COMPILE = old


def synthetic_image(batch=2, height=16, width=16):
    x = (
        torch.linspace(0, 1, width)
        .reshape(1, 1, 1, width)
        .expand(batch, 1, height, width)
    )
    y = (
        torch.linspace(0, 1, height)
        .reshape(1, 1, height, 1)
        .expand(batch, 1, height, width)
    )
    line = torch.zeros(batch, 1, height, width)
    line[:, :, height // 2] = 1
    return torch.cat([x, y, line], dim=1).clamp(0, 1)


def legacy_reference_weight(img, k=4, stride=2, avg_scale=10, dist_scale=3):
    from pixeloe.torch.lab import rgb_to_lab
    from pixeloe.torch.outline import local_stat

    lab = rgb_to_lab(img)
    l = lab[:, 0:1] / 100
    l_med = local_stat(l, k * 2, stride, stat="median")
    l_min = local_stat(l, k, stride, stat="min")
    l_max = local_stat(l, k, stride, stat="max")
    bright_dist = l_max - l_med
    dark_dist = l_med - l_min
    weight = (l_med - 0.5) * avg_scale - (bright_dist - dark_dist) * dist_scale
    weight = torch.sigmoid(weight)
    return (weight - weight.amin()) / (weight.amax() - weight.amin() + 1e-8)


def test_current_expansion_weight_matches_previous_formula():
    img = synthetic_image()
    actual = expansion_weight(img, k=4, stride=2)
    expected = legacy_reference_weight(img, k=4, stride=2)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mapping", ["current", "contrast_ratio", "contrast_gated"])
def test_expansion_weight_shape_range_and_finite(mapping):
    img = synthetic_image()
    weight = expansion_weight(img, k=4, stride=2, mapping=mapping)
    assert weight.shape == (2, 1, 16, 16)
    assert torch.isfinite(weight).all()
    assert weight.min() >= 0
    assert weight.max() <= 1


def test_outline_expansion_preserves_shape():
    img = synthetic_image(batch=1)
    out, weight = outline_expansion(img, erode_iters=2, dilate_iters=2, k=4)
    assert out.shape == img.shape
    assert weight.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()
    assert torch.isfinite(weight).all()


def test_per_image_weight_normalization_is_independent_per_batch_item():
    img = synthetic_image(batch=2)
    img[1] = img[1] * 0.25
    weight = expansion_weight(img, k=4, stride=2, normalize="per_image")
    assert torch.allclose(weight.amin(dim=(1, 2, 3)), torch.zeros(2), atol=1e-6)
    assert torch.allclose(weight.amax(dim=(1, 2, 3)), torch.ones(2), atol=1e-6)
