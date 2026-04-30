import pytest
import torch

import pixeloe.torch.env as pixeloe_env
from pixeloe.torch.pixelize import pixelize


@pytest.fixture(autouse=True)
def disable_compile():
    old = pixeloe_env.TORCH_COMPILE
    pixeloe_env.TORCH_COMPILE = False
    yield
    pixeloe_env.TORCH_COMPILE = old


def synthetic_image(batch=1, height=18, width=22):
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
    checker = (
        (torch.arange(height).reshape(height, 1) + torch.arange(width)) % 2
    ).float()
    checker = checker.reshape(1, 1, height, width).expand(batch, 1, height, width)
    return torch.cat([x, y, checker], dim=1).clamp(0, 1)


def test_pixelize_default_preserves_padded_grid_size_after_upscale():
    img = synthetic_image(height=18, width=22)
    out = pixelize(img, pixel_size=4, thickness=0, do_color_match=False)
    assert out.shape == (1, 3, 20, 24)
    assert torch.isfinite(out).all()


def test_pixelize_no_post_upscale_returns_pixel_grid():
    img = synthetic_image(height=18, width=22)
    out = pixelize(
        img,
        pixel_size=4,
        thickness=0,
        do_color_match=False,
        no_post_upscale=True,
    )
    assert out.shape == (1, 3, 5, 6)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("mapping", ["current", "contrast_ratio", "contrast_gated"])
def test_pixelize_accepts_weight_mapping_strategies(mapping):
    img = synthetic_image(height=16, width=16)
    out, expanded, weight = pixelize(
        img,
        pixel_size=4,
        thickness=2,
        do_color_match=False,
        return_intermediate=True,
        weight_mapping=mapping,
    )
    assert out.shape == img.shape
    assert expanded.shape == img.shape
    assert weight.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_pixelize_weighted_quantization_without_outline_uses_weight_mapping():
    img = synthetic_image(height=12, width=12)
    out = pixelize(
        img,
        pixel_size=4,
        thickness=0,
        do_color_match=False,
        do_quant=True,
        num_colors=4,
        quant_mode="weighted-kmeans",
        dither_mode="none",
        weight_mapping="contrast_ratio",
    )
    assert out.shape == (1, 3, 12, 12)
    assert torch.isfinite(out).all()
