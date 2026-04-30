import pytest
import torch

import pixeloe.torch.env as pixeloe_env
from pixeloe.torch.color import quantize_and_dither
from pixeloe.torch.downscale.contrast_based import contrast_downscale


@pytest.fixture(autouse=True)
def disable_compile():
    old = pixeloe_env.TORCH_COMPILE
    pixeloe_env.TORCH_COMPILE = False
    yield
    pixeloe_env.TORCH_COMPILE = old


def synthetic_image(batch=2, height=12, width=16):
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
    blue = torch.full((batch, 1, height, width), 0.25)
    return torch.cat([x, y, blue], dim=1).clamp(0, 1)


def test_contrast_downscale_shape_and_finite():
    img = synthetic_image()
    out = contrast_downscale(img, patch_size=4)
    assert out.shape == (2, 3, 3, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dither_method", ["none", "ordered", "error_diffusion"])
def test_quantize_and_dither_methods_shape_and_finite(dither_method):
    img = synthetic_image(batch=1, height=6, width=6)
    out = quantize_and_dither(img, num_centroids=4, dither_method=dither_method)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
