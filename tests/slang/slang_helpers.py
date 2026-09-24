"""Shared helpers for the Slang conformance tests (torch path = oracle)."""

from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pixeloe.slang.runtime.registry import ALL_BACKENDS, create_context

ROOT = Path(__file__).resolve().parents[2]
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKENDS = ALL_BACKENDS


@cache
def backend_available(name):
    try:
        shared_context(name)
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False
    return True


@cache
def shared_context(name):
    return create_context(name)


def all_backends():
    return [b for b in BACKENDS if backend_available(b)]


@cache
def _photo(name):
    return Image.open(ROOT / "img" / name).convert("RGB")


def photo(height, width, name="snow-leopard.webp", batch=1):
    """A real photo resized to (height, width) as [B,3,H,W] float32 on the
    torch reference device; batch items are different crops."""
    img = _photo(name)
    items = []
    for i in range(batch):
        w0, h0 = img.size
        crop = img.crop((i * w0 // 16, i * h0 // 16, w0 - i * w0 // 32, h0))
        arr = np.asarray(crop.resize((width, height), Image.Resampling.BICUBIC))
        items.append(torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255)
    return torch.stack(items).to(TORCH_DEVICE)


def dev(ctx, tensor):
    return ctx.from_torch(tensor)


def host(ctx, arr, like):
    out = ctx.to_torch(arr, like=like)
    ctx.release(arr)
    return out


def reference_spread(fn, *tensors, **kwargs):
    """How much the reference itself moves between torch CUDA and CPU
    (float summation order, library kernels): compare(cuda_out, cpu_out).
    A reproduction is held to a small multiple of this, never to less than
    the reference is to itself. Returns None without CUDA."""
    if TORCH_DEVICE != "cuda":
        return None
    out_gpu = fn(*tensors, **kwargs)
    out_cpu = fn(
        *(t.cpu() if isinstance(t, torch.Tensor) else t for t in tensors), **kwargs
    )
    if isinstance(out_gpu, tuple):
        out_gpu, out_cpu = out_gpu[0], out_cpu[0]
    return compare(out_gpu, out_cpu)


def compare(actual, expected):
    """Error statistics of actual vs expected (both torch tensors)."""
    a = actual.detach().float().cpu()
    e = expected.detach().float().cpu()
    assert a.shape == e.shape, (a.shape, e.shape)
    diff = (a - e).abs()
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "frac_gt_1e-4": float((diff > 1e-4).float().mean()),
        "frac_gt_1_255": float((diff > 1 / 255).float().mean()),
        "finite": bool(torch.isfinite(a).all()),
    }
