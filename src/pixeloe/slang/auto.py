"""Backend selection for the public pixelize(): the fastest Slang backend that
runs on this machine for the input tensor, or None for the torch pipeline.

Candidates by input device, fastest first (benchmarks/slang_vs_torch):
    CUDA tensor   cuda -> vulkan -> d3d12   (all share the tensor's GPU)
    CPU tensor    cpu                       (native kernels; needs MSVC)
Each candidate is probed once per process by pixelizing a small image; a
backend whose runtime, compiler or driver is missing fails the probe and the
next one is tried.
"""

import torch

from ..backend import requested_backend
from ..logger import logger
from .pixelize import pixelize as slang_pixelize
from .runtime.registry import ALL_BACKENDS, get_context

CANDIDATES = {"cuda": ("cuda", "vulkan", "d3d12"), "cpu": ("cpu",)}
SUPPORTED_DTYPES = (torch.float32, torch.float16)
PROBE_SHAPE = (1, 3, 32, 32)

_PROBED = {}


def _probe(name, device):
    """True when backend `name` pixelizes a small image on `device`."""
    key = (name, str(device))
    ok = _PROBED.get(key)
    if ok is None:
        try:
            img = torch.linspace(0, 1, 32 * 32 * 3, device=device)
            img = img.reshape(PROBE_SHAPE)
            out = slang_pixelize(img, pixel_size=4, context=get_context(name))
            ok = out.shape == PROBE_SHAPE and bool(torch.isfinite(out).all())
        except Exception as exc:  # noqa: BLE001 - any failure: not usable here
            logger.info("slang backend %s unavailable: %s", name, exc)
            ok = False
        _PROBED[key] = ok
    return ok


def select_context(img_t, backend=None):
    """Slang context for img_t, or None to run the torch pipeline.

    backend: "auto" (probe the candidates), "torch", or a Slang backend spec
    ("cuda", "vulkan", "d3d12", "cpu", with the options create_context
    accepts); an explicit Slang backend is used without probing.
    """
    backend = requested_backend(backend)
    if backend == "torch":
        return None
    if backend != "auto":
        if backend.partition(":")[0] not in ALL_BACKENDS:
            raise ValueError(f"unknown pixelize backend: {backend!r}")
        return get_context(backend)
    if img_t.dtype not in SUPPORTED_DTYPES or img_t.requires_grad:
        return None
    if img_t.is_cuda:
        if img_t.device.index not in (None, torch.cuda.current_device()):
            return None
        device_class = "cuda"
    elif img_t.device.type == "cpu":
        device_class = "cpu"
    else:
        return None
    for name in CANDIDATES[device_class]:
        if _probe(name, img_t.device):
            return get_context(name)
    return None
