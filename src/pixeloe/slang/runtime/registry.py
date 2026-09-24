"""Context construction and process-wide contexts, one per backend."""

from .context import BACKENDS, Context
from .cpu.context import CpuContext

ALL_BACKENDS = tuple(BACKENDS) + ("cpu",)

_CONTEXTS = {}


def create_context(backend="cuda", profile=False):
    """A new context. backend: "cuda", "d3d12", "vulkan" or "cpu".
    GPU: "vulkan:<adapter index or name>", e.g. "d3d12:B50" (adapters other
    than torch's CUDA GPU stage tensors through host memory).
    CPU: "cpu[:<isa>][:t<threads>]", e.g. "cpu:avx512:t64"."""
    name, _, adapter = backend.partition(":")
    if name == "cpu":
        isa, threads = "slang", None
        for part in filter(None, adapter.split(":")):
            if part.startswith("t") and part[1:].isdigit():
                threads = int(part[1:])
            else:
                isa = part
        return CpuContext(profile=profile, isa=isa, threads=threads)
    return Context(name, profile=profile, adapter=adapter or None)


def default_backend(tensor=None):
    if tensor is not None and getattr(tensor, "is_cuda", False):
        return "cuda"
    return "cpu"


def get_context(backend=None, tensor=None, profile=False):
    backend = backend or default_backend(tensor)
    key = (backend, profile)
    ctx = _CONTEXTS.get(key)
    if ctx is None:
        ctx = create_context(backend, profile)
        _CONTEXTS[key] = ctx
    return ctx
