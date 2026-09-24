"""Filesystem locations used by the Slang runtime."""

import os
from pathlib import Path

SHADER_ROOT = Path(__file__).resolve().parent.parent / "shaders"


def cache_root():
    """Per-user cache for built CPU kernels (PIXELOE_CACHE overrides)."""
    env = os.environ.get("PIXELOE_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "pixeloe" / "slang"
