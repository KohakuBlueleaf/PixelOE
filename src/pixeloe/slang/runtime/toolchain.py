"""Environment for Slang's downstream compilers, applied before slangpy loads.

- CUDA target: Slang compiles through NVRTC, located via CUDA_PATH, which its
  runtime reads when the slangpy extension is loaded.
- CPU target: Slang compiles generated C++ with MSVC; VSLANG=1033 makes cl.exe
  print English (ASCII) diagnostics, which slangpy decodes as UTF-8.
"""

import os
from pathlib import Path

CUDA_TOOLKIT_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")


def ensure_nvrtc():
    """When CUDA_PATH is unset, point it at the newest installed toolkit that
    ships an NVRTC DLL."""
    if os.environ.get("CUDA_PATH") or not CUDA_TOOLKIT_ROOT.is_dir():
        return
    for root in sorted(CUDA_TOOLKIT_ROOT.glob("v*"), reverse=True):
        if any(root.glob("bin/**/nvrtc64_*.dll")):
            os.environ["CUDA_PATH"] = str(root)
            return


def prepare_toolchain():
    ensure_nvrtc()
    os.environ.setdefault("VSLANG", "1033")
