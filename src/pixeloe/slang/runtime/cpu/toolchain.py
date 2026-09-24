"""MSVC discovery and native builds: the parallel dispatcher DLL and ISA
variants of Slang-generated kernel C++."""

import functools
import hashlib
import subprocess
from pathlib import Path

VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
DISPATCH_SRC = Path(__file__).with_name("dispatch.cpp")

# instruction-set variants of the kernel DLLs. "slang" = the DLL slangpy
# builds itself (MSVC defaults: SSE2 code); the others rebuild the same
# generated C++ with MSVC /O2 and an /arch target, keeping /fp:precise so the
# compensated and fixed-point sums keep their ordering.
ISA_FLAGS = {
    "sse2": "/O2 /fp:precise",
    "avx2": "/O2 /fp:precise /arch:AVX2",
    "avx512": "/O2 /fp:precise /arch:AVX512",
}
ISAS = ("slang",) + tuple(ISA_FLAGS)


@functools.cache
def vcvars64():
    """Path of vcvars64.bat of the newest Visual Studio with the x64 C++ tools."""
    if not VSWHERE.exists():
        raise RuntimeError("vswhere.exe not found: install Visual Studio C++ tools")
    root = subprocess.run(
        [
            str(VSWHERE),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    bat = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not bat.exists():
        raise RuntimeError(f"vcvars64.bat not found under {root}")
    return bat


def build_dll(src, dll, flags):
    """cl <flags> /LD src -> dll (in dll's directory)."""
    dll = Path(dll)
    dll.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'call "{vcvars64()}" >nul && cl /nologo {flags} /EHsc /LD "{src}" '
        f"/Fe:{dll.name} /Fo:{dll.stem}.obj"
    )
    proc = subprocess.run(
        f'cmd /d /s /c "{cmd}"',
        capture_output=True,
        text=True,
        cwd=dll.parent,
        check=False,
    )
    if proc.returncode != 0 or not dll.exists():
        raise RuntimeError(f"build of {dll.name} failed:\n{proc.stdout}\n{proc.stderr}")
    return dll


def build_dispatcher(out_root):
    """dispatch.cpp -> DLL (OpenMP, /O2), cached by the source's hash."""
    tag = hashlib.sha256(DISPATCH_SRC.read_bytes()).hexdigest()[:12]
    dll = Path(out_root) / tag / "pixeloe_dispatch.dll"
    if not dll.exists():
        build_dll(DISPATCH_SRC, dll, "/O2 /openmp")
    return dll
