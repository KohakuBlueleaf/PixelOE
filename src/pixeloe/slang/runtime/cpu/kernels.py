"""Slang CPU-target kernels as native DLLs, cached on disk.

Slang compiles a compute entry point for the CPU into C++ and a DLL (via
MSVC) exporting `entry(ComputeVaryingInput*, void* params, void* globals)`
that runs a range of workgroups. slangpy's CPU device performs that build
when a pipeline is first dispatched; with `dump_intermediates` it leaves the
generated C++ and DLL on disk. This module triggers the build with a dummy
one-group dispatch, keeps the DLL and C++ in a content-hashed cache, and
reads from the C++ what a launch needs: the entry-point parameter struct
(declaration order, natural C layout) and the workgroup size.
"""

import ctypes
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import slangpy as spy

from ..paths import SHADER_ROOT
from .toolchain import ISA_FLAGS, build_dll

CACHE_VERSION = "cpu-kernels-1"

_BUFFER = re.compile(r"(RW)?StructuredBuffer<(\w+)>")
_SCALARS = {
    "uint32_t": ctypes.c_uint32,
    "int32_t": ctypes.c_int32,
    "float": ctypes.c_float,
    "uint64_t": ctypes.c_uint64,
    "int64_t": ctypes.c_int64,
}


class BufferRef(ctypes.Structure):
    """Slang CPU prelude (RW)StructuredBuffer<T>: { T* data; size_t count; }."""

    _fields_ = [("data", ctypes.c_void_p), ("count", ctypes.c_size_t)]


@dataclass
class CpuKernel:
    name: str
    fn: int  # address of the exported range function
    numthreads: tuple
    params: type  # ctypes.Structure of the entry-point parameters
    fields: list  # [(slang name, kind)] kind: "buffer" or ctypes scalar type
    dll: object  # keeps the DLL loaded


def shader_hash():
    h = hashlib.sha256(CACHE_VERSION.encode())
    h.update(spy.__version__.encode() if hasattr(spy, "__version__") else b"")
    for f in sorted(SHADER_ROOT.rglob("*.slang")):
        h.update(f.relative_to(SHADER_ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def entry_uniforms(module, entry):
    """[(type, name)] of an entry point's `uniform` parameters, parsed from
    its Slang source; entry points stamped out by a `#define M(NAME, ...)`
    macro are read from the macro's `void NAME(...)` signature."""
    text = (SHADER_ROOT / f"{module}.slang").read_text()
    m = re.search(rf"\bvoid\s+{re.escape(entry)}\s*\(", text)
    if m is None:
        use = re.search(rf"^(\w+)\(\s*{re.escape(entry)}\s*,", text, re.MULTILINE)
        if use is None:
            raise KeyError(f"{entry} not found in {module}.slang")
        macro = re.search(rf"#define {use.group(1)}\(.*?\n(?:.*\\\n)*.*", text)
        text = macro.group(0).replace("\\\n", "\n")
        m = re.search(r"\bvoid\s+NAME\s*\(", text)
    depth, i = 1, m.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    params = []
    for part in re.split(r",(?![^<]*>)", text[m.end() : i - 1]):
        tokens = part.split()
        if tokens and tokens[0] == "uniform":
            params.append((" ".join(tokens[1:-1]), tokens[-1]))
    return params


def parse_kernel(cpp_text, entry):
    """Parameter fields and workgroup size of `entry` in generated C++."""
    m = re.search(
        rf"// \[numthreads\((\d+), (\d+), (\d+)\)\]\s*SLANG_PRELUDE_EXPORT\s*void {re.escape(entry)}\(",
        cpp_text,
    )
    if m is None:
        raise RuntimeError(f"entry {entry} not found in generated C++")
    numthreads = tuple(int(g) for g in m.groups())
    s = re.search(r"struct (EntryPointParams_\d+)\s*\{(.*?)\};", cpp_text, re.DOTALL)
    fields = []
    for line in s.group(2).strip().splitlines():
        decl = line.strip().rstrip(";")
        ctype, name = decl.rsplit(" ", 1)
        name = re.sub(r"_\d+$", "", name)
        if _BUFFER.fullmatch(ctype):
            fields.append((name, "buffer"))
        elif ctype in _SCALARS:
            fields.append((name, _SCALARS[ctype]))
        else:
            raise RuntimeError(f"unsupported CPU kernel parameter type {ctype}")
    struct_fields = [
        (name, BufferRef if kind == "buffer" else kind) for name, kind in fields
    ]
    params = type(f"{entry}_params", (ctypes.Structure,), {"_fields_": struct_fields})
    return fields, params, numthreads


class KernelLibrary:
    """Builds and loads CPU kernels; one instance per process is enough."""

    def __init__(self, cache_root):
        self.dir = Path(cache_root) / shader_hash()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._device = None
        self._dump = None
        self._dummy = None
        self._kernels = {}

    def _builder(self):
        if self._device is None:
            self._dump = self.dir / "_dump"
            shutil.rmtree(self._dump, ignore_errors=True)
            self._dump.mkdir()
            opts = spy.SlangCompilerOptions(
                {
                    "include_paths": [SHADER_ROOT, SHADER_ROOT / "common"],
                    "dump_intermediates": True,
                    "dump_intermediates_prefix": str(self._dump / "k_"),
                    "optimization": spy.SlangOptimizationLevel.maximal,
                }
            )
            self._device = spy.Device(type=spy.DeviceType.cpu, compiler_options=opts)
            self._dummy = self._device.create_buffer(
                size=4096,
                usage=spy.BufferUsage.shader_resource
                | spy.BufferUsage.unordered_access,
            )
        return self._device

    def _build(self, module, entry, stem):
        device = self._builder()
        before = set(self._dump.iterdir())
        kernel = device.create_compute_kernel(
            device.load_program(f"{module}.slang", [entry])
        )
        args = {}
        for ptype, pname in entry_uniforms(module, entry):
            if "Buffer" in ptype:
                args[pname] = self._dummy
            elif ptype == "float":
                args[pname] = 0.0
            else:
                args[pname] = 0
        kernel.dispatch(thread_count=[1, 1, 1], **args)
        device.wait()
        new = sorted(
            set(self._dump.iterdir()) - before, key=lambda p: p.stat().st_mtime
        )
        cpp = [p for p in new if p.suffix == ".cpp"]
        dll = [p for p in new if p.suffix == ".dll"]
        if not cpp or not dll:
            raise RuntimeError(f"slang produced no CPU build for {module}:{entry}")
        shutil.copyfile(cpp[-1], stem.with_suffix(".cpp"))
        shutil.copyfile(dll[-1], stem.with_suffix(".dll"))

    def dll_path(self, module, entry, isa="slang"):
        """Kernel DLL for an instruction-set variant (see toolchain.ISA_FLAGS);
        variants are rebuilt from the Slang-generated C++ on first use."""
        stem = self.dir / f"{module.replace('/', '__')}__{entry}"
        if not stem.with_suffix(".dll").exists():
            self._build(module, entry, stem)
        if isa == "slang":
            return stem.with_suffix(".dll")
        dll = self.dir / isa / f"{stem.name}.dll"
        if not dll.exists():
            build_dll(stem.with_suffix(".cpp"), dll, ISA_FLAGS[isa])
        return dll

    def get(self, module, entry, isa="slang"):
        key = (module, entry, isa)
        k = self._kernels.get(key)
        if k is not None:
            return k
        dll_path = self.dll_path(module, entry, isa)
        cpp = self.dir / f"{module.replace('/', '__')}__{entry}.cpp"
        fields, params, numthreads = parse_kernel(
            cpp.read_text(errors="replace"), entry
        )
        dll = ctypes.CDLL(str(dll_path))
        fn = ctypes.cast(getattr(dll, entry), ctypes.c_void_p).value
        k = CpuKernel(entry, fn, numthreads, params, fields, dll)
        self._kernels[key] = k
        return k
