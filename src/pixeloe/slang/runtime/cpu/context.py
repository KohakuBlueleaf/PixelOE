"""CPU backend: Slang kernels compiled to native code, dispatched across all
cores; buffers are host numpy arrays. Same interface as runtime.Context."""

import ctypes
import math
import time
from collections import defaultdict

import numpy as np
import torch

from ..context import ROW_LEN, ContextStats, DeviceArray, _round_capacity, as_dtype
from ..paths import cache_root
from .kernels import BufferRef, KernelLibrary
from .toolchain import ISAS, build_dispatcher

_LIBRARY = None
_DISPATCHER = None


def _shared():
    """Process-wide kernel library and dispatcher (built once, disk-cached)."""
    global _LIBRARY, _DISPATCHER
    if _LIBRARY is None:
        _LIBRARY = KernelLibrary(cache_root() / "cpu")
        dll = ctypes.CDLL(str(build_dispatcher(_LIBRARY.dir.parent / "dispatch")))
        dll.pixeloe_dispatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        dll.pixeloe_dispatch.restype = None
        dll.pixeloe_set_threads.argtypes = [ctypes.c_int]
        dll.pixeloe_max_threads.restype = ctypes.c_int
        _DISPATCHER = dll
    return _LIBRARY, _DISPATCHER


class CpuContext:
    """isa: kernel build variant ("slang", "sse2", "avx2", "avx512");
    threads: OpenMP worker count (None = runtime default). The worker count
    is process-wide: the last context to dispatch sets it."""

    backend = "cpu"
    uses_cuda_stream = False

    def __init__(self, backend="cpu", profile=False, isa="slang", threads=None):
        if isa not in ISAS:
            raise ValueError(f"Unknown CPU isa {isa!r} (have {ISAS})")
        self.profile = profile
        self.isa = isa
        self.threads = threads
        self.library, self.dispatcher = _shared()
        self.default_threads = self.dispatcher.pixeloe_max_threads()
        self._constants = {}
        self._free = defaultdict(list)
        self.stats = ContextStats()

    # ------------------------------------------------------------------ buffers
    def empty(self, shape, dtype=np.float32):
        dtype = as_dtype(dtype)
        shape = tuple(int(s) for s in shape)
        nbytes = max(math.prod(shape) * dtype.itemsize, 16)
        capacity = _round_capacity(nbytes)
        free = self._free[capacity]
        if free:
            raw = free.pop()
        else:
            raw = np.empty(capacity, dtype=np.uint8)
            self.stats.pool_bytes += capacity
            self.stats.peak_pool_bytes = max(
                self.stats.peak_pool_bytes, self.stats.pool_bytes
            )
            self.stats.allocations += 1
        self.stats.live_bytes += capacity
        self.stats.peak_live_bytes = max(
            self.stats.peak_live_bytes, self.stats.live_bytes
        )
        return DeviceArray(raw, shape, dtype, capacity)

    def release(self, *arrays):
        for arr in arrays:
            if arr is None:
                continue
            self._free[arr.capacity].append(arr.buffer)
            self.stats.live_bytes -= arr.capacity

    def trim(self):
        for capacity, free in self._free.items():
            self.stats.pool_bytes -= capacity * len(free)
            free.clear()

    def constant(self, data, dtype=None):
        data = np.ascontiguousarray(data, dtype=dtype)
        capacity = max(data.nbytes, 16)
        raw = np.zeros(capacity, dtype=np.uint8)
        raw[: data.nbytes] = data.reshape(-1).view(np.uint8)
        self.stats.constant_bytes += capacity
        return DeviceArray(raw, data.shape, data.dtype, capacity)

    def cached_constant(self, key, factory, dtype=None):
        arr = self._constants.get(key)
        if arr is None:
            arr = self.constant(factory(), dtype)
            self._constants[key] = arr
        return arr

    @staticmethod
    def view(arr):
        return arr.buffer[: arr.nbytes].view(arr.dtype).reshape(arr.shape)

    # ------------------------------------------------------------ torch interop
    # Both directions copy with torch (multi-threaded) through a tensor view
    # of the pooled numpy buffer.
    def from_torch(self, tensor, dtype=np.float32):
        arr = self.empty(tensor.shape, dtype)
        view = torch.from_numpy(self.view(arr))
        view.copy_(tensor.detach().reshape(view.shape))
        return arr

    def to_torch(self, arr, like=None):
        view = torch.from_numpy(self.view(arr))
        dtype = view.dtype
        device = "cpu"
        if like is not None:
            dtype = like.dtype if arr.dtype == np.float32 else view.dtype
            device = like.device
        data = torch.empty(view.shape, dtype=dtype, device=device)
        data.copy_(view)
        return data

    def download(self, arr):
        return self.view(arr).copy()

    # ----------------------------------------------------------------- dispatch
    def dispatch(self, module, entry, threads, **params):
        kernel = self.library.get(module, entry, self.isa)
        self.dispatcher.pixeloe_set_threads(self.threads or self.default_threads)
        threads = [int(t) for t in threads] + [1] * (3 - len(threads))
        groups = [-(-t // n) for t, n in zip(threads, kernel.numthreads)]
        args = kernel.params()
        for name, kind in kernel.fields:
            value = params[name]
            if kind == "buffer":
                raw = value.buffer
                setattr(args, name, BufferRef(raw.ctypes.data, raw.nbytes // 4))
            else:
                setattr(args, name, value)
        t0 = time.perf_counter() if self.profile else 0.0
        if all(groups):
            self.dispatcher.pixeloe_dispatch(kernel.fn, *groups, ctypes.byref(args))
        if self.profile:
            ms = (time.perf_counter() - t0) * 1e3
            self.stats.per_kernel_ms[entry] += ms
            self.stats.per_kernel_calls[entry] += 1
            self.stats.dispatch_log.append(
                {
                    "name": entry,
                    "threads": tuple(threads),
                    "group": tuple(kernel.numthreads),
                    "ms": ms,
                }
            )
        self.stats.dispatches += 1

    def dispatch_flat(self, module, entry, items, /, **params):
        if items <= ROW_LEN:
            threads, row_len = (items, 1), items
        else:
            threads, row_len = (ROW_LEN, -(-items // ROW_LEN)), ROW_LEN
        self.dispatch(module, entry, threads, row_len=row_len, **params)

    def copy(self, src, dst):
        dst.buffer[: src.nbytes] = src.buffer[: src.nbytes]

    def clone(self, src):
        dst = self.empty(src.shape, src.dtype)
        self.copy(src, dst)
        return dst

    def clear_uint(self, arr):
        arr.buffer[:] = 0

    def flush(self):
        pass

    def wait(self):
        pass
