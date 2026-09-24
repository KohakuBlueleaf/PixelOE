"""Slang device context: kernels, pooled buffers, batched dispatch, torch
interop and GPU-timestamp profiling."""

import atexit
import gc
import weakref
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from .paths import SHADER_ROOT
from .toolchain import prepare_toolchain

prepare_toolchain()

import slangpy as spy

# GPU backends (the CPU backend is cpu/context.py's CpuContext)
BACKENDS = {
    "cuda": spy.DeviceType.cuda,
    "d3d12": spy.DeviceType.d3d12,
    "vulkan": spy.DeviceType.vulkan,
}
# Backends whose buffers torch sees through CUDA external memory.
INTEROP_BACKENDS = ("d3d12", "vulkan")

BUFFER_USAGE = (
    spy.BufferUsage.shader_resource
    | spy.BufferUsage.unordered_access
    | spy.BufferUsage.copy_source
    | spy.BufferUsage.copy_destination
)

# Threads per row of a folded 1-D launch: 65535 groups of 256 (the D3D12
# per-axis group limit), a multiple of every 1-D kernel's group size.
ROW_LEN = 65535 * 256


@dataclass
class DeviceArray:
    """A device buffer viewed as a typed n-d array."""

    buffer: spy.Buffer
    shape: tuple
    dtype: np.dtype
    capacity: int

    @property
    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self):
        return self.numel * self.dtype.itemsize


@dataclass
class DispatchRecord:
    name: str
    threads: tuple
    query: object  # timestamp query index, or (start, end) CUDA events


@dataclass
class ContextStats:
    live_bytes: int = 0
    peak_live_bytes: int = 0
    constant_bytes: int = 0
    pool_bytes: int = 0
    peak_pool_bytes: int = 0
    dispatches: int = 0
    submits: int = 0
    allocations: int = 0
    per_kernel_ms: dict = field(default_factory=lambda: defaultdict(float))
    per_kernel_calls: dict = field(default_factory=lambda: defaultdict(int))

    def reset_peaks(self):
        self.peak_live_bytes = self.live_bytes
        self.peak_pool_bytes = self.pool_bytes
        self.dispatches = 0
        self.submits = 0
        self.allocations = 0
        self.per_kernel_ms = defaultdict(float)
        self.per_kernel_calls = defaultdict(int)


_TORCH_DTYPES = {
    np.dtype(np.float32): (torch.float32, spy.DataType.float32),
    np.dtype(np.uint32): (torch.int32, spy.DataType.uint32),
}


class Context:
    """Owns one slangpy device and everything allocated on it.

    Dispatches are recorded into one command encoder and submitted by
    ``flush``; buffers come from a size-keyed free-list pool, so repeated
    calls with the same shapes allocate nothing.

    Torch interop (``transfer``):
      "cuda"  the adapter is torch's CUDA GPU.
              cuda backend: shares torch's CUDA context; work is submitted
              on torch's current stream, buffers are viewed zero-copy.
              d3d12/vulkan: buffers are CUDA-shared; queue <-> stream
              ordering via shared semaphores (sync_to_cuda / sync_to_device).
      "host"  any other adapter (e.g. an Intel GPU): tensors are staged
              through host memory with blocking uploads / readbacks.

    adapter: None (the system default, the NVIDIA GPU here), an index into
    Device.enumerate_adapters(backend), or a case-insensitive name substring.
    """

    MAX_QUERIES = 16384

    def __init__(self, backend="cuda", profile=False, adapter=None):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown slang GPU backend: {backend}")
        self.backend = backend
        self.profile = profile
        include_paths = [SHADER_ROOT, SHADER_ROOT / "common"]
        info = resolve_adapter(backend, adapter)
        self.adapter_name = info.name if info else "default"
        on_torch_gpu = info is None or info.vendor_id == NVIDIA_VENDOR_ID
        if backend == "cuda" and not on_torch_gpu:
            raise ValueError("the cuda backend only runs on the NVIDIA GPU")
        if on_torch_gpu:
            self.transfer = "cuda"
            self.device = spy.create_torch_device(
                type=BACKENDS[backend], include_paths=include_paths
            )
        else:
            self.transfer = "host"
            self.device = spy.create_device(
                type=BACKENDS[backend],
                include_paths=include_paths,
                adapter_luid=list(info.luid),
            )
        _LIVE_CONTEXTS.add(self)
        self.usage = BUFFER_USAGE
        if self.transfer == "cuda" and backend in INTEROP_BACKENDS:
            self.usage = self.usage | spy.BufferUsage.shared
        self._kernels = {}
        self._constants = {}
        self._stage = {}
        self._free = defaultdict(list)
        self.stats = ContextStats()
        self._encoder = None
        self._records = []
        self._query_pool = None
        self._query_next = 0
        if profile:
            self._query_pool = self.device.create_query_pool(
                type=spy.QueryType.timestamp, count=self.MAX_QUERIES
            )

    @property
    def uses_cuda_stream(self):
        return self.transfer == "cuda"

    # ------------------------------------------------------------------ kernels
    def kernel(self, module, entry):
        key = (module, entry)
        k = self._kernels.get(key)
        if k is None:
            program = self.device.load_program(f"{module}.slang", [entry])
            k = self.device.create_compute_kernel(program)
            self._kernels[key] = k
        return k

    # ------------------------------------------------------------------ buffers
    def empty(self, shape, dtype=np.float32):
        dtype = np.dtype(dtype)
        shape = tuple(int(s) for s in shape)
        nbytes = max(int(np.prod(shape)) * dtype.itemsize, 16)
        capacity = _round_capacity(nbytes)
        free = self._free[capacity]
        if free:
            buffer = free.pop()
        else:
            buffer = self.device.create_buffer(size=capacity, usage=self.usage)
            self.stats.pool_bytes += capacity
            self.stats.peak_pool_bytes = max(
                self.stats.peak_pool_bytes, self.stats.pool_bytes
            )
            self.stats.allocations += 1
        self.stats.live_bytes += capacity
        self.stats.peak_live_bytes = max(
            self.stats.peak_live_bytes, self.stats.live_bytes
        )
        return DeviceArray(buffer, shape, dtype, capacity)

    def release(self, *arrays):
        for arr in arrays:
            if arr is None:
                continue
            self._free[arr.capacity].append(arr.buffer)
            self.stats.live_bytes -= arr.capacity

    def trim(self):
        """Free every pooled buffer that is not live."""
        for capacity, free in self._free.items():
            self.stats.pool_bytes -= capacity * len(free)
            free.clear()

    def constant(self, data, dtype=None):
        """A dedicated (never pooled) buffer holding host data.

        Uploads execute immediately on the host timeline, so they must never
        target a pooled buffer that recorded-but-unsubmitted work may still
        read; callers cache the returned constant.
        """
        data = np.ascontiguousarray(data, dtype=dtype)
        capacity = max(data.nbytes, 16)
        buffer = self.device.create_buffer(
            size=capacity, usage=self.usage, data=_pad_bytes(data, capacity)
        )
        self.stats.constant_bytes += capacity
        return DeviceArray(buffer, data.shape, data.dtype, capacity)

    def cached_constant(self, key, factory, dtype=None):
        """constant(factory()) memoised under key for this context."""
        arr = self._constants.get(key)
        if arr is None:
            arr = self.constant(factory(), dtype)
            self._constants[key] = arr
        return arr

    # ------------------------------------------------------------ torch interop
    def _torch_view(self, arr):
        _, spy_dtype = _TORCH_DTYPES[arr.dtype]
        return arr.buffer.to_torch(spy_dtype, list(arr.shape))

    def from_torch(self, tensor, dtype=np.float32):
        """Copy a torch tensor into a pooled device array (float32 or uint32)."""
        arr = self.empty(tensor.shape, dtype)
        if self.transfer == "host":
            host = np.ascontiguousarray(tensor.detach().to("cpu").numpy(), dtype=dtype)
            stage = self._staging(arr.capacity, spy.MemoryType.upload)
            # the staging write is immediate: its previous copy must be done
            self.wait()
            stage.copy_from_numpy(host.reshape(-1))
            self._enc().copy_buffer(arr.buffer, 0, stage, 0, host.nbytes)
            return arr
        # recorded work may still read a pooled buffer that this copy (issued
        # on torch's stream right now) overwrites: submit it first
        self.flush()
        view = self._torch_view(arr)
        src = tensor.detach()
        if not src.is_cuda:
            src = src.cuda()
        view.copy_(src.reshape(view.shape).to(view.dtype))
        return arr

    def to_torch(self, arr, like=None):
        """A fresh torch tensor holding arr (device and float dtype of `like`)."""
        if self.transfer == "host":
            data = torch.from_numpy(self.download(arr))
        else:
            self.flush()
            data = self._torch_view(arr).clone()
        if like is not None:
            dtype = like.dtype if arr.dtype == np.float32 else data.dtype
            data = data.to(device=like.device, dtype=dtype)
        return data

    def download(self, arr):
        """Blocking copy of arr to a numpy array. Host transfer copies through
        a host-cached read-back buffer: reading device-local memory directly
        measured 158 ms for 24 MB on Vulkan (Arc B50) vs ~2 ms staged."""
        if self.transfer == "host":
            stage = self._staging(arr.capacity, spy.MemoryType.read_back)
            self._enc().copy_buffer(stage, 0, arr.buffer, 0, arr.nbytes)
            self.wait()
            raw = stage.to_numpy().view(np.uint8)[: arr.nbytes]
        else:
            self.wait()
            raw = arr.buffer.to_numpy().view(np.uint8)[: arr.nbytes]
        return raw.view(arr.dtype).reshape(arr.shape).copy()

    def _staging(self, capacity, memory_type):
        """One persistent staging buffer per (size class, direction)."""
        key = (capacity, memory_type)
        buf = self._stage.get(key)
        if buf is None:
            usage = (
                spy.BufferUsage.copy_source
                if memory_type == spy.MemoryType.upload
                else spy.BufferUsage.copy_destination
            )
            buf = self.device.create_buffer(
                size=capacity, usage=usage, memory_type=memory_type
            )
            self._stage[key] = buf
            self.stats.constant_bytes += capacity
        return buf

    # ----------------------------------------------------------------- dispatch
    def _enc(self):
        if self._encoder is None:
            self._encoder = self.device.create_command_encoder()
        return self._encoder

    def dispatch(self, module, entry, threads, **params):
        """Record one compute dispatch; buffers may be passed as DeviceArray."""
        kernel = self.kernel(module, entry)
        enc = self._enc()
        args = {
            name: (value.buffer if isinstance(value, DeviceArray) else value)
            for name, value in params.items()
        }
        threads = [int(t) for t in threads] + [1] * (3 - len(threads))
        if self.profile and self.backend == "cuda":
            # timestamp queries read 0 on the CUDA backend: bracket the
            # dispatch with CUDA events on the stream it is submitted to
            self.flush()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel.dispatch(thread_count=threads, command_encoder=self._enc(), **args)
            self.flush()
            end.record()
            self._records.append(DispatchRecord(entry, tuple(threads), (start, end)))
        elif self.profile:
            q = self._query_next
            if q + 2 > self.MAX_QUERIES:
                raise RuntimeError("profiling query pool exhausted; call wait()")
            kernel.dispatch(
                thread_count=threads,
                command_encoder=enc,
                query_pool=self._query_pool,
                query_index_before=q,
                query_index_after=q + 1,
                **args,
            )
            self._records.append(DispatchRecord(entry, tuple(threads), q))
            self._query_next += 2
        else:
            kernel.dispatch(thread_count=threads, command_encoder=enc, **args)
        self.stats.dispatches += 1

    def dispatch_flat(self, module, entry, items, /, **params):
        """1-D launch over `items` threads folded into rows of ROW_LEN; the
        kernel indexes q = tid.y * row_len + tid.x and bounds-checks q."""
        if items <= ROW_LEN:
            threads, row_len = (items, 1), items
        else:
            threads, row_len = (ROW_LEN, -(-items // ROW_LEN)), ROW_LEN
        self.dispatch(module, entry, threads, row_len=row_len, **params)

    def copy(self, src, dst):
        self._enc().copy_buffer(dst.buffer, 0, src.buffer, 0, src.nbytes)

    def clone(self, src):
        dst = self.empty(src.shape, src.dtype)
        self.copy(src, dst)
        return dst

    def clear_uint(self, arr):
        self._enc().clear_buffer(arr.buffer)

    def flush(self):
        """Submit recorded work, ordered after (and before) torch's current
        CUDA stream work on GPU backends."""
        if self._encoder is None:
            return
        cmd = self._encoder.finish()
        self._encoder = None
        self.stats.submits += 1
        if self.transfer == "host":
            self.device.submit_command_buffer(cmd)
        elif self.backend == "cuda":
            stream = torch.cuda.current_stream().cuda_stream
            self.device.submit_command_buffer(
                cmd, cuda_stream=spy.NativeHandle(spy.NativeHandleType.CUstream, stream)
            )
        else:
            stream = torch.cuda.current_stream().cuda_stream
            self.device.sync_to_cuda(stream)
            self.device.submit_command_buffer(cmd)
            self.device.sync_to_device(stream)

    def wait(self):
        """Block until all submitted work (and torch's stream) is done."""
        self.flush()
        self.device.wait()
        if self.transfer == "cuda":
            torch.cuda.current_stream().synchronize()
        if self.profile and self._records:
            seconds = None
            if self._query_next:
                seconds = self._query_pool.get_timestamp_results(0, self._query_next)
            for rec in self._records:
                if isinstance(rec.query, tuple):
                    ms = rec.query[0].elapsed_time(rec.query[1])
                else:
                    ms = (seconds[rec.query + 1] - seconds[rec.query]) * 1e3
                self.stats.per_kernel_ms[rec.name] += ms
                self.stats.per_kernel_calls[rec.name] += 1
            self._records.clear()
            self._query_next = 0
            self._query_pool.reset()

    def close(self):
        """Drop every device object this context owns, then the device.
        DeviceArrays still held by callers are invalid afterwards."""
        if self.device is None:
            return
        self.wait()
        self._encoder = None
        self._kernels.clear()
        self._constants.clear()
        self._stage.clear()
        self._free.clear()
        self._query_pool = None
        gc.collect()
        self.device.close()
        self.device = None


NVIDIA_VENDOR_ID = 0x10DE


def resolve_adapter(backend, adapter):
    """AdapterInfo for an index / name substring, or None for the default."""
    if adapter is None:
        return None
    adapters = spy.Device.enumerate_adapters(BACKENDS[backend])
    if isinstance(adapter, int) or (isinstance(adapter, str) and adapter.isdigit()):
        return adapters[int(adapter)]
    matches = [a for a in adapters if adapter.lower() in a.name.lower()]
    if not matches:
        names = ", ".join(a.name for a in adapters)
        raise ValueError(f"no {backend} adapter matches {adapter!r} (have: {names})")
    return matches[0]


_LIVE_CONTEXTS = weakref.WeakSet()


@atexit.register
def _close_contexts():
    """Close contexts before interpreter teardown: a device, or a buffer of
    one, destroyed after torch's CUDA state raises inside sgl and aborts."""
    for ctx in list(_LIVE_CONTEXTS):
        try:
            ctx.close()
        except Exception:  # noqa: BLE001, S110 - teardown order is not ours
            pass


def _pad_bytes(data, capacity):
    raw = np.zeros(capacity, dtype=np.uint8)
    raw[: data.nbytes] = data.reshape(-1).view(np.uint8)
    return raw


def _round_capacity(nbytes):
    """Pool size classes: 256-byte granularity below 1 MiB, 64 KiB above."""
    grain = 256 if nbytes < (1 << 20) else (1 << 16)
    return (nbytes + grain - 1) // grain * grain
