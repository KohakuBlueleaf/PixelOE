"""Process / device memory probes (Windows + CUDA)."""

import ctypes
import ctypes.wintypes as wt
import sys

import torch


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory():
    """(working set, peak working set, private bytes, peak private) in bytes."""
    if sys.platform != "win32":
        return {}
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    # untyped, ctypes truncates the pseudo-handle (-1) and the call fails
    k32.K32GetProcessMemoryInfo.argtypes = [
        wt.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wt.DWORD,
    ]
    k32.K32GetProcessMemoryInfo.restype = wt.BOOL
    if not k32.K32GetProcessMemoryInfo(
        k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {
        "working_set": counters.WorkingSetSize,
        "peak_working_set": counters.PeakWorkingSetSize,
        "private": counters.PagefileUsage,
        "peak_private": counters.PeakPagefileUsage,
    }


def device_used():
    """Bytes in use on CUDA device 0 (all processes, driver view)."""
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(0)
    return total - free
