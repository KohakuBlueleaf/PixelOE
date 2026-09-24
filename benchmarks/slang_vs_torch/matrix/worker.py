"""Benchmark worker: one implementation path, many (size, case) configs.

Writes one JSON record per config (timings, memory, per-kernel breakdown)
plus the output image as .npy for the cross-path correctness comparison.

usage: python worker.py <path_name> <device_class gpu|cpu> <run_dir>
"""

import gc
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))

from configs import (
    CASES,
    CPU_MATRIX,
    CPU_PATHS,
    GPU_MATRIX,
    GPU_PATHS,
    SIZES,
)
from memory import device_used, process_memory

import pixeloe.torch.env as pixeloe_env
from pixeloe.slang.pixelize import pixelize as slang_pixelize
from pixeloe.slang.runtime.registry import create_context
from pixeloe.torch.pixelize import pixelize as torch_pixelize

ROOT = HERE.parents[2]
IMAGE = ROOT / "img" / "snow-leopard.webp"
MIN_ITERS = 5
MAX_ITERS = 50
TARGET_SECONDS = 1.5
WARMUP = 2


def load_input(width, height, batch, device, dtype):
    img = Image.open(IMAGE).convert("RGB").resize((width, height), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    t = t[None].repeat(batch, 1, 1, 1)
    return t.to(device=device, dtype=getattr(torch, dtype)).contiguous()


class TorchRunner:
    def __init__(self, spec):
        self.spec = spec
        self.device = spec["device"]
        pixeloe_env.TORCH_COMPILE = bool(spec.get("compile"))
        self.fn = torch_pixelize

    def run(self, img, kwargs):
        return self.fn(img, backend="torch", **kwargs)

    def sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def reset_memory(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def memory(self):
        if self.device != "cuda":
            return {}
        return {
            "allocator_peak_allocated": torch.cuda.max_memory_allocated(),
            "allocator_peak_reserved": torch.cuda.max_memory_reserved(),
        }

    def breakdown(self, img, kwargs):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as prof:
            self.run(img, kwargs)
            self.sync()
        rows = []
        for evt in prof.key_averages():
            dev_us = getattr(evt, "device_time_total", 0) or getattr(
                evt, "cuda_time_total", 0
            )
            rows.append(
                {
                    "name": evt.key,
                    "calls": evt.count,
                    "self_cpu_ms": evt.self_cpu_time_total / 1e3,
                    "device_ms": dev_us / 1e3,
                }
            )
        key = "device_ms" if self.device == "cuda" else "self_cpu_ms"
        rows.sort(key=lambda r: -r[key])
        return {"kind": "torch.profiler", "sort_key": key, "ops": rows[:60]}


class SlangRunner:
    def __init__(self, spec):
        self.spec = spec
        self.backend = spec["backend"]
        self.device = "cpu" if self.backend == "cpu" else "cuda"
        self.ctx = create_context(self.backend)
        self.prof_ctx = None
        self.fn = slang_pixelize
        self.extra = {k: spec[k] for k in ("colorfix_blur", "blur_impl") if k in spec}

    def run(self, img, kwargs, ctx=None):
        return self.fn(img, context=ctx or self.ctx, **kwargs, **self.extra)

    def sync(self):
        self.ctx.wait()
        if self.device == "cuda":
            torch.cuda.synchronize()

    def reset_memory(self):
        gc.collect()
        self.ctx.wait()
        self.ctx.trim()
        self.ctx.stats.reset_peaks()
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def memory(self):
        s = self.ctx.stats
        out = {
            "slang_peak_live_bytes": s.peak_live_bytes,
            "slang_peak_pool_bytes": s.peak_pool_bytes,
            "slang_constant_bytes": s.constant_bytes,
            "slang_dispatches_per_call": None,
        }
        if self.device == "cuda":
            out["allocator_peak_allocated"] = torch.cuda.max_memory_allocated()
            out["allocator_peak_reserved"] = torch.cuda.max_memory_reserved()
        return out

    def breakdown(self, img, kwargs):
        if self.prof_ctx is None:
            self.prof_ctx = create_context(self.backend, profile=True)
        ctx = self.prof_ctx
        self.run(img, kwargs, ctx)  # compile + warm the profiling context
        ctx.wait()
        ctx.stats.reset_peaks()
        self.run(img, kwargs, ctx)
        ctx.wait()
        rows = [
            {
                "name": name,
                "calls": ctx.stats.per_kernel_calls[name],
                "device_ms": ms,
            }
            for name, ms in ctx.stats.per_kernel_ms.items()
        ]
        rows.sort(key=lambda r: -r["device_ms"])
        return {
            "kind": "slang timestamps",
            "sort_key": "device_ms",
            "dispatches": ctx.stats.dispatches,
            "submits": ctx.stats.submits,
            "ops": rows,
        }


def time_config(runner, img, kwargs):
    t0 = time.perf_counter()
    out = runner.run(img, kwargs)
    runner.sync()
    cold = time.perf_counter() - t0
    for _ in range(WARMUP):
        runner.run(img, kwargs)
    runner.sync()
    times = []
    start = time.perf_counter()
    while len(times) < MAX_ITERS:
        t0 = time.perf_counter()
        runner.run(img, kwargs)
        runner.sync()
        times.append(time.perf_counter() - t0)
        if len(times) >= MIN_ITERS and time.perf_counter() - start > TARGET_SECONDS:
            break
    times_ms = [t * 1e3 for t in times]
    q = statistics.quantiles(times_ms, n=10) if len(times_ms) >= 2 else times_ms * 9
    return out, {
        "cold_ms": cold * 1e3,
        "iters": len(times_ms),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "p10_ms": q[0],
        "p90_ms": q[-1],
        "stdev_ms": statistics.pstdev(times_ms),
        "all_ms": times_ms,
    }


def main():
    path_name, device_class, run_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    paths = GPU_PATHS if device_class == "gpu" else CPU_PATHS
    matrix = GPU_MATRIX if device_class == "gpu" else CPU_MATRIX
    spec = paths[path_name]
    out_dir = run_dir / "runs" / path_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")
    base_device = device_used()
    base_proc = process_memory()
    t0 = time.perf_counter()
    runner = (TorchRunner if spec["impl"] == "torch" else SlangRunner)(spec)
    init_ms = (time.perf_counter() - t0) * 1e3
    after_init_device = device_used()

    device = runner.device
    only = spec.get("cases")
    for size_name, case_names in matrix.items():
        width, height, pixel_size, batch = SIZES[size_name]
        for case_name in case_names:
            if only is not None and case_name not in only:
                continue
            record = {
                "path": path_name,
                "spec": spec,
                "size": size_name,
                "case": case_name,
                "width": width,
                "height": height,
                "pixel_size": pixel_size,
                "batch": batch,
                "init_ms": init_ms,
                "device_used_baseline": base_device,
                "device_used_after_init": after_init_device,
                "process_baseline": base_proc,
            }
            kwargs = {"pixel_size": pixel_size, "thickness": 3}
            kwargs.update(CASES[case_name])
            record["kwargs"] = kwargs
            try:
                img = load_input(width, height, batch, device, spec["dtype"])
                runner.reset_memory()
                dev_before = device_used()
                out, timing = time_config(runner, img, kwargs)
                record["timing"] = timing
                record["memory"] = runner.memory()
                record["memory"]["device_used_before"] = dev_before
                record["memory"]["device_used_after"] = device_used()
                record["memory"]["process"] = process_memory()
                arr = out.float().cpu().numpy()
                np.save(out_dir / f"{size_name}__{case_name}.npy", arr[:1])
                record["output_shape"] = list(arr.shape)
                record["breakdown"] = runner.breakdown(img, kwargs)
                record["ok"] = True
                del out, img
            except Exception as exc:  # noqa: BLE001 - recorded per config
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
            with open(out_dir / f"{size_name}__{case_name}.json", "w") as f:
                json.dump(record, f, indent=1)
            status = (
                f"{record['timing']['median_ms']:.3f} ms"
                if record["ok"]
                else record["error"][:200]
            )
            print(f"[{path_name}] {size_name}/{case_name}: {status}", flush=True)


if __name__ == "__main__":
    main()
