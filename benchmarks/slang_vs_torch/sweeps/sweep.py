"""Scaling sweeps: one parameter varied, everything else at default settings.

Sweeps (default: pixel size 4, thickness 3, contrast downscale, colour match):
  pixels     square input 256 .. 3072 px per side, batch 1
  batch      512², batch 1 .. 16
  pixel_size 1024², p = 2 .. 16
  thickness  1024², thickness 0 .. 6
  colors     1024², k-means palette 8 .. 256 colours (ordered dither)

Each implementation path runs in its own subprocess. Every point records
median / p10 / p90 ms and peak memory (GPU allocator or slang pool; process
peak working set on CPU).

usage: python sweep.py                  run everything -> outputs/bench/sweeps/<stamp>/
       python sweep.py --worker PATH OUT   (internal)
"""

import argparse
import datetime
import gc
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from memory import process_memory

import pixeloe.torch.env as pixeloe_env
from pixeloe.slang.pixelize import pixelize as slang_pixelize
from pixeloe.slang.runtime.registry import create_context
from pixeloe.torch.pixelize import pixelize as torch_pixelize

ROOT = HERE.parents[2]
IMAGE = ROOT / "img" / "snow-leopard.webp"
OUT_ROOT = ROOT / "outputs" / "bench" / "sweeps"

PATHS = {
    "torch-cuda-fp32": {"impl": "torch", "device": "cuda", "dtype": "float32"},
    "torch-cuda-fp16": {"impl": "torch", "device": "cuda", "dtype": "float16"},
    "torch-cuda-fp16-compile": {
        "impl": "torch",
        "device": "cuda",
        "dtype": "float16",
        "compile": True,
    },
    "slang-cuda": {"impl": "slang", "backend": "cuda", "dtype": "float32"},
    "slang-vulkan": {"impl": "slang", "backend": "vulkan", "dtype": "float32"},
    "slang-d3d12": {"impl": "slang", "backend": "d3d12", "dtype": "float32"},
    "torch-cpu-fp32": {"impl": "torch", "device": "cpu", "dtype": "float32"},
    "slang-cpu": {"impl": "slang", "backend": "cpu", "dtype": "float32"},
}


def sweeps(cpu):
    """[(sweep, x value, width, height, batch, kwargs)]"""
    pts = []
    sides = [256, 384, 512, 768, 1024, 1536, 2048, 3072]
    if cpu:
        sides = sides[:6]
    for s in sides:
        pts.append(("pixels", s * s, s, s, 1, {}))
    for b in [1, 2, 4, 8, 16] if not cpu else [1, 2, 4]:
        pts.append(("batch", b, 512, 512, b, {}))
    side = 512 if cpu else 1024
    for p in [2, 3, 4, 6, 8, 12, 16]:
        pts.append(("pixel_size", p, side, side, 1, {"pixel_size": p}))
    for t in range(7):
        pts.append(("thickness", t, side, side, 1, {"thickness": t}))
    for k in [8, 16, 32, 64, 128, 256]:
        pts.append(
            (
                "colors",
                k,
                side,
                side,
                1,
                {"do_quant": True, "num_colors": k, "dither_mode": "ordered"},
            )
        )
    return pts


class PrivateBytesPeak:
    """Peak process private bytes above the value at start, sampled every
    ~1 ms on a thread (Windows keeps no resettable per-interval peak)."""

    def __enter__(self):
        self.base = process_memory().get("private", 0)
        self.peak = self.base
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, process_memory().get("private", 0))
            time.sleep(0.001)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    @property
    def mib(self):
        return (self.peak - self.base) / 2**20


def load(width, height, batch, device, dtype):
    img = Image.open(IMAGE).convert("RGB").resize((width, height), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return (
        t[None]
        .repeat(batch, 1, 1, 1)
        .to(device=device, dtype=getattr(torch, dtype))
        .contiguous()
    )


def worker(path_name, out_file):
    spec = PATHS[path_name]
    cpu = spec.get("device") == "cpu" or spec.get("backend") == "cpu"
    device = "cpu" if cpu else "cuda"
    if spec["impl"] == "torch":
        pixeloe_env.TORCH_COMPILE = bool(spec.get("compile"))
        ctx = None

        def run(x, kw):
            return torch_pixelize(x, **kw)

    else:
        ctx = create_context(spec["backend"])

        def run(x, kw):
            return slang_pixelize(x, context=ctx, **kw)

    def sync():
        if ctx is not None:
            ctx.wait()
        if device == "cuda":
            torch.cuda.synchronize()

    def reset():
        if ctx is not None:
            ctx.wait()
            ctx.trim()
            ctx.stats.reset_peaks()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    with open(out_file, "a") as f:
        for sweep, xval, w, h, b, extra in sweeps(cpu):
            kw = {"pixel_size": 4, "thickness": 3}
            kw.update(extra)
            rec = {
                "path": path_name,
                "sweep": sweep,
                "x": xval,
                "width": w,
                "height": h,
                "batch": b,
                "kwargs": kw,
            }
            try:
                x = load(w, h, b, device, spec["dtype"])
                reset()
                gc.collect()
                with PrivateBytesPeak() as host_mem:
                    t0 = time.perf_counter()
                    run(x, kw)
                    sync()
                    rec["cold_ms"] = (time.perf_counter() - t0) * 1e3
                    run(x, kw)
                    sync()
                    times = []
                    start = time.perf_counter()
                    while len(times) < 30:
                        t0 = time.perf_counter()
                        run(x, kw)
                        sync()
                        times.append((time.perf_counter() - t0) * 1e3)
                        if len(times) >= 3 and time.perf_counter() - start > 0.6:
                            break
                rec["host_mem_mib"] = host_mem.mib
                q = statistics.quantiles(times, n=10) if len(times) > 1 else times * 9
                rec.update(
                    ms=statistics.median(times), p10=q[0], p90=q[-1], iters=len(times)
                )
                if ctx is not None:
                    rec["mem_mib"] = ctx.stats.peak_pool_bytes / 2**20
                elif device == "cuda":
                    rec["mem_mib"] = torch.cuda.max_memory_reserved() / 2**20
                rec["ok"] = True
                del x
            except Exception as exc:  # noqa: BLE001 - recorded, the sweep goes on
                rec["ok"] = False
                rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
                if device == "cuda":
                    torch.cuda.empty_cache()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = f"{rec['ms']:.2f} ms" if rec["ok"] else rec["error"][:120]
            print(f"[{path_name}] {sweep}={xval}: {status}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--run-dir", default="", help="append to an existing run (later points win)"
    )
    a = ap.parse_args()
    if a.worker:
        worker(*a.worker)
        return
    run_dir = (
        Path(a.run_dir)
        if a.run_dir
        else OUT_ROOT / datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    paths = [p for p in PATHS if not a.only or p in a.only.split(",")]
    for p in paths:
        print(f"== {p}", flush=True)
        with open(run_dir / "logs" / f"{p}.log", "w") as log:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    p,
                    str(run_dir / "points.jsonl"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                check=False,
            )
    print(f"sweeps: {run_dir}")


if __name__ == "__main__":
    main()
