"""Slang-only device comparison: CPU worker counts vs RTX 4090 vs Arc Pro B50.

Two sweeps at default settings (p = 4, thickness 3, contrast, colour match):
  threads     1024² input; CPU backend at 1 .. 172 OpenMP workers, plus every
              GPU backend once (drawn as reference levels)
  resolution  256² .. 2048²; each GPU backend and the CPU at 1 / 16 / 172
              workers (1 worker stops at 1024²)
Every device runs in its own subprocess. GPU times are end-to-end from a
torch tensor: CUDA-interop on the 4090, host-staged upload / read-back on
the B50.

usage: python devices.py   -> outputs/bench/devices/<stamp>/points.jsonl
"""

import argparse
import datetime
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pixeloe.torch.env as pixeloe_env
from pixeloe.slang.pixelize import pixelize
from pixeloe.slang.runtime.registry import create_context

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "img" / "snow-leopard.webp"
GPUS = {
    "4090 CUDA": "cuda",
    "4090 Vulkan": "vulkan",
    "4090 D3D12": "d3d12",
    "B50 D3D12": "d3d12:B50",
    "B50 Vulkan": "vulkan:B50",
}
THREADS = [1, 2, 4, 8, 16, 32, 48, 64, 86, 128, 172]
SIDES = [256, 512, 768, 1024, 1536, 2048]
CPU_RES = {1: 1024, 16: 2048, 172: 2048}  # workers -> largest side measured


def load(side, device):
    img = Image.open(IMAGE).convert("RGB").resize((side, side), Image.BICUBIC)
    t = (torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255)[None]
    return t.to(device)


def measure(ctx, x, target_s=1.0, max_iters=20):
    kw = {"pixel_size": 4, "thickness": 3}
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    pixelize(x, context=ctx, **kw)
    sync()
    times = []
    start = time.perf_counter()
    while len(times) < max_iters:
        t0 = time.perf_counter()
        pixelize(x, context=ctx, **kw)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
        if len(times) >= 3 and time.perf_counter() - start > target_s:
            break
    q = statistics.quantiles(times, n=10) if len(times) > 1 else times * 9
    return {
        "ms": statistics.median(times),
        "p10": q[0],
        "p90": q[-1],
        "iters": len(times),
    }


def emit(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["sweep"], rec["device"], rec.get("x"), f"{rec['ms']:.2f} ms", flush=True)


def run_gpu(name, out):
    ctx = create_context(GPUS[name])
    dev = "cuda" if ctx.uses_cuda_stream else "cpu"
    emit(
        out,
        dict(
            sweep="threads",
            device=name,
            x=None,
            adapter=ctx.adapter_name,
            **measure(ctx, load(1024, dev)),
        ),
    )
    for side in SIDES:
        emit(
            out,
            dict(
                sweep="resolution",
                device=name,
                x=side * side,
                **measure(ctx, load(side, dev)),
            ),
        )


def run_cpu(out):
    x1024 = load(1024, "cpu")
    for t in THREADS:
        emit(
            out,
            dict(
                sweep="threads",
                device="CPU",
                x=t,
                **measure(create_context(f"cpu:t{t}"), x1024),
            ),
        )
    for workers, max_side in CPU_RES.items():
        ctx = create_context(f"cpu:t{workers}")
        for side in SIDES:
            if side <= max_side:
                emit(
                    out,
                    dict(
                        sweep="resolution",
                        device=f"CPU {workers}t",
                        x=side * side,
                        **measure(ctx, load(side, "cpu")),
                    ),
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("DEVICE", "OUT"))
    a = ap.parse_args()
    pixeloe_env.TORCH_COMPILE = False
    if a.worker:
        name, out = a.worker
        run_cpu(Path(out)) if name == "CPU" else run_gpu(name, Path(out))
        return
    run_dir = (
        ROOT
        / "outputs"
        / "bench"
        / "devices"
        / datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True)
    out = run_dir / "points.jsonl"
    for name in list(GPUS) + ["CPU"]:
        print(f"== {name}", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--worker", name, str(out)],
            cwd=ROOT,
            check=False,
        )
    print(f"devices: {run_dir}")


if __name__ == "__main__":
    main()
