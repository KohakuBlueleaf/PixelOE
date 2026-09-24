"""Single-config timing for kernel work: e2e (min / median) and per-kernel
device time of one pixelize call.

usage: python quick.py [--backend cuda] [--size 1920x1080] [key=value ...]
"""

import argparse
import json
import statistics
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("kw", nargs="*", help="pixelize keyword arguments, key=value")
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    pixeloe_env.TORCH_COMPILE = False
    w, h = (int(v) for v in a.size.split("x"))
    kw = {"pixel_size": 4, "thickness": 3}
    for item in a.kw:
        key, _, value = item.partition("=")
        try:
            kw[key] = json.loads(value)
        except json.JSONDecodeError:
            kw[key] = value
    img = Image.open(IMAGE).convert("RGB").resize((w, h), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    ctx = create_context(a.backend)
    x = (
        x[None]
        .contiguous()
        .to("cuda" if getattr(ctx, "uses_cuda_stream", False) else "cpu")
    )
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)

    t_end = time.perf_counter() + 1.0
    while time.perf_counter() < t_end:
        pixelize(x, context=ctx, **kw)
        sync()
    times, host = [], []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        pixelize(x, context=ctx, **kw)
        t1 = time.perf_counter()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
        host.append((t1 - t0) * 1e3)
    print(
        f"e2e  min {min(times):.3f} ms  median {statistics.median(times):.3f} ms  "
        f"(host returns after {statistics.median(host):.3f} ms)"
    )

    if hasattr(ctx, "close"):
        ctx.close()
    pctx = create_context(a.backend, profile=True)
    t_end = time.perf_counter() + 1.0
    while time.perf_counter() < t_end:
        pixelize(x, context=pctx, **kw)
        pctx.wait()
    runs = 5
    pctx.stats.reset_peaks()
    for _ in range(runs):
        pixelize(x, context=pctx, **kw)
        pctx.wait()
    rows = sorted(pctx.stats.per_kernel_ms.items(), key=lambda kv: -kv[1])
    total = sum(ms for _, ms in rows) / runs
    print(f"kernels {total:.3f} ms, {pctx.stats.dispatches // runs} dispatches")
    for name, ms in rows:
        calls = pctx.stats.per_kernel_calls[name] // runs
        print(f"  {ms / runs:8.3f} ms  x{calls:<3d} {name}")


if __name__ == "__main__":
    main()
