"""Single-config timing of the torch pipeline (the reference for quick.py).

usage: python torch_ref.py [--device cuda|xpu|cpu] [--dtype float16] [--compile]
       [--size 1920x1080] [key=value ...]
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
from pixeloe.torch.pixelize import pixelize

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "img" / "snow-leopard.webp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("kw", nargs="*", help="pixelize keyword arguments, key=value")
    a = ap.parse_args()
    pixeloe_env.TORCH_COMPILE = a.compile
    w, h = (int(v) for v in a.size.split("x"))
    kw = {"pixel_size": 4, "thickness": 3, "backend": "torch"}
    for item in a.kw:
        key, _, value = item.partition("=")
        try:
            kw[key] = json.loads(value)
        except json.JSONDecodeError:
            kw[key] = value
    img = Image.open(IMAGE).convert("RGB").resize((w, h), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    x = x[None].contiguous().to(a.device, getattr(torch, a.dtype))
    sync = {
        "cuda": torch.cuda.synchronize,
        "xpu": lambda: torch.xpu.synchronize(),
    }.get(x.device.type, lambda: None)
    for _ in range(3):  # compile
        pixelize(x, **kw)
    sync()
    t_end = time.perf_counter() + 1.0
    while time.perf_counter() < t_end:
        pixelize(x, **kw)
        sync()
    times = []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        pixelize(x, **kw)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"e2e  min {min(times):.3f} ms  median {statistics.median(times):.3f} ms")


if __name__ == "__main__":
    main()
