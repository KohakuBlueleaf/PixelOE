"""One warmed-up Slang pixelize call on the CUDA backend inside a CUDA
profiler range, for Nsight Compute (`ncu --profile-from-start off`).

usage: ncu --profile-from-start off --set full -o <report> \
           python ncu_driver.py [--width 1920 --height 1080] [--kw '{"thickness": 3}']
"""

import argparse
import json
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
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--kw", default="{}")
    a = ap.parse_args()
    pixeloe_env.TORCH_COMPILE = False
    kw = {"pixel_size": 4, "thickness": 3, **json.loads(a.kw)}
    img = Image.open(IMAGE).convert("RGB").resize((a.width, a.height), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    x = x[None].contiguous().cuda()
    ctx = create_context("cuda")
    for _ in range(3):
        pixelize(x, context=ctx, **kw)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    pixelize(x, context=ctx, **kw)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
