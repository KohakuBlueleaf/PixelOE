"""cProfile of the host side of one Slang pixelize call (Python + binding).

usage: python host_profile.py [--backend cuda] [--size 1920x1080]
"""

import argparse
import cProfile
import pstats
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
    ap.add_argument("--calls", type=int, default=200)
    a = ap.parse_args()
    pixeloe_env.TORCH_COMPILE = False
    w, h = (int(v) for v in a.size.split("x"))
    img = Image.open(IMAGE).convert("RGB").resize((w, h), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    ctx = create_context(a.backend)
    x = x[None].contiguous().to("cuda" if ctx.uses_cuda_stream else "cpu")
    kw = {"pixel_size": 4, "thickness": 3}
    for _ in range(20):
        pixelize(x, context=ctx, **kw)
    torch.cuda.synchronize()
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(a.calls):
        pixelize(x, context=ctx, **kw)
        torch.cuda.synchronize()
    prof.disable()
    stats = pstats.Stats(prof)
    stats.sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    main()
