"""Pixelize an image with the Slang backend.

usage: python -m pixeloe.slang.cli INPUT OUTPUT [--backend cuda|d3d12|vulkan|cpu]
       [--target-size 256] [--pixel-size 4] [--thickness 3] [--mode contrast]
       [--colors 0] [--dither ordered] [--no-upscale]
"""

import argparse
import time

import torch
from PIL import Image

from ..torch.utils import pre_resize, to_numpy
from .pixelize import pixelize
from .runtime.registry import get_context


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--backend", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument("--pixel-size", type=int, default=4)
    p.add_argument("--thickness", type=int, default=3)
    p.add_argument("--mode", default="contrast")
    p.add_argument("--colors", type=int, default=0, help="0 = no quantization")
    p.add_argument("--dither", default="ordered")
    p.add_argument("--no-upscale", action="store_true")
    p.add_argument("--local-stats", default="lattice", choices=["lattice", "sliding"])
    a = p.parse_args()

    img = Image.open(a.input).convert("RGB")
    x = pre_resize(img, target_size=a.target_size, patch_size=a.pixel_size)
    if a.backend != "cpu":
        x = x.cuda()
    ctx = get_context(a.backend)
    t0 = time.perf_counter()
    out = pixelize(
        x,
        pixel_size=a.pixel_size,
        thickness=a.thickness,
        mode=a.mode,
        do_quant=a.colors > 0,
        num_colors=max(a.colors, 2),
        dither_mode=a.dither,
        no_post_upscale=a.no_upscale,
        local_stats=a.local_stats,
        context=ctx,
    )
    Image.fromarray(to_numpy(out)[0]).save(a.output)
    print(
        f"{a.output}  ({a.backend}, {(time.perf_counter() - t0) * 1e3:.1f} ms incl. first-call compile)"
    )


if __name__ == "__main__":
    main()
