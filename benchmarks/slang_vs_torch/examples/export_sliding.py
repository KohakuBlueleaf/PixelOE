"""Export lattice vs sliding-window outline statistics, side by side.

outputs/slang_examples/<image>__stats__compare.png:
  lattice pixelized | sliding pixelized | lattice weight | sliding weight
plus the timing of both variants.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from pixeloe.slang.pixelize import pixelize
from pixeloe.slang.runtime.registry import get_context
from pixeloe.torch.utils import pre_resize, to_numpy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "slang_examples"
IMAGES = ["snow-leopard.webp", "house.webp", "horse-girl.webp"]


def label(arr, text):
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 7 * len(text) + 8, 14], fill=(0, 0, 0))
    d.text((4, 1), text, fill=(255, 255, 255))
    return np.asarray(img)


def timed(fn, sync):
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(5):
        out = fn()
    sync()
    return out, (time.perf_counter() - t0) / 5 * 1e3


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    ctx = get_context(backend)
    sync = torch.cuda.synchronize if backend != "cpu" else (lambda: None)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in IMAGES:
        src = Image.open(ROOT / "img" / name).convert("RGB")
        x = pre_resize(src, target_size=256, patch_size=4)
        x = x.cuda() if backend != "cpu" else x
        tiles, weights = [], []
        for stats in ("lattice", "sliding"):
            kw = {"pixel_size": 4, "thickness": 3, "local_stats": stats, "context": ctx}
            (out, _, w), ms = timed(
                lambda x=x, kw=kw: pixelize(x, return_intermediate=True, **kw), sync
            )
            tiles.append(label(to_numpy(out)[0], f"{stats} {ms:.1f} ms"))
            wimg = (w[0, 0].float().cpu().numpy() * 255).astype(np.uint8)
            weights.append(label(np.stack([wimg] * 3, -1), f"{stats} weight"))
            print(f"{name:18s} {stats:8s} {ms:7.2f} ms")
        sheet = np.concatenate(tiles + weights, axis=1)
        Image.fromarray(sheet).save(OUT / f"{Path(name).stem}__stats__compare.png")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
