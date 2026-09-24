"""2x2 grids per example image:  raw | torch  /  slang | slang sliding-window.

outputs/slang_examples/<image>__grid.png (default settings, p=4, thickness 3).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pixeloe.slang.pixelize import pixelize as slang_pixelize
from pixeloe.slang.runtime.registry import get_context
from pixeloe.torch import env
from pixeloe.torch.pixelize import pixelize as torch_pixelize
from pixeloe.torch.utils import pre_resize, to_numpy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "slang_examples"
IMAGES = ["snow-leopard.webp", "house.webp", "horse-girl.webp", "dragon-girl.webp"]


def label(arr, text):
    """Caption readable at the grid's full size: ~4% of the tile height."""
    img = Image.fromarray(arr)
    size = max(16, img.height // 24)
    font = ImageFont.load_default(size=size)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    pad = size // 3
    d.rectangle([0, 0, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad], fill=(0, 0, 0))
    d.text((pad - x0, pad - y0), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    env.TORCH_COMPILE = False
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    ctx = get_context(backend)
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {"pixel_size": 4, "thickness": 3}
    for name in IMAGES:
        src = Image.open(ROOT / "img" / name).convert("RGB")
        x = pre_resize(src, target_size=256, patch_size=4)
        x = x.cuda() if backend != "cpu" else x
        raw = to_numpy(x)[0]
        ref = to_numpy(torch_pixelize(x, **kw))[0]
        lat = to_numpy(slang_pixelize(x, context=ctx, **kw))[0]
        sld = to_numpy(slang_pixelize(x, context=ctx, local_stats="sliding", **kw))[0]
        top = np.concatenate([label(raw, "raw"), label(ref, "torch")], axis=1)
        bottom = np.concatenate(
            [label(lat, f"slang-{backend}"), label(sld, f"slang-{backend} sliding")],
            axis=1,
        )
        path = OUT / f"{Path(name).stem}__grid.png"
        Image.fromarray(np.concatenate([top, bottom], axis=0)).save(path)
        print(path)


if __name__ == "__main__":
    main()
