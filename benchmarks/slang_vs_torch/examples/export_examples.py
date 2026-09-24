"""Export example outputs: torch path vs Slang path, side by side.

Writes outputs/slang_examples/<name>__{torch,slang}.png, a diff image and a
comparison sheet per case.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pixeloe.slang.pixelize import pixelize as slang_pixelize
from pixeloe.slang.runtime.registry import get_context
from pixeloe.torch import env
from pixeloe.torch.pixelize import pixelize as torch_pixelize
from pixeloe.torch.utils import pre_resize, to_numpy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "slang_examples"
CASES = {
    "default": {},
    "k_centroid": {"mode": "k_centroid"},
    "quant32_ordered": {"do_quant": True, "num_colors": 32, "dither_mode": "ordered"},
    "quant16_error_diffusion": {
        "do_quant": True,
        "num_colors": 16,
        "dither_mode": "error_diffusion",
    },
}
IMAGES = ["snow-leopard.webp", "house.webp", "horse-girl.webp"]


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return img


def main():
    env.TORCH_COMPILE = False
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = get_context(backend)
    for name in IMAGES:
        src = Image.open(ROOT / "img" / name).convert("RGB")
        x = pre_resize(src, target_size=256, patch_size=4)
        x = x.cuda() if backend != "cpu" else x
        stem = Path(name).stem
        for case, kw in CASES.items():
            kw = dict(pixel_size=4, thickness=3, **kw)
            ref = to_numpy(torch_pixelize(x, backend="torch", **kw))[0]
            out = to_numpy(slang_pixelize(x, context=ctx, **kw))[0]
            diff = np.abs(ref.astype(int) - out.astype(int))
            Image.fromarray(ref).save(OUT / f"{stem}__{case}__torch.png")
            Image.fromarray(out).save(OUT / f"{stem}__{case}__slang-{backend}.png")
            diff_img = np.clip(diff * 16, 0, 255).astype(np.uint8)
            h, w, _ = ref.shape
            sheet = Image.new("RGB", (w * 3, h))
            sheet.paste(label(Image.fromarray(ref), "torch"), (0, 0))
            sheet.paste(label(Image.fromarray(out), f"slang-{backend}"), (w, 0))
            sheet.paste(label(Image.fromarray(diff_img), "|diff| x16"), (2 * w, 0))
            sheet.save(OUT / f"{stem}__{case}__compare.png")
            print(
                f"{stem:14s} {case:24s} max|diff|={diff.max():3d}/255 "
                f"pixels differing={float((diff.max(-1) > 0).mean()) * 100:.2f}%"
            )
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
