"""Where do slang and torch k-centroid outputs differ, and why?

For the example image: classify every output block as
  tie      final 2-means clusters have equal size (both SSEs equal exactly)
  non-tie  otherwise
and count differing blocks per class. Also compares torch-CUDA with
torch-CPU on the same input, and writes a sheet
  torch-cuda | torch-cpu | slang | tie mask (white) | slang-vs-torch diff
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pixeloe.slang.pixelize import pixelize as slang_pixelize
from pixeloe.slang.runtime.registry import get_context
from pixeloe.torch import env
from pixeloe.torch.downscale.k_centroid import k_centroid_preprocess
from pixeloe.torch.pixelize import pixelize as torch_pixelize
from pixeloe.torch.utils import batched_kmeans_iter, pre_resize, to_numpy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "slang_examples"


def tie_mask(expanded, p):
    b, c, h, w = expanded.shape
    patches, _ = k_centroid_preprocess(expanded, b, c, h, w, h // p, w // p)
    maxv = patches.max(dim=1, keepdim=True).values
    minv = patches.min(dim=1, keepdim=True).values
    interp = torch.linspace(0, 1, 2, device=expanded.device)[None, :, None]
    cent = interp * minv + (1 - interp) * maxv
    data = patches.unsqueeze(2)
    for _ in range(4):
        labels = ((data - cent.unsqueeze(1)) ** 2).sum(-1).argmin(-1)
        cent, diff = batched_kmeans_iter(data, cent)
        if diff < 1 / 256:
            break
    return (labels.sum(1) * 2 == labels.shape[1]).reshape(h // p, w // p)


def main():
    env.TORCH_COMPILE = False
    name = sys.argv[1] if len(sys.argv) > 1 else "house.webp"
    p = 4
    kw = {
        "pixel_size": p,
        "thickness": 3,
        "mode": "k_centroid",
        "no_post_upscale": True,
    }
    x = pre_resize(Image.open(ROOT / "img" / name).convert("RGB"), 256, p).cuda()
    down_cuda, expanded, _ = torch_pixelize(x, return_intermediate=True, **kw)
    down_cpu = torch_pixelize(x.cpu(), **kw).cuda()
    down_slang = slang_pixelize(x, context=get_context("cuda"), **kw)
    tie = tie_mask(expanded, p)

    def differ(a, b):
        return (a - b).abs().amax(1)[0] > 1e-4

    d_slang = differ(down_slang, down_cuda)
    d_cpu = differ(down_cpu, down_cuda)
    n = tie.numel()
    print(
        f"{name}: {n} blocks, ties {int(tie.sum())} ({tie.float().mean() * 100:.1f}%)"
    )
    print(
        f"slang vs torch-cuda differ: {int(d_slang.sum())} blocks, "
        f"of which ties {int((d_slang & tie).sum())}, non-ties {int((d_slang & ~tie).sum())}"
    )
    print(
        f"torch-cpu vs torch-cuda differ: {int(d_cpu.sum())} blocks, "
        f"of which ties {int((d_cpu & tie).sum())}, non-ties {int((d_cpu & ~tie).sum())}"
    )

    def up(t):
        return to_numpy(F.interpolate(t, scale_factor=p, mode="nearest-exact"))[0]

    mask = np.repeat(np.repeat(tie.cpu().numpy(), p, 0), p, 1)
    mask_img = np.stack([mask * 255] * 3, -1).astype(np.uint8)
    diff = np.abs(up(down_slang).astype(int) - up(down_cuda).astype(int))
    diff_img = np.clip(diff * 4, 0, 255).astype(np.uint8)
    tiles = [up(down_cuda), up(down_cpu), up(down_slang), mask_img, diff_img]
    sheet = np.concatenate(tiles, axis=1)
    path = OUT / f"{Path(name).stem}__k_centroid__ties.png"
    Image.fromarray(sheet).save(path)
    print("sheet:", path)


if __name__ == "__main__":
    main()
