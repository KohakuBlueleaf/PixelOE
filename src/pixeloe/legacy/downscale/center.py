from functools import partial

import numpy as np
import cv2
import torch

from ..utils import apply_chunk_torch as apply_chunk


def center_downscale(
    img,
    target_size=128,
):
    h, w, _ = img.shape

    ratio = w / h
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    patch_size = max(int(round(h // target_hw[1])), int(round(w // target_hw[0])))

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab[:, :, 0] = apply_chunk(
        img_lab[:, :, 0],
        patch_size,
        patch_size,
        lambda x: x[..., x.shape[-1] // 2][..., None],
    )
    img_lab[:, :, 1] = apply_chunk(
        img_lab[:, :, 1],
        patch_size,
        patch_size,
        lambda x: torch.median(x, dim=1, keepdims=True).values,
    )
    img_lab[:, :, 2] = apply_chunk(
        img_lab[:, :, 2],
        patch_size,
        patch_size,
        lambda x: torch.median(x, dim=1, keepdims=True).values,
    )
    img = cv2.cvtColor(img_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_NEAREST)
    return img_sm
