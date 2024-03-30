from functools import partial

import numpy as np
import cv2

from ..utils import apply_chunk


def find_pixel(chunks):
    mid = chunks[..., chunks.shape[-1] // 2][..., np.newaxis]
    med = np.median(chunks, axis=1, keepdims=True)
    mu = np.mean(chunks, axis=1, keepdims=True)
    maxi = np.max(chunks, axis=1, keepdims=True)
    mini = np.min(chunks, axis=1, keepdims=True)

    output = mid
    mini_loc = (med < mu) & (maxi - med > med - mini)
    maxi_loc = (med > mu) & (maxi - med < med - mini)

    output[mini_loc] = mini[mini_loc]
    output[maxi_loc] = maxi[maxi_loc]

    return output


def contrast_based_downscale(
    img,
    target_size=128,
):
    H, W, C = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    patch_size = max(int(round(H // target_hw[1])), int(round(W // target_hw[0])))

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab[:, :, 0] = apply_chunk(img_lab[:, :, 0], patch_size, patch_size, find_pixel)
    img_lab[:, :, 1] = apply_chunk(
        img_lab[:, :, 1],
        patch_size,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img_lab[:, :, 2] = apply_chunk(
        img_lab[:, :, 2],
        patch_size * 2,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img = cv2.cvtColor(img_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_NEAREST)
    return img_sm
