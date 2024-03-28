from functools import partial
from time import time

import numpy as np
import cv2

from .outline import outline_expansion
from .color import match_color, color_styling
from .utils import apply_chunk


def pixelize(
    img,
    target_size=128,
    patch_size=16,
    thickness=2,
    color_matching=True,
    contrast=1.0,
    saturation=1.0,
    colors=None,
    no_upscale=False,
):
    H, W, C = img.shape

    ratio = W / H
    target_pixel_count = (target_size**2 * patch_size**2 / ratio) ** 0.5
    target_size = (target_size**2 / ratio) ** 0.5
    img = cv2.resize(img, (int(target_pixel_count * ratio), int(target_pixel_count)))
    org_img = img.copy()

    if thickness:
        img = outline_expansion(img, thickness, thickness, patch_size, 9, 4)

    if color_matching:
        img = match_color(img, org_img)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
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
        partial(np.median, axis=1, keepdims=True),
    )
    img_lab[:, :, 2] = apply_chunk(
        img_lab[:, :, 2],
        patch_size,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    img_sm = cv2.resize(
        img,
        (int(target_size * ratio), int(target_size)),
        interpolation=cv2.INTER_NEAREST,
    )

    if contrast != 1.0 or saturation != 1.0:
        img_sm = color_styling(img_sm, saturation, contrast)

    if no_upscale:
        return img_sm

    img_lg = cv2.resize(img_sm, (W, H), interpolation=cv2.INTER_NEAREST)
    return img_lg


if __name__ == "__main__":
    t0 = time()
    img = cv2.imread("img/test.png")
    t1 = time()
    img = pixelize(img, 256, patch_size=8)
    t2 = time()
    cv2.imwrite("img/test2.png", img)
    t3 = time()

    print(f"read time: {t1 - t0:.3f}sec")
    print(f"pixelize time: {t2 - t1:.3f}sec")
    print(f"write time: {t3 - t2:.3f}sec")
