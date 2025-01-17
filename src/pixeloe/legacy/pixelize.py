from time import time

import cv2
import numpy as np

from .color import match_color, color_styling, color_quant
from .downscale import downscale_mode
from .outline import outline_expansion, expansion_weight
from .utils import isiterable


def pixelize(
    img,
    mode="contrast",
    target_size=128,
    patch_size=16,
    pixel_size=None,
    thickness=2,
    color_matching=True,
    contrast=1.0,
    saturation=1.0,
    colors=None,
    color_quant_method="kmeans",
    colors_with_weight=False,
    no_upscale=False,
    no_downscale=False,
):
    weighted_color = colors is not None and colors_with_weight
    h, w, _ = img.shape
    if pixel_size is None:
        pixel_size = patch_size

    ratio = w / h
    if isiterable(target_size) and len(target_size) > 1:
        target_org_hw = tuple([int(i * patch_size) for i in target_size][:2])
        ratio = target_org_hw[0] / target_org_hw[1]
        target_org_size = target_org_hw[1]
        target_size = ((target_org_size**2) / (patch_size**2) * ratio) ** 0.5
    else:
        if isiterable(target_size):
            target_size = target_size[0]
        target_org_size = (target_size**2 * patch_size**2 / ratio) ** 0.5
        target_org_hw = (int(target_org_size * ratio), int(target_org_size))

    img = cv2.resize(img, target_org_hw)
    org_img = img.copy()

    if thickness:
        img, weight = outline_expansion(img, thickness, thickness, patch_size, 9, 4)
    elif weighted_color:
        weight = expansion_weight(img, patch_size, (patch_size // 4) * 2, 9, 4)[
            ..., None
        ]
        weight = np.abs(weight * 2 - 1)

    if color_matching:
        img = match_color(img, org_img)

    if no_downscale:
        return img
    img_sm = downscale_mode[mode](img, target_size)

    weight_mat = None
    if weighted_color:
        weight_mat = cv2.resize(
            weight,
            (img_sm.shape[1], img_sm.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        # TODO: How to get more reasonable weight?
        weight_gamma = target_size / 512
        weight_mat = weight_mat**weight_gamma
    if colors is not None:
        img_sm_c = color_quant(
            img_sm,
            colors,
            weight_mat,
            # TODO: How to get more reasonable repeat times?
            int((patch_size * colors) ** 0.5),
            color_quant_method,
        )
        img_sm = match_color(img_sm_c, img_sm, 3)

    if contrast != 1 or saturation != 1:
        img_sm = color_styling(img_sm, saturation, contrast)

    if no_upscale:
        return img_sm

    return cv2.resize(
        img_sm,
        (img_sm.shape[1] * pixel_size, img_sm.shape[0] * pixel_size),
        interpolation=cv2.INTER_NEAREST,
    )


if __name__ == "__main__":
    t0 = time()
    img = cv2.imread("img/house.webp")
    t1 = time()
    img = pixelize(img, target_size=128, patch_size=8)
    t2 = time()
    cv2.imwrite("test.webp", img)
    t3 = time()

    print(f"read time: {t1 - t0:.3f}sec")
    print(f"pixelize time: {t2 - t1:.3f}sec")
    print(f"write time: {t3 - t2:.3f}sec")
