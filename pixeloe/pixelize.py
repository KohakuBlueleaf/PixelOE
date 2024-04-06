from time import time

import cv2

from .color import match_color, color_styling, kmeans_color_quant
from .downscale import downscale_mode
from .outline import outline_expansion
from .utils import isiterable


def pixelize(
    img,
    mode="contrast",
    target_size=128,
    patch_size=16,
    thickness=2,
    color_matching=True,
    contrast=1.0,
    saturation=1.0,
    colors=None,
    no_upscale=False,
    no_downscale=False,
):
    H, W, _ = img.shape

    ratio = W / H
    if isiterable(target_size):
        target_org_hw = tuple([int(i * patch_size) for i in target_size][:2])
        ratio = target_org_hw[0] / target_org_hw[1]
        target_org_size = target_org_hw[1]
        target_size = ((target_org_size**2) / (patch_size**2) * ratio) ** 0.5
    else:
        target_org_size = (target_size**2 * patch_size**2 / ratio) ** 0.5
        target_org_hw = (int(target_org_size * ratio), int(target_org_size))

    img = cv2.resize(img, target_org_hw)
    org_img = img.copy()

    if thickness:
        img = outline_expansion(img, thickness, thickness, patch_size, 9, 4)

    if color_matching:
        img = match_color(img, org_img)

    if no_downscale:
        return img
    img_sm = downscale_mode[mode](img, target_size)

    if colors is not None:
        img_sm = kmeans_color_quant(img_sm, colors)

    if contrast != 1 or saturation != 1:
        img_sm = color_styling(img_sm, saturation, contrast)

    if no_upscale:
        return img_sm

    img_lg = cv2.resize(img_sm, (W, H), interpolation=cv2.INTER_NEAREST)
    return img_lg


if __name__ == "__main__":
    t0 = time()
    img = cv2.imread("img/house.png")
    t1 = time()
    img = pixelize(img, target_size=128, patch_size=8)
    t2 = time()
    cv2.imwrite("test.png", img)
    t3 = time()

    print(f"read time: {t1 - t0:.3f}sec")
    print(f"pixelize time: {t2 - t1:.3f}sec")
    print(f"write time: {t3 - t2:.3f}sec")
