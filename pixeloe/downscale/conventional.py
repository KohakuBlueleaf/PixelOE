import cv2


def nearest(
    img,
    target_size=128,
):
    H, W, C = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_NEAREST)
    return img_sm


def bicubic(
    img,
    target_size=128,
):
    H, W, C = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_CUBIC)
    return img_sm
