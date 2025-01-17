import cv2
import numpy as np
from PIL import Image


def nearest(
    img,
    target_size=128,
):
    h, w, _ = img.shape

    ratio = w / h
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img_sm = img.resize(target_hw, Image.NEAREST)
    img_sm = cv2.cvtColor(np.asarray(img_sm), cv2.COLOR_RGB2BGR)
    return img_sm


def bicubic(
    img,
    target_size=128,
):
    h, w, _ = img.shape

    ratio = w / h
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img_sm = img.resize(target_hw, Image.BICUBIC)
    img_sm = cv2.cvtColor(np.asarray(img_sm), cv2.COLOR_RGB2BGR)
    return img_sm
