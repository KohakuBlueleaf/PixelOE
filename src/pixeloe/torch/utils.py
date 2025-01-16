import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image


def to_numpy(tensor):
    """
    Convert a torch.Tensor [C,H,W] with range [0..1] back to a NumPy HWC image [0..255].
    """
    return list(
        (tensor.float().permute(0, 2, 3, 1).cpu().numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )


def pre_resize(
    img_pil,
    target_size=128,
    patch_size=8,
):
    W, H = img_pil.size
    ratio = W / H
    in_h = (target_size**2 / ratio) ** 0.5
    in_w = int(in_h * ratio) * patch_size
    in_h = int(in_h) * patch_size
    img_pil = img_pil.resize((in_w, in_h), Image.Resampling.BICUBIC)
    img_t = to_tensor(img_pil)
    return img_t[None]
