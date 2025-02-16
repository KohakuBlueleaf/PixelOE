import sys

if "pixeloe" in sys.modules:
    sys.modules["pixeloe_nodes"] = sys.modules.pop("pixeloe")

from .installer import install_pixeloe

install_pixeloe()

import torch
from pixeloe.torch.pixelize import pixelize
from pixeloe.torch.outline import outline_expansion
from pixeloe.torch.utils import pre_resize


# Constants
FUNCTION = "execute"
CATEGORY = "utils/pixel"


def image_preprocess(img: torch.Tensor, device: str):
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.size(3) <= 4:
        img = img.permute(0, 3, 1, 2)
        use_channel_last = True
    if img.size(1) == 4:
        img = img[:, :3]
    org_device = img.device
    if device != "default":
        img = img.to(device)
    return img, use_channel_last, org_device


class PixelOE:
    INPUT_TYPES = lambda: {
        "required": {
            "pixel_size": ("INT", {"default": 4, "min": 1, "max": 32}),
            "thickness": ("INT", {"default": 2, "min": 0, "max": 6}),
            "img": ("IMAGE",),
            "mode": (["contrast", "k_centroid", "lanczos", "nearest", "bilinear"],),
            "color_quant": ("BOOLEAN", {"default": False}),
            "no_post_upscale": ("BOOLEAN", {"default": False}),
            "num_colors": ("INT", {"default": 256, "min": 2, "max": 256}),
            "quant_mode": (["kmeans", "weighted-kmeans", "repeat-kmeans"],),
            "dither_mode": (["ordered", "error_diffusion", "none"],),
            "device": (["default", "cpu", "cuda", "mps"],),
        },
    }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "pixel_image",
        "oe_image",
        "oe_weight",
    )
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        pixel_size: int,
        thickness: int,
        img: torch.Tensor,
        mode: str,
        color_quant: bool,
        no_post_upscale: bool,
        num_colors: int,
        quant_mode: str,
        dither_mode: str,
        device: str,
    ):
        img, use_channel_last, org_device = image_preprocess(img, device)
        result, oe_image, oe_weight = pixelize(
            img,
            pixel_size,
            thickness,
            mode,
            do_color_match=True,
            do_quant=color_quant,
            num_colors=num_colors,
            quant_mode=quant_mode,
            dither_mode=dither_mode,
            return_intermediate=True,
        )
        if oe_weight is not None:
            oe_weight = oe_weight.to(org_device).repeat(1, 3, 1, 1)
        else:
            oe_weight = torch.zeros_like(result)
        if oe_image is None:
            oe_image = torch.zeros_like(result)
        result = result.to(org_device)
        oe_image = oe_image.to(org_device)
        if use_channel_last:
            result = result.permute(0, 2, 3, 1)
            oe_image = oe_image.permute(0, 2, 3, 1)
            if oe_weight is not None:
                oe_weight = oe_weight.permute(0, 2, 3, 1)
        return result, oe_image, oe_weight


class OutlineExpansion:
    INPUT_TYPES = lambda: {
        "required": {
            "img": ("IMAGE",),
            "pixel_size": ("INT", {"default": 4, "min": 1, "max": 32}),
            "thickness": ("INT", {"default": 3, "min": 1, "max": 6}),
            "device": (["default", "cpu", "cuda", "mps"],),
        },
    }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = (
        "oe_image",
        "oe_weight",
    )
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        img: torch.Tensor,
        pixel_size: int,
        thickness: int,
        device: str,
    ):
        img, use_channel_last, org_device = image_preprocess(img, device)
        oe_image, oe_weight = outline_expansion(img, thickness, thickness, pixel_size)
        oe_image = oe_image.to(org_device)
        oe_weight = oe_weight.to(org_device).repeat(1, 3, 1, 1)
        if use_channel_last:
            oe_image = oe_image.permute(0, 2, 3, 1)
            oe_weight = oe_weight.permute(0, 2, 3, 1)
        return oe_image, oe_weight


class PreResize:
    INPUT_TYPES = lambda: {
        "required": {
            "img": ("IMAGE",),
            "target_pixels": ("INT", {"default": 256, "min": 1, "max": 1024}),
            "pixel_size": ("INT", {"default": 4, "min": 1, "max": 32}),
            "device": (["default", "cpu", "cuda", "mps"],),
        },
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img",)
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        img: torch.Tensor,
        target_pixels: int,
        pixel_size: int,
        device: str,
    ):
        img, use_channel_last, org_device = image_preprocess(img, device)
        img = pre_resize(img, target_size=target_pixels, patch_size=pixel_size)
        img = img.to(org_device)
        if use_channel_last:
            img = img.permute(0, 2, 3, 1)
        return img


NODE_CLASS_MAPPINGS = {
    "PixelOE": PixelOE,
    "OutlineExpansion": OutlineExpansion,
    "PreResize": PreResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelOE": "PixelOE",
    "OutlineExpansion": "OutlineExpansion",
    "PreResize": "PreResize",
}
