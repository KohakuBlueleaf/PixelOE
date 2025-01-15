import torch
import torch.nn.functional as F
from kornia.color import lab_to_rgb, rgb_to_lab

from ..env import TORCH_COMPILE


@torch.compile(disable=not TORCH_COMPILE)
def find_pixel_luminance(chunk):
    mid_idx = chunk.shape[1] // 2
    mid = chunk[:, mid_idx].unsqueeze(1)
    med = chunk.median(dim=1).values.unsqueeze(1)
    mu = chunk.mean(dim=1, keepdim=True)
    maxi = chunk.max(dim=1).values.unsqueeze(1)
    mini = chunk.min(dim=1).values.unsqueeze(1)

    out = mid.clone()
    cond1 = (med < mu) & ((maxi - med) > (med - mini))
    cond2 = (med > mu) & ((maxi - med) < (med - mini))

    out[cond1[:, 0]] = mini[cond1[:, 0]]
    out[cond2[:, 0]] = maxi[cond2[:, 0]]
    return out


@torch.compile(disable=not TORCH_COMPILE)
def contrast_downscale(img, patch_size=8):
    """
    Contrast-based downscaling of an image using unfold to process patches concurrently.
    """
    H, W = img.shape[1], img.shape[2]
    patch_h = patch_size
    patch_w = patch_size
    out_h = H // patch_h
    out_w = W // patch_w

    lab = rgb_to_lab(img.unsqueeze(0))  # [1,3,H,W]
    L, A, B = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    # Unfold channels into patches
    patches_L = F.unfold(L, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches_A = F.unfold(A, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches_B = F.unfold(B, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

    # Reshape to [N, patch_area] where N = out_h*out_w
    patches_L = patches_L.squeeze(0).transpose(0, 1)  # [N, patch_h*patch_w]
    patches_A = patches_A.squeeze(0).transpose(0, 1)  # [N, patch_h*patch_w]
    patches_B = patches_B.squeeze(0).transpose(0, 1)  # [N, patch_h*patch_w]

    # Process luminance concurrently across patches
    result_L = find_pixel_luminance(patches_L)  # [1, patch**2, N]
    # Compute median for A and B channels concurrently
    result_A = patches_A.median(dim=1).values[:, None]  # [1, patch**2, N]
    result_B = patches_B.median(dim=1).values[:, None]  # [1, patch**2, N]

    # Reshape results to [1,1,out_h,out_w]
    # fold
    result_L = result_L.transpose(0, 1).reshape(1, 1, out_h, -1)
    result_A = result_A.transpose(0, 1).reshape(1, 1, out_h, -1)
    result_B = result_B.transpose(0, 1).reshape(1, 1, out_h, -1)

    out_lab = torch.cat([result_L, result_A, result_B], dim=1)  # [1,3,out_h,out_w]
    out_rgb = lab_to_rgb(out_lab)  # [1,3,out_h,out_w]
    return out_rgb[0]  # [3,out_h,out_w]
