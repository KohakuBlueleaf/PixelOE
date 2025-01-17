import torch
import torch.nn.functional as F
from kornia.color import lab_to_rgb, rgb_to_lab

from ..utils import compile_wrapper


@compile_wrapper
def find_pixel_luminance(chunk):
    mid_idx = chunk.shape[2] // 2
    mid = chunk[:, :, mid_idx].unsqueeze(2)
    med = chunk.median(dim=2).values.unsqueeze(2)
    mu = chunk.mean(dim=2, keepdim=True)
    maxi = chunk.max(dim=2).values.unsqueeze(2)
    mini = chunk.min(dim=2).values.unsqueeze(2)

    out = mid.clone()
    cond1 = (med < mu) & ((maxi - med) > (med - mini))
    cond2 = (med > mu) & ((maxi - med) < (med - mini))

    out[cond1] = mini[cond1]
    out[cond2] = maxi[cond2]
    return out


@compile_wrapper
def contrast_downscale(img, patch_size=8):
    """
    Contrast-based downscaling of an image using unfold to process patches concurrently.
    """
    N, _, H, W = img.shape
    patch_h = patch_size
    patch_w = patch_size
    out_h = H // patch_h

    lab = rgb_to_lab(img)  # [B,3,H,W]
    L, A, B = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    # Unfold channels into patches
    patches_l = F.unfold(L, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches_a = F.unfold(A, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches_b = F.unfold(B, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

    # Reshape to [N, patch_area] where N = out_h*out_w
    patches_l = patches_l.transpose(1, 2)  # [B, N, patch_h*patch_w]
    patches_a = patches_a.transpose(1, 2)  # [B, N, patch_h*patch_w]
    patches_b = patches_b.transpose(1, 2)  # [B, N, patch_h*patch_w]

    # Process luminance concurrently across patches
    result_l = find_pixel_luminance(patches_l)  # [B, patch**2, N]
    # Compute median for A and B channels concurrently
    result_a = patches_a.median(dim=2).values  # [B, patch**2, N]
    result_b = patches_b.median(dim=2).values  # [B, patch**2, N]

    # Reshape results to [B,1,out_h,out_w]
    result_l = result_l.reshape(N, 1, out_h, -1)
    result_a = result_a.reshape(N, 1, out_h, -1)
    result_b = result_b.reshape(N, 1, out_h, -1)

    out_lab = torch.cat([result_l, result_a, result_b], dim=1)  # [B,3,out_h,out_w]
    out_rgb = lab_to_rgb(out_lab)  # [B,3,out_h,out_w]
    return out_rgb  # [B,3,out_h,out_w]
