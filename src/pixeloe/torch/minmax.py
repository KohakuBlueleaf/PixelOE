import torch
import torch.nn.functional as F
import numpy as np
from .env import TORCH_COMPILE


@torch.compile(disable=not TORCH_COMPILE)
def dilate_cont(img, kernel, iterations=1):
    # Ensure input has a batch dimension
    squeeze_output = False
    if img.dim() == 3:  # If shape is (C,H,W), add batch dimension
        img = img.unsqueeze(0)
        squeeze_output = True

    N, C, H, W = img.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    # Prepare the kernel by reshaping for patch addition
    kernel_flat = kernel.view(1, 1, kH * kW, 1).to(img.device)

    x = img
    for _ in range(iterations):
        # Extract sliding patches from the image
        patches = F.unfold(x, kernel_size=(kH, kW), stride=1, padding=(pH, pW))
        # Reshape patches to shape (N, C, kH*kW, H*W)
        patches = patches.view(N, C, kH * kW, H * W)
        # Add kernel weights to each patch element
        patches = patches + kernel_flat - 1
        # Perform max pooling over the kernel window to get dilated values
        x_vals = patches.max(dim=2).values
        # Reshape the result back to the image shape
        x = x_vals.view(N, C, H, W)

    # Clamp the output to [0,1] to avoid super-bright values
    x = x.clamp(0, 1)

    # If we added a batch dimension earlier, remove it
    if squeeze_output:
        x = x.squeeze(0)
    return x


@torch.compile(disable=not TORCH_COMPILE)
def erode_cont(img, kernel, iterations=1):
    # Ensure input has a batch dimension
    squeeze_output = False
    if img.dim() == 3:  # If shape is (C,H,W), add batch dimension
        img = img.unsqueeze(0)
        squeeze_output = True

    N, C, H, W = img.shape
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    # Prepare the kernel by reshaping for patch addition
    kernel_flat = kernel.view(1, 1, kH * kW, 1).to(img.device)

    x = img
    for _ in range(iterations):
        # Extract sliding patches from the image
        patches = F.unfold(x, kernel_size=(kH, kW), stride=1, padding=(pH, pW))
        # Reshape patches to shape (N, C, kH*kW, H*W)
        patches = patches.view(N, C, kH * kW, H * W)
        # Add kernel weights to each patch element
        patches = patches - kernel_flat + 1
        # Perform min pooling over the kernel window to get eroded values
        x_vals = patches.min(dim=2).values
        # Reshape the result back to the image shape
        x = x_vals.view(N, C, H, W)

    # If we added a batch dimension earlier, remove it
    if squeeze_output:
        x = x.squeeze(0)
    return x


def circle_kernel(r=3):
    real_r = r
    r = int(r)
    kernel = np.zeros((2 * r + 1, 2 * r + 1))
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            points = np.array([[i, j]] * 8) + np.array(
                [
                    [-0.5, -0.5],
                    [-0.5, 0.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [0, 0.5],
                    [0, -0.5],
                    [0.5, 0],
                    [-0.5, 0],
                ]
            )
            distances = np.linalg.norm(points - r, axis=1)
            max_distance = np.max(distances)
            min_distance = np.min(distances)
            if max_distance <= real_r:
                kernel[i, j] = 1
            elif min_distance <= real_r:
                # a*min + b*max = r, a+b=1, kernel[i, j] = b
                b = (real_r - min_distance) / (max_distance - min_distance)
                kernel[i, j] = b
    return torch.from_numpy(kernel).float()


KERNELS = {
    1: circle_kernel(1),
    2: circle_kernel(1.5),
    3: circle_kernel(2)[1:4, 1:4].contiguous().clone(),
    4: circle_kernel(2.5),
    5: circle_kernel(3)[1:6, 1:6].contiguous().clone(),
    6: circle_kernel(3.5),
}
