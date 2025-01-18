import torch
import torch.nn.functional as F
import numpy as np
from .utils import compile_wrapper


@compile_wrapper
def dilate_cont(img, kernel, iterations=1):
    # Ensure input has a batch dimension
    squeeze_output = False
    if img.dim() == 3:  # If shape is (c,h,w), add batch dimension
        img = img.unsqueeze(0)
        squeeze_output = True

    n, c, h, w = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Prepare the kernel by reshaping for patch addition
    kernel_flat = kernel.reshape(1, 1, kh * kw, 1)

    x = img
    for _ in range(iterations):
        # Extract sliding patches from the image
        patches = F.unfold(x, kernel_size=(kh, kw), stride=1, padding=(ph, pw))
        # Reshape patches to shape (n, c, kh*kw, h*w)
        patches = patches.reshape(n, c, kh * kw, h * w)
        # Add kernel weights to each patch element
        patches = patches + kernel_flat - 1
        # Perform max pooling over the kernel window to get dilated values
        x_vals = patches.max(dim=2).values
        # Reshape the result back to the image shape
        x = x_vals.reshape(n, c, h, w)

    # clamp the output to [0,1] to avoid super-bright values
    x = x.clamp(0, 1)

    # If we added a batch dimension earlier, remove it
    if squeeze_output:
        x = x.squeeze(0)
    return x


@compile_wrapper
def erode_cont(img, kernel, iterations=1):
    # Ensure input has a batch dimension
    squeeze_output = False
    if img.dim() == 3:  # If shape is (c,h,w), add batch dimension
        img = img.unsqueeze(0)
        squeeze_output = True

    n, c, h, w = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Prepare the kernel by reshaping for patch addition
    kernel_flat = kernel.reshape(1, 1, kh * kw, 1)

    x = img
    for _ in range(iterations):
        # Extract sliding patches from the image
        patches = F.unfold(x, kernel_size=(kh, kw), stride=1, padding=(ph, pw))
        # Reshape patches to shape (n, c, kh*kw, h*w)
        patches = patches.reshape(n, c, kh * kw, h * w)
        # Add kernel weights to each patch element
        patches = patches - kernel_flat + 1
        # Perform min pooling over the kernel window to get eroded values
        x_vals = patches.min(dim=2).values
        # Reshape the result back to the image shape
        x = x_vals.reshape(n, c, h, w)

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
