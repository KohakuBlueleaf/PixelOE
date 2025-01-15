import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_lab, lab_to_rgb


def match_color(source, target, level=5):
    source_lab = rgb_to_lab(source)
    target_lab = rgb_to_lab(target)
    result = (source_lab - torch.mean(source_lab)) / torch.std(source_lab)
    result = result * torch.std(target_lab) + torch.mean(target_lab)
    source = lab_to_rgb(result)
    source = wavelet_colorfix(source, target, level)
    return source


def wavelet_colorfix(
    inp: torch.Tensor, target: torch.Tensor, level: int = 5
) -> torch.Tensor:
    """
    Perform wavelet-based color transfer between input and target images.

    Args:
        inp (torch.Tensor): Input tensor of shape (B, C, H, W)
        target (torch.Tensor): Target tensor of shape (B, C, H, W)
        level (int): Number of wavelet decomposition levels

    Returns:
        torch.Tensor: Color-transferred result
    """

    @lru_cache(maxsize=32)
    def gaussian_kernel(radius: int, device: torch.device) -> torch.Tensor:
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel_1d = torch.exp(-x.pow(2) / (2 * radius * radius))
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        return kernel_2d / kernel_2d.sum()

    def wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
        # Create Gaussian kernel
        kernel = gaussian_kernel(radius, x.device)

        # Expand kernel for each input channel
        kernel = kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1))

        # Pad input for same size output
        pad_size = radius
        x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

        # Apply convolution for each batch and channel
        return F.conv2d(x_pad, kernel.to(x), groups=x.size(1))

    # Ensure inputs are on the same device
    target = target.to(inp)

    # Initialize high frequency components
    high_freq = torch.zeros_like(inp)
    x = inp.clone()

    # Perform wavelet decomposition
    for i in range(1, level + 1):
        radius = 2**i
        low_freq = wavelet_blur(x, radius)
        high_freq = high_freq + (x - low_freq)
        x = low_freq

    # Get target's low frequency components
    target_x = target.clone()
    for i in range(1, level + 1):
        radius = 2**i
        target_x = wavelet_blur(target_x, radius)

    # Combine high frequency from input with low frequency from target
    return high_freq + target_x


def color_quantization_kmeans(img, K=32, max_iter=10):
    """
    Naive k-means color quantization on an image tensor.
    """
    pixels = img.permute(1, 2, 0).reshape(-1, 3)
    centroids = pixels[torch.randperm(pixels.shape[0])[:K]]

    for _ in range(max_iter):
        dists = (pixels.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=2)
        labels = dists.argmin(dim=1)
        new_centroids = []
        for c in range(K):
            cluster_points = pixels[labels == c]
            if cluster_points.shape[0] > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[c])
        new_centroids = torch.stack(new_centroids, dim=0)
        diff = (new_centroids - centroids).abs().sum()
        centroids = new_centroids
        if diff < 1e-5:
            break

    quant_pixels = centroids[labels]
    return quant_pixels.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
