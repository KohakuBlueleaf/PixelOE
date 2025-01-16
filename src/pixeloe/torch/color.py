import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_lab, lab_to_rgb

from .env import TORCH_COMPILE


@torch.compile(disable=not TORCH_COMPILE)
def match_color(source, target, level=5):
    source_lab = rgb_to_lab(source)
    target_lab = rgb_to_lab(target)
    result = (source_lab - torch.mean(source_lab)) / torch.std(source_lab)
    result = result * torch.std(target_lab) + torch.mean(target_lab)
    source = lab_to_rgb(result)
    source = wavelet_colorfix(source, target, level)
    return source


@lru_cache(maxsize=32)
def gaussian_kernel(radius: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float16, device=device)
    kernel_1d = torch.exp(-x.pow(2) / (2 * radius * radius))
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    return kernel_2d / kernel_2d.sum()


@torch.compile(disable=not TORCH_COMPILE)
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


@torch.compile(disable=not TORCH_COMPILE)
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


@torch.compile(disable=not TORCH_COMPILE)
def color_quantization_kmeans(img, K=32, max_iter=10):
    """
    Naive k-means color quantization on an image tensor.
    Returns both quantized image and centroids for later use.
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
    return (
        quant_pixels.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1),
        centroids,
        labels,
    )


def find_nearest_palette_color(pixel, palette):
    """
    Find the nearest color in the palette for a given pixel.
    """
    dists = (palette.unsqueeze(0) - pixel.unsqueeze(1)).pow(2).sum(dim=2)
    return palette[dists.argmin(dim=1)]


def find_nearest_palette_colors_with_distance(pixel, palette):
    """
    Find the two nearest colors in the palette for a given pixel and return them
    along with the relative distance to the first color.
    """
    dists = (palette.unsqueeze(0) - pixel.unsqueeze(1)).pow(2).sum(dim=2)
    sorted_indices = dists.argsort(dim=1)
    closest_two_indices = sorted_indices[:, :2]
    closest_two_colors = palette[closest_two_indices]

    # Calculate interpolation factor based on distances
    closest_two_dists = torch.gather(dists, 1, closest_two_indices)
    dist_ratio = closest_two_dists[:, 0] / (
        closest_two_dists[:, 0] + closest_two_dists[:, 1] + 1e-6
    )
    return closest_two_colors[:, 0], closest_two_colors[:, 1], dist_ratio


@torch.compile(disable=not TORCH_COMPILE)
def error_diffusion_iter(output, height, width, palette, kernel):
    # Find nearest palette colors for current pixels
    current_pixels = output[0].permute(1, 2, 0).reshape(-1, 3)
    quantized_pixels = find_nearest_palette_color(current_pixels, palette)
    quantized = quantized_pixels.reshape(height, width, 3).permute(2, 0, 1)

    # Calculate error
    error = output[0] - quantized

    # Diffuse error using convolution
    error_padded = F.pad(error.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    diffused_error = F.conv2d(
        error_padded,
        kernel.view(1, 1, 2, 3).repeat(3, 1, 1, 1),
        padding=0,
        groups=3,
    )
    return diffused_error


def parallel_error_diffusion(image, height, width, palette, device):
    # Error diffusion kernel
    kernel = (
        torch.tensor([[0, 0, 7], [3, 5, 1]], dtype=torch.float16, device=device) / 16.0
    )

    # Initialize output tensor
    output = image.clone()

    # Process image in parallel using strided convolutions
    for y in range(0, height, 2):
        diffused_error = error_diffusion_iter(output, height, width, palette, kernel)

        # Update next rows
        if y + 2 < height:
            output[:, :, y + 1 : y + 3] += diffused_error[:, :, y + 1 : y + 3]

    # Final quantization
    final_pixels = output[0].permute(1, 2, 0).reshape(-1, 3)
    final_quantized = find_nearest_palette_color(final_pixels, palette)
    return final_quantized.reshape(height, width, 3).permute(2, 0, 1)


@lru_cache(maxsize=32)
def _generate_bayer_matrix(n, device):
    if n == 2:
        return torch.tensor([[0, 2], [3, 1]], device=device)
    smaller = _generate_bayer_matrix(n // 2, device)
    result = torch.concat(
        [
            torch.concat([4 * smaller, 4 * smaller + 2], dim=1),
            torch.concat([4 * smaller + 3, 4 * smaller + 1], dim=1),
        ],
        dim=0,
    )
    return result


@lru_cache(maxsize=32)
def generate_bayer_matrix(n, device):
    return _generate_bayer_matrix(n, device) / n**2


# @torch.compile(disable=not TORCH_COMPILE)
def ordered_dither(image, height, width, palette, device):
    # ordered dithering
    pattern_size = 8
    bayer = generate_bayer_matrix(pattern_size, device)
    bayer = bayer.repeat(
        (height + pattern_size - 1) // pattern_size,
        (width + pattern_size - 1) // pattern_size,
    )[:height, :width]

    # Reshape image and find two nearest palette colors for each pixel
    pixels = image[0].permute(1, 2, 0).reshape(-1, 3)
    color1, color2, dist_ratio = find_nearest_palette_colors_with_distance(
        pixels, palette
    )

    # Apply threshold matrix to determine color selection
    threshold = bayer.reshape(-1)

    # Choose between color1 and color2 based on threshold and distance ratio
    mask = (threshold > dist_ratio).float().unsqueeze(1)
    output_pixels = color1 * mask + color2 * (1 - mask)
    return output_pixels.reshape(height, width, 3).permute(2, 0, 1)


def parallel_dither_with_palette(image, quantized, palette, method="error_diffusion"):
    """
    Apply dithering using a specific color palette.

    Args:
        image: Input tensor of shape (channels, height, width)
        palette: Tensor of colors to use (K x channels)
        method: 'error_diffusion' or 'ordered'
    """
    device = image.device
    channels, height, width = image.shape
    image = image.unsqueeze(0)  # Add batch dimension

    if method == "error_diffusion":
        output = parallel_error_diffusion(image, height, width, palette, device)
    elif method == "ordered":
        output = ordered_dither(image, height, width, palette, device)
    else:
        output = quantized

    return output


def quantize_and_dither(image, K=32, dither_method="error_diffusion"):
    """
    Combined color quantization and dithering.

    Args:
        image: Input tensor of shape (channels, height, width)
        K: Number of colors in palette
        dither_method: 'error_diffusion' or 'ordered'

    Returns:
        Quantized and dithered image tensor
    """
    # First perform k-means color quantization
    quantized_img, palette, _ = color_quantization_kmeans(image, K=K)

    # Then apply dithering using the generated palette
    dithered_img = parallel_dither_with_palette(
        image, quantized_img, palette, method=dither_method
    )

    return dithered_img
