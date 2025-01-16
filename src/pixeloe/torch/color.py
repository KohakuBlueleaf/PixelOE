import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_lab, lab_to_rgb

from .utils import batched_kmeans_iter
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


def color_quantization_kmeans(img, K=32):
    """
    Naive k-means color quantization on an image tensor.
    Returns both quantized image and centroids for later use.
    """
    B, C, H, W = img.shape
    pixels = img.permute(0, 2, 3, 1).reshape(B, -1, C)

    # Initialize centroids using min-max interpolation
    maxv = pixels.max(dim=1, keepdim=True)
    minv = pixels.min(dim=1, keepdim=True)
    interp = torch.linspace(0, 1, K, device=img.device)[None, :, None]
    centroids = interp * minv.values + (1 - interp) * maxv.values

    pixels = pixels.unsqueeze(2)
    cs = torch.arange(K, device=img.device)

    for iters in range(2 * int(K**0.5)):
        centroids, diff = batched_kmeans_iter(pixels, centroids, cs)
        if diff < 1 / 256:
            # if new centroids are not changing more than 1 in 8bit depth, break
            break
    dists = (pixels - centroids.unsqueeze(1)).pow(2).sum(dim=-1)
    labels = dists.argmin(dim=-1).unsqueeze(-1).expand(-1, -1, C)
    quant_pixels = torch.gather(centroids, 1, labels)

    return (
        quant_pixels.reshape(B, H, W, C).permute(0, 3, 1, 2),
        centroids,
        labels,
    )


def find_nearest_palette_color(pixel, palette):
    """
    Find the nearest color in the palette for a given pixel.
    """
    dists = (palette.unsqueeze(1) - pixel.unsqueeze(2)).pow(2).sum(dim=-1)
    return torch.gather(
        palette, 1, dists.argmin(dim=2).unsqueeze(-1).expand(-1, -1, palette.size(-1))
    )


# @torch.compile(disable=not TORCH_COMPILE)
def find_nearest_palette_colors_with_distance(pixel, palette):
    """
    Find the two nearest colors in the palette for a given pixel and return them
    along with the relative distance to the first color.
    """
    dists = (palette.unsqueeze(1) - pixel.unsqueeze(2)).pow(2).sum(dim=-1)
    sorted_indices = dists.argsort(dim=2)
    top_closest_indices = sorted_indices[:, :, 0].unsqueeze(-1)
    second_closest_indices = sorted_indices[:, :, 1].unsqueeze(-1)
    top_clostes_distances = dists.gather(2, top_closest_indices)
    second_closest_distances = dists.gather(2, second_closest_indices)
    top_closest_colors = torch.gather(
        palette, 1, top_closest_indices.expand(-1, -1, palette.size(-1))
    )
    second_closest_colors = torch.gather(
        palette, 1, second_closest_indices.expand(-1, -1, palette.size(-1))
    )

    # Calculate interpolation factor based on distances
    dist_ratio = top_clostes_distances / (
        top_clostes_distances + second_closest_distances + 1e-6
    )
    return top_closest_colors, second_closest_colors, dist_ratio


@torch.compile(disable=not TORCH_COMPILE)
def error_diffusion_iter(output, height, width, palette, kernel):
    # Find nearest palette colors for current pixels
    B, C, H, W = output.shape
    current_pixels = output.permute(0, 2, 3, 1).reshape(B, -1, C)
    quantized_pixels = find_nearest_palette_color(current_pixels, palette)
    quantized = quantized_pixels.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    # Calculate error
    error = output - quantized

    # Diffuse error using convolution
    error_padded = F.pad(error, (1, 1, 1, 1), mode="reflect")
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
        torch.tensor([[0, 0, 7], [3, 5, 1]], dtype=image.dtype, device=device) / 16.0
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
    B, C, H, W = output.shape
    final_pixels = output.permute(0, 2, 3, 1).reshape(B, -1, C)
    final_quantized = find_nearest_palette_color(final_pixels, palette)
    return final_quantized.reshape(B, H, W, 3).permute(0, 3, 1, 2)


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
    B, C, H, W = image.shape
    pixels = image.permute(0, 2, 3, 1).reshape(B, -1, C)
    color1, color2, dist_ratio = find_nearest_palette_colors_with_distance(
        pixels, palette
    )

    # Apply threshold matrix to determine color selection
    threshold = bayer.reshape(1, -1, 1)

    # Choose between color1 and color2 based on threshold and distance ratio
    mask = (threshold > dist_ratio).float()
    output_pixels = color1 * mask + color2 * (1 - mask)
    return output_pixels.reshape(B, H, W, 3).permute(0, 3, 1, 2)


def parallel_dither_with_palette(image, quantized, palette, method="error_diffusion"):
    """
    Apply dithering using a specific color palette.

    Args:
        image: Input tensor of shape (channels, height, width)
        palette: Tensor of colors to use (K x channels)
        method: 'error_diffusion' or 'ordered'
    """
    device = image.device
    B, C, H, W = image.shape

    if method == "error_diffusion":
        output = parallel_error_diffusion(image, H, W, palette, device)
    elif method == "ordered":
        output = ordered_dither(image, H, W, palette, device)
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
