import math

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from tqdm import trange
from ..utils import batched_kmeans_iter
from ..env import TORCH_COMPILE


# @torch.compile(disable=not TORCH_COMPILE)
def batched_kmeans(data, num_clusters):
    """
    Performs batched k-means clustering.

    Args:
        data (torch.Tensor): Input data tensor of shape [B, N, D].
        num_clusters (int): The number of clusters (centroids).
        max_iters (int): Maximum number of iterations for k-means.

    Returns:
        torch.Tensor: Tensor of cluster centroids of shape [B, num_clusters, D].
    """
    B, N, D = data.shape
    random_idx = torch.stack(
        [
            torch.randperm(data.shape[1], device=data.device)[:num_clusters]
            for _ in range(B)
        ]
    )
    centroids = torch.gather(data, 1, random_idx.unsqueeze(-1).expand(-1, -1, D))
    data = data.unsqueeze(2)

    for _ in range(2 * int(num_clusters**0.5)):
        centroids, diff = batched_kmeans_iter(data, centroids)
        if diff < 1 / 256:
            break

    return centroids


#@torch.compile(disable=not TORCH_COMPILE)
def k_centroid_downscale_torch(img_batch, target_size=128, centroids_k=2):
    """
    PyTorch implementation of k-centroid downscaling, optimized for batch processing.

    Args:
        img_batch (torch.Tensor): Input batch of images (B, C, H, W), assuming channels are RGB.
        target_size (int): The target size for the downscaled image (longest side).
        centroids_k (int): Number of centroids for k-means clustering.

    Returns:
        torch.Tensor: Batch of downscaled images (B, C, H_down, W_down).
    """
    B, C, H, W = img_batch.shape

    ratio = W / H
    height = (target_size**2 / ratio) ** 0.5
    width = height * ratio
    height_down = int(height)
    width_down = int(width)

    # Calculate scaling factors
    h_factor = H / height_down
    w_factor = W / width_down

    # Create grid for unfolding
    kernel_size = (int(math.ceil(h_factor)), int(math.ceil(w_factor)))
    stride = kernel_size
    expected_height = height_down * kernel_size[0]
    expected_width = width_down * kernel_size[1]
    diff_h = expected_height - H
    diff_w = expected_width - W
    img_batch = F.pad(
        img_batch,
        (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2),
        mode="reflect",
    )

    # Unfold the input batch into patches
    patches = F.unfold(img_batch, kernel_size=kernel_size, stride=stride)
    patches = patches.reshape(B, C, -1, patches.shape[-1]).permute(0, 3, 2, 1)
    # patches shape: (B, num_patches, kernel_h * kernel_w, C)

    # Reshape patches for batched k-means: [B * num_patches, num_pixels_in_patch, C]
    num_patches = patches.shape[1]
    patches = patches.flatten(0, 1)

    # Apply batched k-means
    kmeans_centroids = batched_kmeans(
        patches, num_clusters=centroids_k
    )  # [B * num_patches, centroids_k, C]

    # patches: [B, num_patches, num_pixels_in_patch, 1, C]
    # kmeans_centroids: [B, num_patches, 1, centroids_k, C]
    # dist: [B, num_patches, num_pixels_in_patch, centroids_k]
    # Determine the closest centroid for each original pixel in the patch
    distances = torch.sum(
        (patches.unsqueeze(-2) - kmeans_centroids.unsqueeze(-3)) ** 2, dim=-1
    ).sum(dim=1)
    closest_centroids_indices = torch.argmin(distances, dim=-1)
    # closest_centroids_indices: [B, num_patches, num_pixels_in_patch]

    # Get the colors of the closest centroids for each pixel in the patches
    closest_centroid_colors = torch.gather(
        kmeans_centroids, 1, closest_centroids_indices[:, None, None].expand(-1, -1, C)
    )

    # Reshape back to the patch structure
    quantized_patches_reshaped = (
        closest_centroid_colors.reshape(B, num_patches, -1, C)
        .permute(0, 3, 2, 1)
        .flatten(1, 2)
    )

    return quantized_patches_reshaped.reshape(B, C, height_down, width_down)


if __name__ == "__main__":
    # Load the image using OpenCV
    img_cv2 = cv2.imread("./img/snow-leopard.webp")
    if img_cv2 is None:
        exit(1)

    # Convert to RGB and normalize to [0, 1]
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_torch = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_torch = img_torch.unsqueeze(0)  # Add batch dimension

    # Apply the downscaling
    downscaled_torch = k_centroid_downscale_torch(img_torch, centroids_k=2)

    # Convert the downscaled PyTorch tensor back to a NumPy array for saving
    downscaled_np = (
        downscaled_torch.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255
    ).astype(np.uint8)

    # Convert back to BGR for saving with OpenCV
    downscaled_bgr = cv2.cvtColor(downscaled_np, cv2.COLOR_RGB2BGR)

    # Save the downscaled image
    cv2.imwrite("./img/snow-leopard-k-centroid.png", downscaled_bgr)
