import math

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from tqdm import trange
from ..utils import batched_kmeans_iter
from ..utils import compile_wrapper


# @compile_wrapper
def batched_kmeans(data, num_clusters):
    """
    Performs batched k-means clustering.

    Args:
        data (torch.Tensor): Input data tensor of shape [b, N, D].
        num_clusters (int): The number of clusters (centroids).
        max_iters (int): Maximum number of iterations for k-means.

    Returns:
        torch.Tensor: Tensor of cluster centroids of shape [b, num_clusters, D].
    """
    # deterministically initialize centroids
    maxv = data.max(dim=1, keepdim=True)
    minv = data.min(dim=1, keepdim=True)
    interp = torch.linspace(0, 1, num_clusters, device=data.device)[None, :, None]
    centroids = interp * minv.values + (1 - interp) * maxv.values
    data = data.unsqueeze(2)

    for _ in range(max(2 * int(num_clusters**0.5), 4)):
        centroids, diff = batched_kmeans_iter(data, centroids)
        if diff < 1 / 256:
            break

    return centroids


@compile_wrapper
def k_centroid_preprocess(img_batch, b, c, h, w, height_down, width_down):
    # calculate scaling factors
    h_factor = h / height_down
    w_factor = w / width_down

    # create grid for unfolding
    kernel_size = (int(math.ceil(h_factor)), int(math.ceil(w_factor)))
    stride = kernel_size
    expected_height = height_down * kernel_size[0]
    expected_width = width_down * kernel_size[1]
    diff_h = expected_height - h
    diff_w = expected_width - w
    img_batch = F.pad(
        img_batch,
        (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2),
        mode="replicate",
    )

    # Unfold the input batch into patches
    patches = F.unfold(img_batch, kernel_size=kernel_size, stride=stride)
    patches = patches.reshape(b, c, -1, patches.shape[-1]).permute(0, 3, 2, 1)
    # patches shape: (b, num_patches, kernel_h * kernel_w, c)

    # Reshape patches for batched k-means: [b * num_patches, num_pixels_in_patch, c]
    num_patches = patches.shape[1]
    patches = patches.flatten(0, 1)
    return patches, num_patches


@compile_wrapper
def k_centroid_postprocess(
    patches, kmeans_centroids, num_patches, b, c, height_down, width_down
):
    # patches: [b, num_patches, num_pixels_in_patch, 1, c]
    # kmeans_centroids: [b, num_patches, 1, centroids_k, c]
    # dist: [b, num_patches, num_pixels_in_patch, centroids_k]
    # Determine the closest centroid for each original pixel in the patch
    distances = torch.sum(
        (patches.unsqueeze(-2) - kmeans_centroids.unsqueeze(-3)) ** 2, dim=-1
    ).sum(dim=1)
    closest_centroids_indices = torch.argmin(distances, dim=-1)
    # closest_centroids_indices: [b, num_patches, num_pixels_in_patch]

    # Get the colors of the closest centroids for each pixel in the patches
    closest_centroid_colors = torch.gather(
        kmeans_centroids, 1, closest_centroids_indices[:, None, None].expand(-1, -1, c)
    )

    # Reshape back to the patch structure
    quantized_patches_reshaped = (
        closest_centroid_colors.reshape(b, num_patches, -1, c)
        .permute(0, 3, 2, 1)
        .flatten(1, 2)
    )

    return quantized_patches_reshaped.reshape(b, c, height_down, width_down)


# @compile_wrapper
def k_centroid_downscale_torch(img_batch, pixel_size=128, centroids_k=2):
    """
    PyTorch implementation of k-centroid downscaling, optimized for batch processing.

    Args:
        img_batch (torch.Tensor): Input batch of images (b, c, h, w), assuming channels are RGB.
        target_size (int): The target size for the downscaled image (longest side).
        centroids_k (int): Number of centroids for k-means clustering.

    Returns:
        torch.Tensor: batch of downscaled images (b, c, h_down, w_down).
    """
    b, c, h, w = img_batch.shape
    height_down = h // pixel_size
    width_down = w // pixel_size

    patches, num_patches = k_centroid_preprocess(
        img_batch, b, c, h, w, height_down, width_down
    )
    kmeans_centroids = batched_kmeans(patches, num_clusters=centroids_k)
    return k_centroid_postprocess(
        patches, kmeans_centroids, num_patches, b, c, height_down, width_down
    )


if __name__ == "__main__":
    # Load the image using OpencV
    img_cv2 = cv2.imread("./img/snow-leopard.webp")
    if img_cv2 is None:
        exit(1)

    # convert to RGb and normalize to [0, 1]
    img_rgb = cv2.cvtcolor(img_cv2, cv2.cOLOR_bGR2RGb)
    img_torch = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_torch = img_torch.unsqueeze(0)  # Add batch dimension

    # Apply the downscaling
    downscaled_torch = k_centroid_downscale_torch(img_torch, centroids_k=2)

    # convert the downscaled PyTorch tensor back to a NumPy array for saving
    downscaled_np = (
        downscaled_torch.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255
    ).astype(np.uint8)

    # convert back to bGR for saving with OpencV
    downscaled_bgr = cv2.cvtcolor(downscaled_np, cv2.cOLOR_RGb2bGR)

    # Save the downscaled image
    cv2.imwrite("./img/snow-leopard-k-centroid.png", downscaled_bgr)
