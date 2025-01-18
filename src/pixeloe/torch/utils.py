from functools import cache

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from . import env
from ..logger import logger


@cache
def compile_warning_log_once():
    logger.warning(
        "Torch compile is not enabled. "
        "This may result in large vram usage and slow performance."
    )


def compile_wrapper(func):
    compiled = torch.compile(
        func,
        dynamic=True,
        options={
            "shape_padding": True,
        },
    )

    def runner(*args, **kwargs):
        if env.TORCH_COMPILE:
            return compiled(*args, **kwargs)
        else:
            compile_warning_log_once()
            return func(*args, **kwargs)

    return runner


def to_numpy(tensor):
    """
    Convert a torch.Tensor [B,C,H,W] with range [0..1]
    back to a NumPy HWC image [0..255].
    """
    return list(
        (tensor.float().permute(0, 2, 3, 1).cpu().numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def pre_resize(
    img_pil,
    target_size=128,
    patch_size=8,
):
    if isiterable(target_size):
        in_w = target_size[0] * patch_size
        in_h = target_size[1] * patch_size
    else:
        W, H = img_pil.size
        ratio = W / H
        in_h = (target_size**2 / ratio) ** 0.5
        in_w = int(in_h * ratio) * patch_size
        in_h = int(in_h) * patch_size
    img_pil = img_pil.resize((in_w, in_h), Image.Resampling.BICUBIC)
    img_t = to_tensor(img_pil)
    return img_t[None]


@compile_wrapper
def batched_kmeans_iter(datas, centroids, cs=None, weights=None):
    """
    datas: (B, N, C)
    centroids: (B, K, C)
    cs: (B, N)
    weights: (B, N, 1, 1)
    """
    K = centroids.shape[1]
    if cs is None:
        cs = torch.arange(K, device=datas.device)
    dists = (datas - centroids.unsqueeze(1)).pow(2)
    if weights is not None:
        dists = dists * weights
    dists = dists.sum(dim=-1)
    labels = dists.argmin(dim=-1).unsqueeze(-1).repeat(1, 1, K)
    masks = labels == cs
    mask_valid = masks.sum(dim=1) > 0
    cluster_sums = torch.sum(masks.unsqueeze(-1) * datas, dim=1)
    cluster_means = cluster_sums / masks.sum(dim=1)[:, :, None]
    new_centroids = torch.where(mask_valid[:, :, None], cluster_means, centroids)
    diff = torch.max((new_centroids - centroids).abs())
    return new_centroids, diff


@compile_wrapper
def repeat_elements(data, repeat_counts):
    B, _, D = data.shape

    # Flatten batch dimension for repeat_interleave
    flat_data = data.reshape(-1, D)  # [B*N, D]
    flat_counts = repeat_counts.reshape(-1)  # [B*N]

    # Repeat elements
    repeated = torch.repeat_interleave(flat_data, flat_counts, dim=0)

    # Reshape back to [B, N2, D]
    N2 = repeat_counts.sum(dim=1)[0]  # Assuming all batches have same N2
    return repeated.reshape(B, N2, D)


@compile_wrapper
def generate_repeat_table(weights, N, N2):
    """
    Generate a repeat table ensuring each element is repeated at least once.

    Args:
        weights: torch.Tensor of shape [B, N] containing weights for each element
        N: Original sequence length
        N2: Target sequence length after repeating (must be >= N)

    Returns:
        torch.Tensor of shape [B, N] containing integer repeat counts, all >= 1
    """
    if N2 < N:
        raise ValueError(
            f"N2 ({N2}) must be >= N ({N}) to ensure at least one repeat per element"
        )

    B = weights.shape[0]

    # Handle negative or zero weights
    weights = torch.clamp(weights, min=1e-8)

    # Normalize weights with numerical stability
    log_weights = torch.log(weights)
    log_weights_normalized = log_weights - torch.logsumexp(
        log_weights, dim=1, keepdim=True
    )
    normalized_weights = torch.exp(log_weights_normalized)

    # Calculate remaining counts after ensuring minimum of 1
    remaining_N2 = N2 - N

    # Distribute remaining counts according to weights
    remaining_counts = normalized_weights * remaining_N2
    floor_counts = torch.floor(remaining_counts)
    remainders = remaining_counts - floor_counts

    # Initialize repeat counts with 1 for each element
    repeat_counts = torch.ones_like(weights, dtype=torch.long)
    repeat_counts = repeat_counts + floor_counts.long()

    # Distribute any remaining counts
    remaining = (remaining_N2 - floor_counts.sum(dim=1, keepdim=True)).long()
    _, top_remainder_indices = torch.topk(remainders, k=remaining.max().item(), dim=1)

    for b in range(B):
        repeat_counts[b, top_remainder_indices[b, : remaining[b]]] += 1

    return repeat_counts
