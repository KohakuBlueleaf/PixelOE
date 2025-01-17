import torch
from ..utils import compile_wrapper


@compile_wrapper
def lanczos_kernel(x: torch.Tensor, a: int = 3) -> torch.Tensor:
    """
    Compute the Lanczos kernel.

    Args:
        x (torch.Tensor): Input values to compute the kernel for
        a (int): The Lanczos window size (default: 3)

    Returns:
        torch.Tensor: Computed Lanczos kernel values
    """
    # Handle the special case where x = 0
    zero_mask = x.abs() < 1e-7
    x = torch.where(zero_mask, torch.ones_like(x), x)

    # Compute the Lanczos kernel
    px = torch.pi * x
    kernel = torch.where(
        x.abs() < a,
        (torch.sin(px) * torch.sin(px / a)) / (px * px / a),
        torch.zeros_like(x),
    )

    # Set the kernel value to 1 at x = 0
    kernel = torch.where(zero_mask, torch.ones_like(kernel), kernel)
    return kernel


@compile_wrapper
def compute_weights_and_indices(
    in_size: int,
    out_size: int,
    scale: float,
    a: int = 3,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cuda",
    support_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute weights and indices for Lanczos resampling.

    Args:
        in_size (int): Size of input dimension
        out_size (int): Size of output dimension
        scale (float): Scaling factor (out_size / in_size)
        a (int): Lanczos window size
        dtype (torch.dtype): Desired dtype for computations
        device (torch.device): Device to perform computations on
        support_scaling (float): Scale factor for the support window during downsampling

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Weights and indices tensors
    """
    # Compute coordinates in the input space
    coord = torch.arange(out_size, dtype=dtype, device=device) / scale

    # Scale the support window based on the scaling factor
    effective_a = a
    if scale < 1.0:  # Downsampling case
        effective_a = a * support_scaling / scale

    # Compute left and right boundaries for each output pixel
    left = coord - effective_a + 0.5
    right = coord + effective_a + 0.5

    # Generate all indices for gathering
    indices = torch.arange(0, right.max().int() + 1, device=device)
    indices = indices.unsqueeze(0).expand(out_size, -1)

    # Compute valid mask for indices
    valid_mask = (indices >= left.unsqueeze(1)) & (indices < right.unsqueeze(1))
    valid_mask = valid_mask & (indices >= 0) & (indices < in_size)

    # Compute weights using the Lanczos kernel
    x = indices.float() - coord.unsqueeze(1)
    weights = lanczos_kernel(x, a)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

    # Normalize weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-7)

    # Get only relevant indices and weights
    relevant_mask = weights.abs() > 1e-7
    indices = torch.masked_select(indices, relevant_mask)
    weights = torch.masked_select(weights, relevant_mask)

    # Store original indices for proper reshaping
    out_indices = torch.arange(out_size, device=device).repeat_interleave(
        relevant_mask.sum(dim=1)
    )

    return weights, torch.stack([out_indices, indices])


def lanczos_resize(
    x: torch.Tensor, size: tuple[int, int], a: int = 3, support_scaling: float = 1.5
) -> torch.Tensor:
    """
    Resize images using Lanczos resampling.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W] with values in [0, 1]
        size (tuple[int, int]): Target size (H', W')
        a (int): Lanczos window size (default: 3)
        support_scaling (float): Scale factor for the support window during downsampling

    Returns:
        torch.Tensor: Resized tensor of shape [B, C, H', W']
    """
    B, C, H, W = x.shape
    H_out, W_out = size
    dtype, device = x.dtype, x.device

    if H == H_out and W == W_out:
        return x

    # Compute scaling factors
    h_scale = H_out / H
    w_scale = W_out / W

    # Resize height first
    if H != H_out:
        weights_h, indices_h = compute_weights_and_indices(
            H, H_out, h_scale, a, dtype, device, support_scaling
        )
        x = x.reshape(B * C, H, W)
        x = (
            torch.sparse_coo_tensor(indices_h, weights_h, (H_out, H))
            .to_dense()
            .to(x)
            .matmul(x)
        )
        x = x.reshape(B, C, H_out, W)

    # Resize width
    if W != W_out:
        weights_w, indices_w = compute_weights_and_indices(
            W, W_out, w_scale, a, dtype, device, support_scaling
        )
        x = x.reshape(B * C, H_out, W)
        x = x.matmul(
            torch.sparse_coo_tensor(indices_w, weights_w, (W_out, W))
            .to_dense()
            .t()
            .to(x)
        )
        x = x.reshape(B, C, H_out, W_out)

    return x.clamp(0, 1)  # Ensure output is in [0, 1]
