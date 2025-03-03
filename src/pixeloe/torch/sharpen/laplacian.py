import torch
import torch.nn.functional as F


def laplacian_sharpen(x, amount=1.0):
    """
    Apply Laplacian sharpening to a batch of images

    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W]
        amount (float): Strength of sharpening effect

    Returns:
        torch.Tensor: Sharpened tensor with same shape as input
    """
    batch_size, channels, height, width = x.shape

    # Define Laplacian kernel
    # This is a 3x3 approximation of the Laplacian operator
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=x.dtype
    ).to(x.device)

    # Expand to match input shape (channels, 1, kernel_height, kernel_width)
    laplacian_kernel = laplacian_kernel.expand(channels, 1, 3, 3)

    # Apply padding to maintain spatial dimensions
    padding = 1

    # Extract edges using the Laplacian filter
    edges = F.conv2d(x, laplacian_kernel, padding=padding, groups=channels)

    # Add the edges to the original image with the specified amount
    sharpened = x + amount * edges

    # Clamp values to maintain valid range (assuming input is in [0, 1])
    return torch.clamp(sharpened, 0, 1)


# Example usage:
if __name__ == "__main__":
    # Create a sample batch of images: [batch_size, channels, height, width]
    batch_size, channels, height, width = 2, 3, 64, 64
    sample_images = torch.rand(batch_size, channels, height, width)

    # Apply Laplacian sharpening
    sharpened_images = laplacian_sharpen(sample_images, amount=0.5)  # 50% sharpening

    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {sharpened_images.shape}")
    print(f"Min value: {sharpened_images.min()}, Max value: {sharpened_images.max()}")
