import torch
import torch.nn.functional as F


def unsharp_mask(x, kernel_size=3, sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking to a batch of images

    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W]
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian kernel
        amount (float): Strength of sharpening effect (1.0 = 100%)
        threshold (float): Minimum brightness difference to apply sharpening

    Returns:
        torch.Tensor: Sharpened tensor with same shape as input
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    # Create Gaussian kernel
    channels = x.shape[1]

    # Create a 1D Gaussian kernel
    gauss = torch.exp(
        -torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float() ** 2
        / (2 * sigma**2)
    )
    kernel_1d = gauss / gauss.sum()

    # Convert it to a 2D kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)

    # Expand to match input channels
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).to(x.device)

    # Apply padding to maintain spatial dimensions
    padding = kernel_size // 2

    # Create a blurred version of the image
    blurred = F.conv2d(x, kernel.to(x.dtype), padding=padding, groups=channels)

    # Calculate the mask (detail)
    mask = x - blurred

    # Apply threshold to the mask
    if threshold > 0:
        mask = torch.where(torch.abs(mask) < threshold, torch.zeros_like(mask), mask)

    # Apply the sharpening
    sharpened = x + amount * mask

    # Clamp values to maintain valid range (assuming input is in [0, 1])
    return torch.clamp(sharpened, 0, 1)


# Example usage:
if __name__ == "__main__":
    # Create a sample batch of images: [batch_size, channels, height, width]
    batch_size, channels, height, width = 2, 3, 64, 64
    sample_images = torch.rand(batch_size, channels, height, width)

    # Apply unsharp masking
    sharpened_images = unsharp_mask(
        sample_images,
        kernel_size=5,
        sigma=1.0,
        amount=1.5,  # 150% sharpening
        threshold=0.1,
    )

    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {sharpened_images.shape}")
    print(f"Min value: {sharpened_images.min()}, Max value: {sharpened_images.max()}")
