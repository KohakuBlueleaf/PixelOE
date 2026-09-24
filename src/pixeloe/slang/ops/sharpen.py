"""3x3 sharpening (unsharp mask, Laplacian)."""

import numpy as np
import torch

MODULE = "sharpen/sharpen"

LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)


def unsharp_table(kernel_size=3, sigma=1.0):
    """The reference Gaussian: float32 1-D, normalised, outer product."""
    gauss = torch.exp(
        -torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float() ** 2
        / (2 * sigma**2)
    )
    k1 = gauss / gauss.sum()
    return (k1.unsqueeze(0) * k1.unsqueeze(1)).numpy().astype(np.float32)


def sharpen(ctx, src, mode, amount, threshold=0.1):
    """mode 'unsharp' or 'laplacian' -> new array."""
    b, c, h, w = src.shape
    if mode == "unsharp":
        taps = ctx.cached_constant(("unsharp", 3, 1.0), unsharp_table)
        mode_id = 0
    elif mode == "laplacian":
        taps = ctx.cached_constant(("laplacian",), lambda: LAPLACIAN)
        mode_id = 1
    else:
        raise ValueError(f"Unknown sharpen mode: {mode}")
    dst = ctx.empty(src.shape)
    ctx.dispatch(
        MODULE,
        "sharpen",
        (w, h, b * c),
        src=src,
        dst=dst,
        taps=taps,
        height=h,
        width=w,
        mode=mode_id,
        amount=np.float32(amount),
        threshold=np.float32(threshold),
    )
    return dst
