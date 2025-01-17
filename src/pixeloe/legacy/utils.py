import time

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def apply_chunk(data, kernel, stride, func):
    org_shape = data.shape
    unfold_shape = org_shape

    k_shift = max(kernel - stride, 0)
    pad_pattern = (k_shift // 2, k_shift // 2 + k_shift % 2)
    data = np.pad(data, (pad_pattern, pad_pattern), "edge")

    if len(org_shape) == 2:
        data = data[None, None, ...]

    data = (
        F.unfold(torch.tensor(data), kernel, 1, 0, stride).transpose(-1, -2)[0].numpy()
    )
    data[..., : stride**2] = func(data)
    data = data[None, ..., : stride**2]
    data = F.fold(
        torch.tensor(data).transpose(-1, -2),
        # data.transpose(-1, -2),
        unfold_shape,
        stride,
        1,
        0,
        stride,
    )[0].numpy()

    if len(org_shape) < 3:
        data = data[0]
    return data


@torch.no_grad()
def apply_chunk_torch(data, kernel, stride, func):
    org_shape = data.shape
    unfold_shape = org_shape

    k_shift = max(kernel - stride, 0)
    pad_pattern = (k_shift // 2, k_shift // 2 + k_shift % 2)
    data = np.pad(data, (pad_pattern, pad_pattern), "edge")

    if len(org_shape) == 2:
        data = data[None, None, ...]

    data = F.unfold(torch.tensor(data), kernel, 1, 0, stride).transpose(-1, -2)[0]
    data[..., : stride**2] = func(data)
    data = data[None, ..., : stride**2]
    data = F.fold(
        data.transpose(-1, -2),
        unfold_shape,
        stride,
        1,
        0,
        stride,
    )[0].numpy()

    if len(org_shape) < 3:
        data = data[0]
    return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def isiterable(x):
    return hasattr(x, "__iter__")


if __name__ == "__main__":
    from cv2 import medianBlur, resize, INTER_NEAREST_EXACT

    kernel = 17
    stride = 8
    image_size = 128 * 8

    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, (image_size, image_size)).astype(np.float32)
    t0 = time.perf_counter_ns()
    output = apply_chunk(
        data, kernel, stride, lambda x: np.median(x, axis=1, keepdims=True)
    )
    t1 = time.perf_counter_ns()
    print(f"{(t1 - t0)/1e6}ms")
    print()

    t0 = time.perf_counter_ns()
    output3 = apply_chunk_torch(
        data, kernel, stride, lambda x: torch.median(x, dim=1, keepdims=True).values
    )
    t1 = time.perf_counter_ns()
    print(f"{(t1 - t0)/1e6}ms")
    print()

    t0 = time.perf_counter_ns()
    output2 = medianBlur(data.astype(np.uint8), kernel)
    output2 = resize(
        resize(
            output2,
            (image_size // stride, image_size // stride),
            interpolation=INTER_NEAREST_EXACT,
        ),
        (image_size, image_size),
        interpolation=INTER_NEAREST_EXACT,
    )
    t1 = time.perf_counter_ns()
    print(f"{(t1 - t0)/1e6}ms")
    print(output.shape, data.shape)
    print(output2.shape)
    print(np.mean(np.abs(output - output2.astype(np.float32))))
