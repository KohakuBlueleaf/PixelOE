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
        data = data[np.newaxis, np.newaxis, ...]

    data = (
        F.unfold(torch.tensor(data), kernel, 1, 0, stride).transpose(-1, -2)[0].numpy()
    )
    data[..., : stride**2] = func(data)
    data = data[np.newaxis, ..., : stride**2]
    data = F.fold(
        torch.tensor(data).transpose(-1, -2),
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
    data = np.zeros((504, 504))
    output = apply_chunk(data, 9, 4, lambda x: x[..., 0:1])
    print(output.shape, data.shape)
