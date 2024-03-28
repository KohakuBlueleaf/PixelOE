import numpy as np


def im2col_2d(input_data, kernel_h, kernel_w, pad=0, stride=1):
    type = input_data.dtype
    N, H, W = input_data.shape

    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad)], "constant").astype(type)
    col = np.zeros((N, kernel_h, kernel_w, out_h, out_w)).astype(type)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, y, x, :, :] = img[:, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 3, 4, 1, 2).reshape(N * out_h * out_w, -1)

    return col


def col2im_2d(col, input_shape, kernel_h, kernel_w, pad=0, stride=1):
    type = col.dtype
    C, H, W = input_shape

    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    col = (
        col.reshape(1, out_h, out_w, C, kernel_h, kernel_w)
        .transpose(0, 3, 4, 5, 1, 2)
        .astype(type)
    )
    img = np.zeros((1, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1)).astype(
        type
    )

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[0, :, pad : H + pad, pad : W + pad]


def unfold(data_im, kernel=1, pad=1, stride=1):
    # Call im2col with calculated dimensions
    return im2col_2d(data_im, kernel, kernel, pad, stride)


def fold(data_col, target_shape, kernel=1, pad=1, stride=1):
    # Call col2im with calculated dimensions
    return col2im_2d(
        data_col,
        target_shape,
        kernel,
        kernel,
        pad,
        stride,
    )


def apply_chunk(data, kernel, stride, func):
    org_shape = data.shape
    unfold_shape = org_shape
    k_shift = max(kernel - stride, 0) // 2
    data = np.pad(data, ((k_shift, k_shift), (k_shift, k_shift)), mode="edge")
    if len(org_shape) < 3:
        data = data[np.newaxis, ...]
        unfold_shape = (1, *unfold_shape)
    data = unfold(data, kernel, 0, stride)
    data[..., : stride**2] = func(data)
    data = fold(data[..., : stride**2], unfold_shape, stride, 0, stride)
    if len(org_shape) < 3:
        data = data[0]
    return data


def test_unfold_fold():
    # Input image configuration
    C, H, W = 1, 6, 6  # Channels, Height, Width
    data_im = np.random.rand(C, H, W)

    # Kernel, stride, pad, and dilation configuration
    kernel = 2
    stride = 2
    pad = 0

    print(data_im)
    print(data_im.shape)
    # Apply unfold (im2col)
    unfolded = unfold(data_im, kernel, pad, stride)
    print(unfolded)
    # Assuming the correct shape for `unfolded` is (C * output_height * output_width, kernel_height * kernel_width)
    print("Unfolded shape:", unfolded.shape)

    # Apply fold (col2im)
    folded = fold(unfolded, (C, H, W), kernel, pad, stride)
    print(folded)

    # Verify the reconstruction
    print("Folded shape:", folded.shape)
    reconstruction_error = np.abs(data_im - folded).mean()
    print("Reconstruction error (mean absolute difference):", reconstruction_error)

    # Assert if the reconstruction is not accurate within a small threshold
    assert reconstruction_error < 1e-5, "Reconstruction failed, error too high!"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # Execute the test function
    test_unfold_fold()
