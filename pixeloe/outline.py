from functools import partial

import numpy as np
import cv2

from .utils import sigmoid, apply_chunk


def expansion_weight(img, k=16, avg_scale=10, dist_scale=3):
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0] / 255
    avg_y = apply_chunk(img_y, k * 3, k, partial(np.median, axis=1, keepdims=True))
    max_y = apply_chunk(img_y, k, k, partial(np.max, axis=1, keepdims=True))
    min_y = apply_chunk(img_y, k, k, partial(np.min, axis=1, keepdims=True))
    bright_dist = max_y - avg_y
    dark_dist = avg_y - min_y

    weight = (avg_y - 0.5) * avg_scale
    weight = weight - (bright_dist - dark_dist) * dist_scale

    output = sigmoid(weight)
    output = cv2.resize(
        output, (img.shape[1] // k, img.shape[0] // k), interpolation=cv2.INTER_NEAREST
    )
    output = cv2.resize(
        output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    return (output - np.min(output)) / (np.max(output))


kernel_expansion = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.uint8)
kernel_smoothing = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)


def outline_expansion(img, erode=2, dilate=2, k=16, avg_scale=10, dist_scale=3):
    weight = expansion_weight(img, k, avg_scale, dist_scale)[..., np.newaxis]
    orig_weight = sigmoid((weight - 0.5) * 5) * 0.25

    img_erode = img.copy()
    img_erode = cv2.erode(img_erode, kernel_expansion, iterations=erode).astype(
        np.float32
    )
    img_dilate = img.copy()
    img_dilate = cv2.dilate(img_dilate, kernel_expansion, iterations=dilate).astype(
        np.float32
    )

    output = img_erode * weight + img_dilate * (1 - weight)
    output = output * (1 - orig_weight) + img.astype(np.float32) * orig_weight
    output = output.astype(np.uint8).copy()

    output = cv2.erode(output, kernel_smoothing, iterations=erode)
    output = cv2.dilate(output, kernel_smoothing, iterations=dilate * 2)
    output = cv2.erode(output, kernel_smoothing, iterations=erode)

    return output


if __name__ == "__main__":
    img = cv2.imread("test.png")
    H, W, C = img.shape
    ratio = W / H
    target_pixel_count = (1024**2 / ratio) ** 0.5
    img = cv2.resize(img, (int(target_pixel_count * ratio), int(target_pixel_count)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    weight_mat = expansion_weight(img)
    img_out = outline_expansion(img, 2, 2)

    # show the weight mat and output in matplotlib
    import matplotlib.pyplot as plt

    plt.imshow(weight_mat, cmap="gray")
    plt.show()
    plt.imshow(img_out)
    plt.show()
