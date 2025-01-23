import cv2
import numpy as np
import torch

from .utils import sigmoid, apply_chunk, apply_chunk_torch


def expansion_weight(img, k=8, stride=2, avg_scale=10, dist_scale=3):
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0] / 255
    avg_y = apply_chunk_torch(
        img_y, k * 2, stride, lambda x: torch.median(x, dim=1, keepdims=True).values
    )
    max_y = apply_chunk_torch(
        img_y, k, stride, lambda x: torch.max(x, dim=1, keepdims=True).values
    )
    min_y = apply_chunk_torch(
        img_y, k, stride, lambda x: torch.min(x, dim=1, keepdims=True).values
    )
    bright_dist = max_y - avg_y
    dark_dist = avg_y - min_y

    weight = (avg_y - 0.5) * avg_scale
    weight = weight - (bright_dist - dark_dist) * dist_scale

    output = sigmoid(weight)
    output = cv2.resize(
        output,
        (img.shape[1] // stride, img.shape[0] // stride),
        interpolation=cv2.INTER_LINEAR,
    )
    output = cv2.resize(
        output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    return (output - np.min(output)) / (np.max(output))


kernel_expansion = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.uint8)
kernel_smoothing = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)


def outline_expansion(img, erode=2, dilate=2, k=16, avg_scale=10, dist_scale=3):
    weight = expansion_weight(img, k, (k // 4) * 2, avg_scale, dist_scale)[..., None]
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

    weight = np.abs(weight * 2 - 1) * 255
    weight = cv2.dilate(weight.astype(np.uint8), kernel_expansion, iterations=dilate)
    return output, weight.astype(np.float32) / 255


if __name__ == "__main__":
    img = cv2.imread("img/dragon-girl.webp")
    h, w, _ = img.shape
    ratio = w / h
    target_pixel_count = (1024**2 / ratio) ** 0.5
    img = cv2.resize(img, (int(target_pixel_count * ratio), int(target_pixel_count)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    weight_mat = expansion_weight(img, 8, 2, 9, 3)
    img_out = outline_expansion(img, 1, 1)

    edge = img_out - img
    edge = (edge + 255) / 2
    edge[np.abs(edge - 127) >= 30] = 0
    edge[np.abs(edge - 127) < 30] = 255
    edge = cv2.cvtColor(edge.astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255

    # show the weight mat and output in matplotlib in same window
    import matplotlib.pyplot as plt

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("input")
    plt.subplot(1, 4, 2)
    plt.imshow(img_out)
    plt.axis("off")
    plt.title("output")
    plt.subplot(1, 4, 3)
    plt.imshow(weight_mat, cmap="gray")
    plt.axis("off")
    plt.title("weight")
    plt.subplot(1, 4, 4)
    plt.imshow(edge, cmap="gray")
    plt.axis("off")
    plt.title("edge")
    plt.show()
