import cv2
import numpy as np
from PIL import Image


def match_color(source, target, level=5):
    # Convert RGB to L*a*b*, and then match the std/mean
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    result = (source_lab - np.mean(source_lab)) / np.std(source_lab)
    result = result * np.std(target_lab) + np.mean(target_lab)
    source = cv2.cvtColor(
        (result * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    source = source.astype(np.float32)
    # Use wavelet colorfix method to match original low frequency data at first
    source[:, :, 0] = wavelet_colorfix(source[:, :, 0], target[:, :, 0], level=level)
    source[:, :, 1] = wavelet_colorfix(source[:, :, 1], target[:, :, 1], level=level)
    source[:, :, 2] = wavelet_colorfix(source[:, :, 2], target[:, :, 2], level=level)
    output = source
    return output.clip(0, 255).astype(np.uint8)


def wavelet_colorfix(inp, target, level=5):
    inp_high, _ = wavelet_decomposition(inp, level)
    _, target_low = wavelet_decomposition(target, level)
    output = inp_high + target_low
    return output


def wavelet_decomposition(inp, levels):
    high_freq = np.zeros_like(inp)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def wavelet_blur(inp, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(inp, (kernel_size, kernel_size), 0)
    return output


def color_styling(inp, saturation=1.2, contrast=1.1):
    output = inp.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    output[:, :, 1] = output[:, :, 1] * saturation
    output[:, :, 2] = output[:, :, 2] * contrast - (contrast - 1)
    output = np.clip(output, 0, 1)
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return output


def weighted_kmeans(image, colors=32, weights=None, repeats=64):
    h, w, c = image.shape
    pixels = []
    weights = weights / np.max(weights) * repeats
    for i in range(h):
        for j in range(w):
            repeat_times = max(1, int(weights[i, j]))
            pixels.extend([image[i, j]] * repeat_times)
    pixels = np.array(pixels, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 32, 1)
    _, labels, palette = cv2.kmeans(
        pixels, colors, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS
    )

    quantized_image = np.zeros((h, w, c), dtype=np.uint8)
    label_idx = 0
    for i in range(h):
        for j in range(w):
            repeat_times = max(1, int(weights[i, j]))
            quantized_image[i, j] = palette[labels[label_idx]]
            label_idx += repeat_times
    return quantized_image


def kmeans(image, colors=32):
    h, w, c = image.shape
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 32, 1)
    _, labels, palette = cv2.kmeans(
        pixels, colors, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS
    )

    quantized_image = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            quantized_image[i, j] = palette[labels[i * w + j]]
    return quantized_image


def maxcover(image, colors=32):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_quant = img_pil.quantize(colors, 1, kmeans=colors).convert("RGB")
    return cv2.cvtColor(np.array(img_quant), cv2.COLOR_RGB2BGR)


def color_quant(image, colors=32, weights=None, repeats=64, method="kmeans"):
    # TODO: more consistent/better color quant method
    #       (K-means is not good enough)
    match method:
        case "kmeans":
            if weights is not None:
                return weighted_kmeans(image, colors, weights, repeats)
            else:
                return kmeans(image, colors)
        case "maxcover":
            return maxcover(image, colors)
