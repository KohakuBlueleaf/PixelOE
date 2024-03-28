import numpy as np
import cv2


def match_color(source, target):
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
    source[:, :, 0] = wavelet_colorfix(source[:, :, 0], target[:, :, 0])
    source[:, :, 1] = wavelet_colorfix(source[:, :, 1], target[:, :, 1])
    source[:, :, 2] = wavelet_colorfix(source[:, :, 2], target[:, :, 2])
    output = source
    return output.clip(0, 255).astype(np.uint8)


def wavelet_colorfix(input, target):
    input_high, _ = wavelet_decomposition(input, 5)
    _, target_low = wavelet_decomposition(target, 5)
    output = input_high + target_low
    return output


def wavelet_decomposition(input, levels):
    high_freq = np.zeros_like(input)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = wavelet_blur(input, radius)
        high_freq = high_freq + (input - low_freq)
        input = low_freq
    return high_freq, low_freq


def wavelet_blur(input, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(input, (kernel_size, kernel_size), 0)
    return output


def color_styling(input, saturation=1.2, contrast=1.1):
    output = input.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    output[:, :, 1] = output[:, :, 1] * saturation
    output[:, :, 2] = output[:, :, 2] * contrast - (contrast - 1)
    output = np.clip(output, 0, 1)
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return output
