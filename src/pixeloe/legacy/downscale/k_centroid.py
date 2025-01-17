import math
from itertools import product

import cv2
import numpy as np
from PIL import Image


def k_centroid_downscale(cv2img, target_size=128, centroids=2):
    """
    k-centroid downscaling algorithm from Astropulse, under MIT License.
    https://github.com/Astropulse/pixeldetector/blob/6e88e18ddbd16529b5dd85b1c615cbb2e5778bf2/k-centroid.py#L19-L44
    """
    h, w, _ = cv2img.shape

    ratio = w / h
    target_size = (target_size**2 / ratio) ** 0.5
    height = int(target_size)
    width = int(target_size * ratio)

    # Perform outline expansion and color matching
    image = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)).convert("RGB")

    # Downscale outline expanded image with k-centroid
    # Create an empty array for the downscaled image
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factors
    wFactor = image.width / width
    hFactor = image.height / height

    # Iterate over each tile in the downscaled image
    for x, y in product(range(width), range(height)):
        # Crop the tile from the original image
        tile = image.crop(
            (x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor)
        )

        # Quantize the colors of the tile using k-means clustering
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert(
            "RGB"
        )

        # Get the color counts and find the most common color
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]

        # Assign the most common color to the corresponding pixel in the downscaled image
        downscaled[y, x, :] = most_common_color

    return cv2.cvtColor(downscaled, cv2.COLOR_RGB2BGR)
