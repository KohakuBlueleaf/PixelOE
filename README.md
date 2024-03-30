# Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion.

A python implementation for this [project](https://github.com/KohakuBlueleaf/PixelOE-matlab).

- **No AI**
- **No NN**
- **GPU Free**

## Example

|| Original | nearest | bicubic | center | k-centroid |
|-| ------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| Without Outline Expansion| <img src="img/dragon-girl.png" width="350" /> |<img src="demo/dragon-girl-t0-nearest.png" width="350" />|<img src="demo/dragon-girl-t0-bicubic.png" width="350" />|<img src="demo/dragon-girl-t0-center.png" width="350" />|<img src="demo/dragon-girl-t0-k-centroid.png" width="350" />|
| With Outline Expansion| <img src="img/dragon-girl.png" width="350" /> |<img src="demo/dragon-girl-t1-nearest.png" width="350" />|<img src="demo/dragon-girl-t1-bicubic.png" width="350" />|<img src="demo/dragon-girl-t1-center.png" width="350" />|<img src="demo/dragon-girl-t1-k-centroid.png" width="350" />|

|| Original | nearest | bicubic | center | k-centroid |
|-| ------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| Without Outline Expansion| <img src="img/house.png" width="350" /> |<img src="demo/house-t0-nearest.png" width="350" />|<img src="demo/house-t0-bicubic.png" width="350" />|<img src="demo/house-t0-center.png" width="350" />|<img src="demo/house-t0-k-centroid.png" width="350" />|
| With Outline Expansion| <img src="img/house.png" width="350" /> |<img src="demo/house-t2-nearest.png" width="350" />|<img src="demo/house-t2-bicubic.png" width="350" />|<img src="demo/house-t2-center.png" width="350" />|<img src="demo/house-t2-k-centroid.png" width="350" />|

## Usage

You can install this package through `pip`:

```
pip install pixeloe
```

And then use cli to run the command:
```
pixeloe.pixelize --help
```

Which should give you this message:
```
usage: pixeloe.pixelize [-h] [--output_img OUTPUT_IMG] [--mode {contrast-based,k-centroid}] [--target_size TARGET_SIZE]
                        [--patch_size PATCH_SIZE] [--thickness THICKNESS] [--color_matching] [--contrast CONTRAST]
                        [--saturation SATURATION] [--colors COLORS] [--no_upscale]
                        input_img

positional arguments:
  input_img

options:
  -h, --help            show this help message and exit
  --output_img OUTPUT_IMG, -O OUTPUT_IMG
  --mode {contrast-based,k-centroid}, -M {contrast-based,k-centroid}
  --target_size TARGET_SIZE, -S TARGET_SIZE
  --patch_size PATCH_SIZE, -P PATCH_SIZE
  --thickness THICKNESS, -T THICKNESS
  --color_matching
  --contrast CONTRAST
  --saturation SATURATION
  --colors COLORS
  --no_upscale
```

For example
```
pixeloe.pixelize img/test.png --output_img img/test2.png --target_size 256 --patch_size 8
```

---

Or you can import it into your code:

```python
import cv2
from pixeloe import pixelize

img = cv2.imread("img/test.png")
img = pixelize(img, 256, patch_size=8)
cv2.imwrite("img/test2.png", img)
```

## Algorithm Explanation
There are 2 main component of this algorithm:
1. Outline Expansion
2. Contrast-based downscale
  * Not implemented yet, need more tweak and experiments.

Since we have lot of different algorithm for downscale and basically they are better than mine, I will only focus on Outline Expansion part in this section.

### Outline Expansion

The goal of Outline Expansion is to expand important small details and high contrast edges in the image before downscaling, so that they are not lost in the final low resolution pixel art. The key steps are:

1. Compute a weight map that highlights areas to expand:
   - Convert image to grayscale
   - Calculate local median brightness in a neighborhood 2x(or 3x) the patch size
   - Find local max and min brightness within each patch 
   - Compute bright and dark distances as the difference between local max/min and median
   - Combine two weighting terms:
     - weight_h1: Darker median pixels should prioritize keeping brighter details
     - weight_h2: Larger bright vs dark distance indicates which extreme details to keep
   - Apply sigmoid to the summed weights and normalize between 0-1

2. Erode the input image for a number of iterations to shrink bright regions

3. Dilate the input image for a number of iterations to expand bright regions

4. Blend the eroded and dilated images using the computed weight map:
   - Brighter weight values favor the dilated image to keep bright details
   - Darker weight values favor the eroded image to keep dark details

5. Apply morphological closing and opening to the blended result to clean up edge artifacts

The Contrast-Aware Outline Expansion ensures that fine details and sharp edges are broadened before the subsequent downscaling step. This allows them to be represented at the final low target resolution rather than being lost entirely. The selective erosion and dilation based on local contrast helps expand the right regions while preserving overall sharpness.

By integrating this outline expansion with an effective downscaling strategy and optional color palette optimization, the full pixelization pipeline is able to generate attractive pixel-style artwork from high resolution images. The intentional emphasis on important visual elements sets this approach apart from direct downsampling methods.

## Acknowledgement
* Astropulse
  * k-centorid downscaling algorithm.
* Claude 3 opus: 
  * Summarize the algorithm.
  * Convert some matlab code to python.
