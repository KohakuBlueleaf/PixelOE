# Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion.

A python implementation for this [project](https://github.com/KohakuBlueleaf/PixelOE-matlab).

## Example

| Before             | After　             |
| ------------------ | ------------------- |
| ![img](img/test.png) | ![img](img/test2.png) |

## Usage

You can install this package through `pip`:

```
pip install pixeloe
```

And then import it in your code:

```python
import cv2
from pixeloe import pixelize

img = cv2.imread("img/test.png")
img = pixelize(img, 256, patch_size=8)
cv2.imwrite("img/test2.png", img)
```
