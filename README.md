# Detail-Oriented Pixelization based on Contrast-Aware Outline Expansion.

A python implementation for this [project](https://github.com/KohakuBlueleaf/PixelOE-matlab).

- **No AI**
- **No NN**
- **GPU Free**

## Example

| Original | contrast-based | k-centroid |
| ------------------ | ------------------- | ------------------- |
| <img src="img/test.png" width="350" /> | <img src="img/test2.png" width="350" /> | <img src="img/test3.png" width="350" /> |

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
usage: pixeloe.pixelize [-h] [--output_img OUTPUT_IMG] [--target_size TARGET_SIZE] [--patch_size PATCH_SIZE] [--thickness THICKNESS]
                        [--color_matching] [--contrast CONTRAST] [--saturation SATURATION] [--colors COLORS] [--no_upscale]
                        input_img

positional arguments:
  input_img
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
