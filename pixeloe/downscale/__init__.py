from .contrast_based import contrast_based_downscale
from .conventional import bicubic, nearest
from .center import center_downscale
from .k_centroid import k_centroid_downscale


downscale_mode = {
    "bicubic": bicubic,
    "nearest": nearest,
    "center": center_downscale,
    "contrast": contrast_based_downscale,
    "k-centroid": k_centroid_downscale,
}
