from .contrast_based import contrast_based_downscale
from .center import center_downscale
from .k_centroid import k_centroid_downscale


downscale_mode = {
    "center": center_downscale,
    "contrast-based": contrast_based_downscale,
    "k-centroid": k_centroid_downscale,
}
