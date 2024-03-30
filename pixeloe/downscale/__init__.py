from .contrast_based import contrast_based_downscale
from .k_centroid import k_centroid_downscale


downscale_mode = {
    "contrast-based": contrast_based_downscale,
    "k-centroid": k_centroid_downscale,
}
