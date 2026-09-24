"""Benchmark matrix: image sizes x pipeline cases x implementation paths."""

# name -> (width, height, pixel_size, batch)
SIZES = {
    "512": (512, 512, 4, 1),
    "1024": (1024, 1024, 4, 1),
    "1080p": (1920, 1080, 4, 1),
    "2048p8": (2048, 2048, 8, 1),
    "1024x4": (1024, 1024, 4, 4),
}

# name -> pixelize kwargs (on top of pixel_size from SIZES, thickness=3)
CASES = {
    "default": {},
    "k_centroid": {"mode": "k_centroid"},
    "lanczos": {"mode": "lanczos"},
    "nearest": {"mode": "nearest"},
    "bilinear": {"mode": "bilinear"},
    "bicubic": {"mode": "bicubic"},
    "area": {"mode": "area"},
    "thickness0": {"thickness": 0},
    "thickness6": {"thickness": 6},
    "no_color_match": {"do_color_match": False},
    "unsharp": {"sharpen_mode": "unsharp"},
    "laplacian": {"sharpen_mode": "laplacian"},
    "contrast_gated": {"weight_mapping": "contrast_gated"},
    "quant32_ordered": {"do_quant": True, "num_colors": 32, "dither_mode": "ordered"},
    "quant32_ed": {
        "do_quant": True,
        "num_colors": 32,
        "dither_mode": "error_diffusion",
    },
    "quant256_none": {"do_quant": True, "num_colors": 256, "dither_mode": "none"},
    "weighted_kmeans": {
        "do_quant": True,
        "num_colors": 64,
        "quant_mode": "weighted-kmeans",
        "dither_mode": "ordered",
    },
    "repeat_kmeans": {
        "do_quant": True,
        "num_colors": 64,
        "quant_mode": "repeat-kmeans",
        "dither_mode": "ordered",
    },
}

ALL_CASES = list(CASES)
CORE_CASES = ["default", "k_centroid", "quant32_ordered", "no_color_match"]

# size -> cases, per device class
GPU_MATRIX = {
    "512": CORE_CASES,
    "1024": ALL_CASES,
    "1080p": ALL_CASES,
    "2048p8": CORE_CASES,
    "1024x4": CORE_CASES,
}
CPU_MATRIX = {
    "512": ALL_CASES,
    "1024": CORE_CASES,
}

# path name -> runner spec
GPU_PATHS = {
    "torch-cuda-fp32": {"impl": "torch", "device": "cuda", "dtype": "float32"},
    "torch-cuda-fp16": {"impl": "torch", "device": "cuda", "dtype": "float16"},
    # compile costs ~40-60 s per new shape/config: core cases only
    "torch-cuda-fp32-compile": {
        "impl": "torch",
        "device": "cuda",
        "dtype": "float32",
        "compile": True,
        "cases": CORE_CASES,
    },
    "torch-cuda-fp16-compile": {
        "impl": "torch",
        "device": "cuda",
        "dtype": "float16",
        "compile": True,
        "cases": CORE_CASES,
    },
    "slang-cuda": {"impl": "slang", "backend": "cuda", "dtype": "float32"},
    "slang-cuda-fp16in": {"impl": "slang", "backend": "cuda", "dtype": "float16"},
    "slang-cuda-separable": {
        "impl": "slang",
        "backend": "cuda",
        "dtype": "float32",
        "colorfix_blur": "separable",
    },
    "slang-d3d12": {"impl": "slang", "backend": "d3d12", "dtype": "float32"},
    "slang-vulkan": {"impl": "slang", "backend": "vulkan", "dtype": "float32"},
}
CPU_PATHS = {
    "torch-cpu-fp32": {"impl": "torch", "device": "cpu", "dtype": "float32"},
    "slang-cpu": {"impl": "slang", "backend": "cpu", "dtype": "float32"},
}

REFERENCE_PATH = {"gpu": "torch-cuda-fp32", "cpu": "torch-cpu-fp32"}
