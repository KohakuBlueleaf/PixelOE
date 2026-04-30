import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import pixeloe.torch.env as pixeloe_env
from pixeloe.torch.minmax import KERNELS, dilate_cont, erode_cont
from pixeloe.torch.outline import normalize_weight, outline_weight_stats, polarity_score
from pixeloe.torch.utils import pre_resize

pixeloe_env.TORCH_COMPILE = False

INPUT_PATH = Path("img/snow-leopard.webp")
OUT_DIR = Path("outputs/outline_expansion_polarity_ablation_snow_leopard")
TARGET_SIZE = 256
PIXEL_SIZE = 4
THICKNESS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

SETUPS = [
    {
        "name": "baseline_current__avg10__dist3__global_norm",
        "kind": "score_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
    },
    {
        "name": "baseline_current__avg10__dist3__no_norm",
        "kind": "score_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "none",
    },
    {
        "name": "median_only__avg10__dist0__global_norm",
        "kind": "score_weight",
        "avg_scale": 10,
        "dist_scale": 0,
        "normalize": "global",
    },
    {
        "name": "detail_only__avg0__dist3__global_norm",
        "kind": "score_weight",
        "avg_scale": 0,
        "dist_scale": 3,
        "normalize": "global",
    },
    {
        "name": "weaker_background__avg6__dist3__global_norm",
        "kind": "score_weight",
        "avg_scale": 6,
        "dist_scale": 3,
        "normalize": "global",
    },
    {
        "name": "stronger_background__avg14__dist3__global_norm",
        "kind": "score_weight",
        "avg_scale": 14,
        "dist_scale": 3,
        "normalize": "global",
    },
    {
        "name": "weaker_detail__avg10__dist1__global_norm",
        "kind": "score_weight",
        "avg_scale": 10,
        "dist_scale": 1,
        "normalize": "global",
    },
    {
        "name": "stronger_detail__avg10__dist5__global_norm",
        "kind": "score_weight",
        "avg_scale": 10,
        "dist_scale": 5,
        "normalize": "global",
    },
    {
        "name": "strong_both__avg14__dist5__global_norm",
        "kind": "score_weight",
        "avg_scale": 14,
        "dist_scale": 5,
        "normalize": "global",
    },
    {
        "name": "score_conf_soft__avg10__dist3__global_norm",
        "kind": "score_conf_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "score_low": 0.6,
        "score_high": 2.4,
    },
    {
        "name": "score_conf_mid__avg10__dist3__global_norm",
        "kind": "score_conf_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "score_low": 1.0,
        "score_high": 3.0,
    },
    {
        "name": "score_conf_strict__avg10__dist3__global_norm",
        "kind": "score_conf_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "score_low": 1.6,
        "score_high": 4.0,
    },
    {
        "name": "contrast_conf_soft__avg10__dist3__global_norm",
        "kind": "contrast_conf_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "contrast_low": 0.02,
        "contrast_high": 0.18,
    },
    {
        "name": "contrast_conf_mid__avg10__dist3__global_norm",
        "kind": "contrast_conf_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "contrast_low": 0.04,
        "contrast_high": 0.24,
    },
    {
        "name": "contrast_conf_strict__avg10__dist3__global_norm",
        "kind": "contrast_conf_weight",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "contrast_low": 0.08,
        "contrast_high": 0.30,
    },
    {
        "name": "score_conf_blend_original_soft__avg10__dist3",
        "kind": "score_conf_blend_original",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "score_low": 0.6,
        "score_high": 2.4,
    },
    {
        "name": "score_conf_blend_original_mid__avg10__dist3",
        "kind": "score_conf_blend_original",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "score_low": 1.0,
        "score_high": 3.0,
    },
    {
        "name": "contrast_conf_blend_original_soft__avg10__dist3",
        "kind": "contrast_conf_blend_original",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "contrast_low": 0.02,
        "contrast_high": 0.18,
    },
    {
        "name": "contrast_conf_blend_original_mid__avg10__dist3",
        "kind": "contrast_conf_blend_original",
        "avg_scale": 10,
        "dist_scale": 3,
        "normalize": "global",
        "contrast_low": 0.04,
        "contrast_high": 0.24,
    },
]


def smoothstep(x, low, high):
    t = ((x - low) / (high - low)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def tensor_to_uint8_hwc(t):
    arr = t.detach().float().permute(0, 2, 3, 1).cpu().numpy()[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def gray_tensor_to_image(t):
    arr = t[0, 0].detach().float().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def save_webp(img, path):
    img.save(path, lossless=True, quality=0)


def direct_diff_image(original_np, oe_np):
    diff = oe_np.astype(np.float32) - original_np.astype(np.float32)
    return Image.fromarray(np.clip(diff * 2 + 128, 0, 255).astype(np.uint8))


def abs_diff_image(original_np, oe_np):
    diff = np.abs(oe_np.astype(np.float32) - original_np.astype(np.float32))
    return Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8))


def make_sheet(rows, columns, cell_w, cell_h, label_h=30, scale=0.5):
    scaled_w = max(1, int(cell_w * scale))
    scaled_h = max(1, int(cell_h * scale))
    sheet = Image.new(
        "RGB", (scaled_w * columns, (scaled_h + label_h) * len(rows)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        y = row_idx * (scaled_h + label_h)
        for col_idx, (label, image) in enumerate(row):
            x = col_idx * scaled_w
            draw.text((x + 4, y + 6), label, fill=(0, 0, 0))
            if image.size != (scaled_w, scaled_h):
                image = image.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)
            sheet.paste(image.convert("RGB"), (x, y + label_h))
    return sheet


def post_smooth(out, thickness):
    oc_iter = max(thickness - 1, thickness - 1, 1)
    out = erode_cont(out, KERNELS[oc_iter].to(out), 1)
    out = dilate_cont(out, KERNELS[oc_iter].to(out), 2)
    out = erode_cont(out, KERNELS[oc_iter].to(out), 1)
    return out


def compute_weight_and_conf(setup, stats):
    l_med, _l_min, _l_max, bright_dist, dark_dist, local_contrast = stats
    score = polarity_score(
        l_med,
        bright_dist,
        dark_dist,
        setup["avg_scale"],
        setup["dist_scale"],
    )
    raw_weight = torch.sigmoid(score)
    weight = normalize_weight(raw_weight, setup.get("normalize", "global"))
    conf = torch.ones_like(weight)

    if setup["kind"].startswith("score_conf"):
        conf = smoothstep(score.abs(), setup["score_low"], setup["score_high"])
        if setup["kind"] == "score_conf_weight":
            weight = normalize_weight(
                0.5 + (weight - 0.5) * conf,
                setup.get("normalize", "global"),
            )
    elif setup["kind"].startswith("contrast_conf"):
        conf = smoothstep(
            local_contrast,
            setup["contrast_low"],
            setup["contrast_high"],
        )
        if setup["kind"] == "contrast_conf_weight":
            weight = normalize_weight(
                0.5 + (weight - 0.5) * conf,
                setup.get("normalize", "global"),
            )

    return weight, conf, score, local_contrast


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for subdir in ["oe", "weight", "confidence", "direct_diff", "abs_diff", "sheets"]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    img = Image.open(INPUT_PATH).convert("RGB")
    img_t = pre_resize(img, target_size=TARGET_SIZE, patch_size=PIXEL_SIZE)
    img_t = img_t.to(DEVICE).to(DTYPE)
    input_np = tensor_to_uint8_hwc(img_t)
    input_img = Image.fromarray(input_np)
    save_webp(input_img, OUT_DIR / "input_preresized.webp")

    stats = outline_weight_stats(img_t, PIXEL_SIZE, PIXEL_SIZE // 2)
    eroded = erode_cont(img_t, KERNELS[THICKNESS].to(img_t), 1)
    dilated = dilate_cont(img_t, KERNELS[THICKNESS].to(img_t), 1)

    report = {
        "input": str(INPUT_PATH),
        "output_dir": str(OUT_DIR),
        "target_size": TARGET_SIZE,
        "pixel_size": PIXEL_SIZE,
        "thickness": THICKNESS,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "formula": "score = avg_scale * (l_med - 0.5) + dist_scale * (dark_dist - bright_dist)",
        "setups": [],
    }

    rows = []
    oe_rows = [[("input", input_img)]]

    for setup in SETUPS:
        start = time.perf_counter()
        weight, conf, score, local_contrast = compute_weight_and_conf(setup, stats)
        out = eroded * weight + dilated * (1.0 - weight)
        out = post_smooth(out, THICKNESS)
        if setup["kind"].endswith("blend_original"):
            out = out * conf + img_t * (1.0 - conf)
        elapsed = time.perf_counter() - start

        oe_np = tensor_to_uint8_hwc(out)
        oe_img = Image.fromarray(oe_np)
        weight_img = gray_tensor_to_image(weight)
        conf_img = gray_tensor_to_image(conf)
        direct_img = direct_diff_image(input_np, oe_np)
        abs_img = abs_diff_image(input_np, oe_np)

        name = setup["name"]
        paths = {
            "oe": OUT_DIR / "oe" / f"{name}.webp",
            "weight": OUT_DIR / "weight" / f"{name}.webp",
            "confidence": OUT_DIR / "confidence" / f"{name}.webp",
            "direct_diff": OUT_DIR / "direct_diff" / f"{name}.webp",
            "abs_diff": OUT_DIR / "abs_diff" / f"{name}.webp",
        }
        save_webp(oe_img, paths["oe"])
        save_webp(weight_img, paths["weight"])
        save_webp(conf_img, paths["confidence"])
        save_webp(direct_img, paths["direct_diff"])
        save_webp(abs_img, paths["abs_diff"])

        rows.append(
            [
                (f"{name} / OE", oe_img),
                ("weight", weight_img),
                ("confidence", conf_img),
                ("abs diff", abs_img),
            ]
        )
        oe_rows.append([(name, oe_img)])

        report["setups"].append(
            {
                **setup,
                "elapsed_sec": elapsed,
                "paths": {k: str(v) for k, v in paths.items()},
                "weight_min": float(weight.min().detach().cpu()),
                "weight_max": float(weight.max().detach().cpu()),
                "weight_mean": float(weight.mean().detach().cpu()),
                "weight_std": float(weight.std().detach().cpu()),
                "confidence_mean": float(conf.mean().detach().cpu()),
                "confidence_std": float(conf.std().detach().cpu()),
                "score_mean": float(score.mean().detach().cpu()),
                "score_std": float(score.std().detach().cpu()),
                "local_contrast_mean": float(local_contrast.mean().detach().cpu()),
                "local_contrast_std": float(local_contrast.std().detach().cpu()),
                "mean_abs_diff_8bit": float(
                    np.abs(
                        oe_np.astype(np.float32) - input_np.astype(np.float32)
                    ).mean()
                ),
            }
        )

    cell_w, cell_h = input_img.size
    rows_per_page = 4
    for page_idx in range(math.ceil(len(rows) / rows_per_page)):
        page_rows = rows[page_idx * rows_per_page : (page_idx + 1) * rows_per_page]
        save_webp(
            make_sheet(page_rows, 4, cell_w, cell_h),
            OUT_DIR / "sheets" / f"formal_sheet_page_{page_idx + 1:02}.webp",
        )
    for page_idx in range(math.ceil(len(oe_rows) / rows_per_page)):
        page_rows = oe_rows[page_idx * rows_per_page : (page_idx + 1) * rows_per_page]
        save_webp(
            make_sheet(page_rows, 1, cell_w, cell_h),
            OUT_DIR / "sheets" / f"oe_only_page_{page_idx + 1:02}.webp",
        )

    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Generated {len(SETUPS)} formalized variants in {OUT_DIR}")
    for path in sorted((OUT_DIR / "sheets").glob("*.webp")):
        print(" ", path)


if __name__ == "__main__":
    main()
