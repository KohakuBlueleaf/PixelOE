import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from pixeloe.torch.pixelize import pixelize
from pixeloe.torch.utils import pre_resize, to_numpy


WEIGHT_MAPPINGS = ("current", "contrast_ratio", "contrast_gated")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PixelOE outline weight mappings on one or more images."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/outline-weight-study")
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--pixel-size", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--mode", default="contrast")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default=None
    )
    parser.add_argument("--mappings", nargs="+", default=list(WEIGHT_MAPPINGS))
    parser.add_argument("--no-color-match", action="store_true")
    return parser.parse_args()


def select_dtype(device, dtype_name):
    if dtype_name is not None:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
    if device == "cuda":
        return torch.float16
    return torch.float32


def psnr(a, b):
    mse = torch.mean((a.float() - b.float()) ** 2)
    if mse.item() == 0:
        return float("inf")
    return float(20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)))


def make_contact_sheet(images, labels):
    if not images:
        raise ValueError("No images were provided")
    label_h = 24
    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(zip(images, labels)):
        x = i * w
        sheet.paste(image.convert("RGB"), (x, label_h))
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
    return sheet


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = select_dtype(args.device, args.dtype)
    report = []

    for input_path in args.inputs:
        img = Image.open(input_path).convert("RGB")
        img_t = pre_resize(
            img, target_size=args.target_size, patch_size=args.pixel_size
        )
        img_t = img_t.to(args.device).to(dtype)

        outputs = {}
        pil_outputs = []
        labels = []
        stem = input_path.stem

        for mapping in args.mappings:
            start = time.perf_counter()
            out_t, _, weight_t = pixelize(
                img_t,
                pixel_size=args.pixel_size,
                thickness=args.thickness,
                mode=args.mode,
                do_color_match=not args.no_color_match,
                weight_mapping=mapping,
                return_intermediate=True,
            )
            elapsed = time.perf_counter() - start
            out_img = Image.fromarray(to_numpy(out_t)[0])
            out_path = args.out_dir / f"{stem}_{mapping}.webp"
            out_img.save(out_path, lossless=True, quality=0)

            outputs[mapping] = out_t.detach().cpu()
            pil_outputs.append(out_img)
            labels.append(mapping)
            report.append(
                {
                    "input": str(input_path),
                    "mapping": mapping,
                    "output": str(out_path),
                    "elapsed_sec": elapsed,
                    "shape": list(out_t.shape),
                    "weight_min": float(weight_t.min().detach().cpu()),
                    "weight_max": float(weight_t.max().detach().cpu()),
                    "weight_mean": float(weight_t.mean().detach().cpu()),
                    "weight_std": float(weight_t.std().detach().cpu()),
                }
            )

        baseline = outputs.get("current")
        if baseline is not None:
            for row in report:
                if row["input"] == str(input_path) and row["mapping"] in outputs:
                    row["psnr_vs_current"] = psnr(outputs[row["mapping"]], baseline)

        sheet = make_contact_sheet(pil_outputs, labels)
        sheet.save(
            args.out_dir / f"{stem}_contact_sheet.webp", lossless=True, quality=0
        )

    with open(args.out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
