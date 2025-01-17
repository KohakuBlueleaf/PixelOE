import os
from argparse import ArgumentParser


def parse_args_pixelize():
    parser = ArgumentParser()
    parser.add_argument("input_img", type=str)
    parser.add_argument("--output_img", "-O", type=str, default=None)
    parser.add_argument(
        "--mode",
        "-M",
        type=str,
        default="contrast",
        choices=["center", "contrast", "k-centroid", "bicubic", "nearest"],
    )
    parser.add_argument("--target_size", "-S", type=int, nargs="+", default=256)
    parser.add_argument("--patch_size", "-P", type=int, default=6)
    parser.add_argument("--thickness", "-T", type=int, default=1)
    parser.add_argument("--no_color_matching", action="store_true")
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--colors", type=int, default=None)
    parser.add_argument("--no_upscale", action="store_true")
    parser.add_argument("--no_downscale", action="store_true")
    return parser.parse_args()


def pixelize():
    args = parse_args_pixelize()
    import numpy as np
    import cv2
    from PIL import Image

    from .legacy.pixelize import pixelize
    from time import perf_counter_ns

    img = Image.open(args.input_img)
    img.load()
    t0 = perf_counter_ns()
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = pixelize(
        img,
        args.mode,
        args.target_size,
        args.patch_size,
        None,
        args.thickness,
        not args.no_color_matching,
        args.contrast,
        args.saturation,
        args.colors,
        args.no_upscale,
        args.no_downscale,
    )
    t1 = perf_counter_ns()
    print(f"{(t1 - t0)/1e6}ms")
    if args.output_img is None:
        args.output_img = os.path.join(
            os.path.dirname(args.input_img),
            f"output_{os.path.basename(args.input_img)}",
        )
        print(f"Output image: {args.output_img}")
    data = cv2.imencode(os.path.splitext(args.output_img)[1], img)[1].tobytes()
    with open(args.output_img, "wb") as f:
        f.write(data)


def parse_args_outline_expansion():
    parser = ArgumentParser()
    parser.add_argument("input_img", type=str)
    parser.add_argument("--output_img", "-O", type=str, default=None)
    parser.add_argument("--target_size", "-S", type=int, default=256)
    parser.add_argument("--patch_size", "-P", type=int, default=6)
    parser.add_argument("--thickness", "-T", type=int, default=1)
    parser.add_argument("--no_color_matching", action="store_true")
    return parser.parse_args()


def outline():
    args = parse_args_outline_expansion()
    import numpy as np
    import cv2
    from PIL import Image

    from .legacy.outline import outline_expansion
    from .legacy.color import match_color

    img = Image.open(args.input_img)
    H, W = img.height, img.width
    target_size = args.target_size
    patch_size = args.patch_size

    ratio = W / H
    target_org_size = (target_size**2 * patch_size**2 / ratio) ** 0.5
    target_org_hw = (int(target_org_size * ratio), int(target_org_size))
    img = img.resize(target_org_hw)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_out = outline_expansion(
        img,
        args.thickness,
        args.thickness,
        patch_size,
    )
    if not args.no_color_matching:
        img_out = match_color(img_out, img)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    img_out = Image.fromarray(img_out)
    if args.output_img is None:
        args.output_img = os.path.join(
            os.path.dirname(args.input_img),
            f"output_{os.path.basename(args.input_img)}",
        )
    img_out.save(args.output_img)


command_map = {
    "pixelize": "pixelize",
    "outline": "outline",
}

if __name__ == "__main__":
    pixelize()
