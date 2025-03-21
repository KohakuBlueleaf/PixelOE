from timeit import timeit

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

import pixeloe.torch.env as pixeloe_env
from pixeloe.torch.pixelize import pixelize
from pixeloe.torch.outline import outline_expansion
from pixeloe.torch.utils import to_numpy, pre_resize
from pixeloe.torch.minmax import dilate_cont, erode_cont, KERNELS
from pixeloe.torch.sharpen.laplacian import laplacian_sharpen


DEVICE = "cuda"
DTYPE = torch.float16
COMPILE = True
N = 200


if __name__ == "__main__":
    img = Image.open("./img/snow-leopard.webp")

    pixeloe_env.TORCH_COMPILE = COMPILE
    img_t = to_tensor(img).to(DEVICE).to(DTYPE)[None]
    oe_t, w = outline_expansion(img_t, 6, 6, 8, 10, 3)
    oe = Image.fromarray(to_numpy(oe_t)[0])
    oe.save("./img/snow-leopard-oe-orig.webp", lossless=True, quality=0)
    oe_sharp_t = laplacian_sharpen(oe_t, amount=0.5)
    oe_sharp = Image.fromarray(to_numpy(oe_sharp_t)[0])
    oe_sharp.save("./img/snow-leopard-oe-orig-sharp.webp", lossless=True, quality=0)
    pixeloe_env.TORCH_COMPILE = False

    patch_size = 5
    target_size = 240
    thickness = 4
    lg_patch_size = 4
    lg_target_size = 300
    lg_thickness = 3

    img_t = (
        pre_resize(
            img,
            target_size=target_size,
            patch_size=patch_size,
        )
        .to(DEVICE)
        .to(DTYPE)
    )
    img_t_lg = (
        pre_resize(
            img,
            target_size=lg_target_size,
            patch_size=lg_patch_size,
        )
        .to(DEVICE)
        .to(DTYPE)
    )

    print("\nStart Outline Expansion test:")
    dilate_t = dilate_cont(img_t.repeat(2, 1, 1, 1), KERNELS[thickness].to(img_t), 1)
    dilate_img = Image.fromarray(to_numpy(dilate_t)[0])
    dilate_img.save("./img/snow-leopard-dilate.webp", lossless=True, quality=0)

    erode_t = erode_cont(img_t.repeat(2, 1, 1, 1), KERNELS[thickness].to(img_t), 1)
    erode_img = Image.fromarray(to_numpy(erode_t)[0])
    erode_img.save("./img/snow-leopard-erode.webp", lossless=True, quality=0)

    oe_t, w = outline_expansion(
        img_t.repeat(2, 1, 1, 1), thickness, thickness, patch_size, 10, 3
    )
    oe = Image.fromarray(to_numpy(oe_t)[0])
    oe.save("./img/snow-leopard-oe.webp", lossless=True, quality=0)

    oe_sharpt = laplacian_sharpen(oe_t, amount=0.5)
    oe_sharp = Image.fromarray(to_numpy(oe_sharpt)[0])
    oe_sharp.save("./img/snow-leopard-oe-sharp.webp", lossless=True, quality=0)

    w = Image.fromarray(w[0, 0].float().cpu().numpy().clip(0, 1) * 255).convert("L")
    w.save("./img/snow-leopard-w.webp", lossless=True, quality=0)
    print("Outline Expansion test done")

    print("\nStart Pixelize test:")
    print(f"  Patch Size    : {patch_size}")
    print(f"  Thickness     : {thickness}")
    print(f"  Original Size : {img_t.shape[3]}x{img_t.shape[2]}")
    print(
        f"  Pixelized Size: {img_t.shape[3]//patch_size}x{img_t.shape[2]//patch_size}"
    )
    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),  # for testing batch process
        pixel_size=patch_size,
        thickness=thickness,
        do_color_match=False,
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel.webp", lossless=True, quality=0)
    print("    Pixlize test done")
    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),  # for testing batch process
        pixel_size=patch_size,
        thickness=thickness,
        do_color_match=False,
        sharpen_mode="laplacian",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-sharp.webp", lossless=True, quality=0)
    print("    Sharpen test done")

    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),
        pixel_size=patch_size,
        thickness=thickness,
        mode="k_centroid",
        do_color_match=True,
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-k.webp", lossless=True, quality=0)
    print("    K-Centroid test done")

    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),
        pixel_size=patch_size,
        thickness=thickness,
        do_color_match=True,
        do_quant=True,
        num_colors=128,
        dither_mode="",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-128c.webp", lossless=True, quality=0)
    print("    Color Quantization test done")

    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),
        pixel_size=patch_size,
        thickness=thickness,
        do_color_match=True,
        do_quant=True,
        num_colors=128,
        quant_mode="weighted-kmeans",
        dither_mode="",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save(
        "./img/snow-leopard-pixel-128c-weighted.webp", lossless=True, quality=0
    )
    print("    Weighted kmeans test done")

    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),
        pixel_size=patch_size,
        thickness=thickness,
        do_color_match=True,
        do_quant=True,
        num_colors=256,
        dither_mode="ordered",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-256c-d.webp", lossless=True, quality=0)
    print("    Ordered Dithering test done")

    pixel_art_t = pixelize(
        img_t.repeat(2, 1, 1, 1),
        pixel_size=patch_size,
        thickness=thickness,
        do_color_match=True,
        do_quant=True,
        num_colors=256,
        dither_mode="error_diffusion",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-256c-ed.webp", lossless=True, quality=0)
    print("    Error Diffusion test done")

    print("\nStart Pixelize test:")
    print(f"  Patch Size    : {lg_patch_size}")
    print(f"  Thickness     : {lg_thickness}")
    print(f"  Original Size : {img_t_lg.shape[3]}x{img_t_lg.shape[2]}")
    print(
        f"  Pixelized Size: {img_t_lg.shape[3]//lg_patch_size}x{img_t_lg.shape[2]//lg_patch_size}"
    )
    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-lg.webp", lossless=True, quality=0)
    print("    Pixlize test done")

    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
        sharpen_mode="laplacian",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-lg-sharp.webp", lossless=True, quality=0)
    print("    Sharpen test done")

    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
        mode="k_centroid",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-lg-k.webp", lossless=True, quality=0)
    print("    K-Centroid test done")

    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
        do_quant=True,
        num_colors=128,
        dither_mode="",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-lg-128c.webp", lossless=True, quality=0)
    print("    Color Quantization test done")

    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
        do_quant=True,
        num_colors=128,
        quant_mode="weighted-kmeans",
        dither_mode="",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save(
        "./img/snow-leopard-pixel-lg-128c-weighted.webp", lossless=True, quality=0
    )
    print("    Weighted kmeans test done")

    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
        do_quant=True,
        num_colors=256,
        dither_mode="ordered",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-lg-256c-d.webp", lossless=True, quality=0)
    print("    Ordered Dithering test done")

    pixel_art_t = pixelize(
        img_t_lg.repeat(2, 1, 1, 1),
        pixel_size=lg_patch_size,
        thickness=3,
        do_color_match=True,
        do_quant=True,
        num_colors=256,
        dither_mode="error_diffusion",
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t)[0])
    pixel_art.save("./img/snow-leopard-pixel-lg-256c-ed.webp", lossless=True, quality=0)
    print("    Error Diffusion test done")

    exit()

    pixeloe_env.TORCH_COMPILE = COMPILE
    print("\nStart speed test:")
    print(f"  {target_size=}")
    print(f"  {patch_size=}")
    print(f"  {thickness=}")
    print(f"  {pixeloe_env.TORCH_COMPILE=}")
    print("  Results:")
    for bs in [1, 2, 4, 8]:
        # Warmup
        for _ in range(10):
            pixelize(
                img_t.repeat(bs, 1, 1, 1),
                pixel_size=patch_size,
                thickness=thickness,
                do_color_match=False,
            )
        torch.cuda.empty_cache()
        t = timeit(
            """pixelize(
                img_t.repeat(bs, 1, 1, 1),
                pixel_size=patch_size,
                thickness=thickness,
                do_color_match=False,
            )""",
            globals=globals(),
            number=N,
        )
        speed = N / t * bs
        print(f"    bs{bs:2d}: {speed:6.3f}img/sec")
    target_size = lg_target_size
    patch_size = lg_patch_size
    thickness = lg_thickness
    print(f"  {target_size=}")
    print(f"  {patch_size=}")
    print(f"  {thickness=}")
    print(f"  {pixeloe_env.TORCH_COMPILE=}")
    print("  Results:")
    for bs in [1, 2, 4, 8]:
        # Warmup
        for _ in range(10):
            pixelize(
                img_t_lg.repeat(bs, 1, 1, 1),
                pixel_size=lg_patch_size,
                thickness=lg_thickness,
                do_color_match=False,
            )
        torch.cuda.empty_cache()
        t = timeit(
            """pixelize(
                img_t_lg.repeat(bs, 1, 1, 1),
                pixel_size=lg_patch_size,
                thickness=lg_thickness,
                do_color_match=False,
            )""",
            globals=globals(),
            number=N,
        )
        speed = N / t * bs
        print(f"    bs{bs:2d}: {speed:6.3f}img/sec")
    print("Speed test done")
