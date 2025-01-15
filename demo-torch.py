from torchvision.transforms.functional import to_tensor
from PIL import Image

from pixeloe.torch import outline_expansion, to_numpy, pixelize_pytorch, pre_resize
from pixeloe.torch.minmax import dilate_cont, erode_cont, KERNELS
from pixeloe.torch.env import TORCH_COMPILE


if __name__ == "__main__":
    from tqdm import trange

    img = Image.open("./img/snow-leopard.png")

    img_t = to_tensor(img).cuda()
    oe_t, w = outline_expansion(img_t, 6, 6, 8, 10, 3)
    oe = Image.fromarray(to_numpy(oe_t))
    oe.save("./img/snow-leopard-oe-orig.png")

    # with torch.inference_mode():
    #     # Load the test image using OpenCV
    #     for size, patch in [(256, 4), (256, 6), (256, 8)]:
    #         for thickness in range(1, 7):
    #             img_t = pre_resize(img, target_size=size, patch_size=patch).cuda()
    #             outline_expanded, w = outline_expansion(img_t, thickness, thickness, patch, 10, 3)
    #             oe_pixel = Image.fromarray(to_numpy(outline_expanded))
    #             oe_pixel.save(f"./img/snow-leopard_output/test-oe-{size}-{patch}-{thickness}.png")
    #         torch.cuda.empty_cache()

    patch_size = 6
    target_size = 256
    thickness = 4

    img_t = pre_resize(img, target_size=target_size, patch_size=patch_size).cuda()

    dilate_t = dilate_cont(img_t, KERNELS[thickness], 1)
    dilate_img = Image.fromarray(to_numpy(dilate_t))
    dilate_img.save("./img/snow-leopard-dilate.png")

    erode_t = erode_cont(img_t, KERNELS[thickness], 1)
    erode_img = Image.fromarray(to_numpy(erode_t))
    erode_img.save("./img/snow-leopard-erode.png")

    oe_t, w = outline_expansion(img_t, thickness, thickness, patch_size, 10, 3)
    oe = Image.fromarray(to_numpy(oe_t))
    oe.save("./img/snow-leopard-oe.png")
    w = Image.fromarray(w.cpu().numpy().clip(0, 1) * 255).convert("L")
    w.save("./img/snow-leopard-w.png")

    pixel_art_t = pixelize_pytorch(
        img_t,
        target_size=target_size,
        patch_size=patch_size,
        thickness=thickness,
        do_color_match=False,
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t))
    pixel_art.save("./img/snow-leopard-pixel.png")

    pixel_art_t = pixelize_pytorch(
        pre_resize(img, target_size=360, patch_size=4).cuda(),
        target_size=320,
        patch_size=4,
        thickness=3,
        do_color_match=True,
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t))
    pixel_art.save("./img/snow-leopard-pixel-lg.png")

    print("Start speed test:")
    print(f"  {target_size=}")
    print(f"  {patch_size=}")
    print(f"  {thickness=}")
    print(f"  {TORCH_COMPILE=}")
    for _ in trange(500, smoothing=0.01):
        pixel_art_t = pixelize_pytorch(
            img_t,
            target_size=target_size,
            patch_size=patch_size,
            thickness=thickness,
            do_color_match=False,
        )
    print("Speed test done")

