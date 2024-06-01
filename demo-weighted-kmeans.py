import cv2
import numpy as np
import torch

from PIL import Image
from pytorch_msssim import ssim as _ssim

from pixeloe.pixelize import pixelize


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0)
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0)
    return _ssim(img1, img2)


if __name__ == "__main__":
    size = 256
    thickness = 2
    patch_size = 6
    pixel_size = 4
    img = Image.open("img/fox-girl.png")
    img = np.array(img)
    img_arr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for size in [256]:
        for colors in [32, 48, 64]:
            img_weighted = pixelize(
                img_arr,
                target_size=size,
                thickness=thickness,
                patch_size=patch_size,
                colors=colors,
                colors_with_weight=True,
            )
            img_weighted = cv2.cvtColor(img_weighted, cv2.COLOR_BGR2RGB)
            img_normal = pixelize(
                img_arr,
                target_size=size,
                thickness=thickness,
                patch_size=patch_size,
                colors=colors,
                colors_with_weight=False,
            )
            img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
            img_ref = pixelize(
                img_arr,
                target_size=size,
                thickness=thickness,
                patch_size=patch_size,
            )
            img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

            psnr_weighted = psnr(np.array(img_weighted), np.array(img_ref))
            psnr_normal = psnr(np.array(img_normal), np.array(img_ref))
            print(
                f"colors: {colors:03}, size: {size:04}, "
                f"psnr_weighted: {psnr_weighted:5.2f}, psnr_normal: {psnr_normal:5.2f}"
            )
            ssim_weighted = ssim(np.array(img_weighted), np.array(img_ref))
            ssim_normal = ssim(np.array(img_normal), np.array(img_ref))
            print(
                f"colors: {colors:03}, size: {size:04}, "
                f"ssim_weighted: {ssim_weighted:5.3f}, ssim_normal: {ssim_normal:5.3f}"
            )

            img_weighted = Image.fromarray(img_weighted)
            img_normal = Image.fromarray(img_normal)
            img_ref = Image.fromarray(img_ref)
            grid = Image.new("RGB", (img_weighted.width * 3, img_weighted.height))
            grid.paste(img_ref, (0, 0))
            grid.paste(img_weighted, (img_ref.width, 0))
            grid.paste(img_normal, (img_ref.width * 2, 0))
            grid.save(f"demo/weighted_kmeans/{size}_{colors}.png")
