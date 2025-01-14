import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from kornia.color import lab_to_rgb, rgb_to_lab
from PIL import Image

from .env import TORCH_COMPILE
from .minmax import dilate_cont, erode_cont, KERNELS


def to_numpy(tensor):
    """
    Convert a torch.Tensor [C,H,W] with range [0..1] back to a NumPy HWC image [0..255].
    """
    return (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


@torch.compile(disable=not TORCH_COMPILE)
def local_stat(tensor, kernel, stride, stat="median"):
    B, C, H, W = tensor.shape
    patches = F.unfold(
        tensor, kernel_size=kernel, stride=stride, padding=kernel // 2
    )
    if stat == "median":
        vals = patches.median(dim=1, keepdims=True).values.repeat(
            1, patches.size(1), 1
        )
    elif stat == "max":
        vals = patches.max(dim=1, keepdims=True).values.repeat(
            1, patches.size(1), 1
        )
    elif stat == "min":
        vals = patches.min(dim=1, keepdims=True).values.repeat(
            1, patches.size(1), 1
        )
    out = F.fold(
        vals,
        output_size=(H, W),
        kernel_size=kernel,
        stride=stride,
        padding=kernel // 2,
    )
    return out / ((kernel // stride) ** 2 + 1e-8)


@torch.compile(disable=not TORCH_COMPILE)
def expansion_weight(img, k=16, stride=4, avg_scale=10, dist_scale=3):
    """
    Compute a weight matrix for outline expansion.
    """
    lab = rgb_to_lab(img.unsqueeze(0))  # [1,3,H,W]
    L = lab[:, 0:1] / 100  # [1,1,H,W]

    L_med = local_stat(L, k * 2, stride, stat="median")
    L_min = local_stat(L, k, stride, stat="min")
    L_max = local_stat(L, k, stride, stat="max")

    bright_dist = L_max - L_med
    dark_dist = L_med - L_min

    weight = (L_med - 0.5) * avg_scale - (bright_dist - dark_dist) * dist_scale
    weight = torch.sigmoid(weight)

    weight = (weight - weight.amin()) / (weight.amax() - weight.amin() + 1e-8)
    return weight[0, 0]  # shape [H,W]


def outline_expansion(
    img, erode_iters=2, dilate_iters=2, k=16, avg_scale=10, dist_scale=3
):
    """
    Perform contrast-aware outline expansion on an image.
    """
    w = expansion_weight(img, k, k // 2, avg_scale, dist_scale)

    e = erode_cont(img, KERNELS[erode_iters], 1)
    d = dilate_cont(img, KERNELS[dilate_iters], 1)

    w3 = w.unsqueeze(0).repeat(3, 1, 1)
    out = e * w3 + d * (1.0 - w3)

    oc_iter = max(erode_iters-2, dilate_iters-2, 1)

    out = erode_cont( out, KERNELS[oc_iter], 1)
    out = dilate_cont(out, KERNELS[oc_iter], 2)
    out = erode_cont( out, KERNELS[oc_iter], 1)

    return out, w


@torch.compile(disable=not TORCH_COMPILE)
def find_pixel_luminance(chunk):
    mid_idx = chunk.shape[1] // 2
    mid = chunk[:, mid_idx].unsqueeze(1)
    med = chunk.median(dim=1).values.unsqueeze(1)
    mu = chunk.mean(dim=1, keepdim=True)
    maxi = chunk.max(dim=1).values.unsqueeze(1)
    mini = chunk.min(dim=1).values.unsqueeze(1)

    out = mid.clone()
    cond1 = (med < mu) & ((maxi - med) > (med - mini))
    cond2 = (med > mu) & ((maxi - med) < (med - mini))

    out[cond1[:, 0]] = mini[cond1[:, 0]]
    out[cond2[:, 0]] = maxi[cond2[:, 0]]
    return out


@torch.compile(disable=not TORCH_COMPILE)
def contrast_downscale(img, patch_size=8):
    """
    Contrast-based downscaling of an image using unfold to process patches concurrently.
    """
    H, W = img.shape[1], img.shape[2]
    patch_h = patch_size
    patch_w = patch_size
    out_h = H // patch_h
    out_w = W // patch_w

    lab = rgb_to_lab(img.unsqueeze(0))  # [1,3,H,W]
    L, A, B = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    # Unfold channels into patches
    patches_L = F.unfold(L, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches_A = F.unfold(A, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches_B = F.unfold(B, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

    # Reshape to [N, patch_area] where N = out_h*out_w
    patches_L = patches_L.squeeze(0).transpose(0, 1)  # [N, patch_h*patch_w]
    patches_A = patches_A.squeeze(0).transpose(0, 1)  # [N, patch_h*patch_w]
    patches_B = patches_B.squeeze(0).transpose(0, 1)  # [N, patch_h*patch_w]

    # Process luminance concurrently across patches
    result_L = find_pixel_luminance(patches_L)  # [1, patch**2, N]
    # Compute median for A and B channels concurrently
    result_A = patches_A.median(dim=1).values[:, None]  # [1, patch**2, N]
    result_B = patches_B.median(dim=1).values[:, None]  # [1, patch**2, N]

    # Reshape results to [1,1,out_h,out_w]
    # fold
    result_L = result_L.transpose(0, 1).reshape(1, 1, out_h, -1)
    result_A = result_A.transpose(0, 1).reshape(1, 1, out_h, -1)
    result_B = result_B.transpose(0, 1).reshape(1, 1, out_h, -1)

    out_lab = torch.cat([result_L, result_A, result_B], dim=1)  # [1,3,out_h,out_w]
    out_rgb = lab_to_rgb(out_lab)  # [1,3,out_h,out_w]
    return out_rgb[0]  # [3,out_h,out_w]


def match_color_pytorch(src, tgt):
    """
    Match color statistics from target to source in LAB space.
    """
    s = src.reshape(3, -1)
    t = tgt.reshape(3, -1)

    mean_s, std_s = s.mean(dim=1), s.std(dim=1) + 1e-8
    mean_t, std_t = t.mean(dim=1), t.std(dim=1) + 1e-8

    s_norm = (s - mean_s[:, None]) / std_s[:, None]
    s_match = s_norm * std_t[:, None] + mean_t[:, None]
    out = s_match.reshape_as(src)

    return out


def color_quantization_kmeans(img, K=32, max_iter=10):
    """
    Naive k-means color quantization on an image tensor.
    """
    pixels = img.permute(1, 2, 0).reshape(-1, 3)
    centroids = pixels[torch.randperm(pixels.shape[0])[:K]]

    for _ in range(max_iter):
        dists = (pixels.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=2)
        labels = dists.argmin(dim=1)
        new_centroids = []
        for c in range(K):
            cluster_points = pixels[labels == c]
            if cluster_points.shape[0] > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[c])
        new_centroids = torch.stack(new_centroids, dim=0)
        diff = (new_centroids - centroids).abs().sum()
        centroids = new_centroids
        if diff < 1e-5:
            break

    quant_pixels = centroids[labels]
    return quant_pixels.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)


def pre_resize(
    img_pil,
    target_size=128,
    patch_size=8,
):
    W, H = img_pil.size
    ratio = W / H
    in_h = (target_size**2 / ratio) ** 0.5
    in_w = int(in_h * ratio) * patch_size
    in_h = int(in_h) * patch_size
    img_pil = img.resize((in_w, in_h), Image.Resampling.BICUBIC)
    img_t = to_tensor(img_pil)
    return img_t


def pixelize_pytorch(
    img_t,
    target_size=256,
    patch_size=6,
    thickness=3,
    mode="contrast",
    do_color_match=True,
    do_quant=False,
    K=32,
):
    """
    Main pipeline: pixelize an image using PyTorch.
    """

    if thickness > 0:
        expanded, w = outline_expansion(img_t, thickness, thickness, patch_size)
    else:
        expanded = img_t

    if do_color_match:
        e_lab = rgb_to_lab(expanded.unsqueeze(0))
        o_lab = rgb_to_lab(img_t.unsqueeze(0))
        matched_lab = match_color_pytorch(e_lab[0], o_lab[0])
        expanded = lab_to_rgb(matched_lab.unsqueeze(0))[0]

    H, W = expanded.shape[1], expanded.shape[2]
    ratio = W / H
    out_h = int((target_size**2 / ratio) ** 0.5)
    out_w = int(out_h * ratio)

    if mode == "contrast":
        down = contrast_downscale(expanded, patch_size)
    else:
        down = F.interpolate(
            expanded.unsqueeze(0), size=(out_h, out_w), mode="nearest"
        )[0]

    if do_quant:
        down_q = color_quantization_kmeans(down, K=K)
        down_final = match_color_pytorch(down_q, down)
    else:
        down_final = down

    out_pixel = F.interpolate(
        down_final.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0]

    return out_pixel


if __name__ == "__main__":
    from tqdm import trange

    img = Image.open("test.png")

    img_t = to_tensor(img).cuda()
    oe_t, w = outline_expansion(img_t, 6, 6, 8, 10, 3)
    oe = Image.fromarray(to_numpy(oe_t))
    oe.save("test-oe-orig.png")

    # with torch.inference_mode():
    #     # Load the test image using OpenCV
    #     for size, patch in [(256, 4), (256, 6), (256, 8)]:
    #         for thickness in range(1, 7):
    #             img_t = pre_resize(img, target_size=size, patch_size=patch).cuda()
    #             outline_expanded, w = outline_expansion(img_t, thickness, thickness, patch, 10, 3)
    #             oe_pixel = Image.fromarray(to_numpy(outline_expanded))
    #             oe_pixel.save(f"test_output/test-oe-{size}-{patch}-{thickness}.png")
    #         torch.cuda.empty_cache()

    patch_size = 4
    target_size = 320
    thickness = 4

    img_t = pre_resize(img, target_size=target_size, patch_size=patch_size).cuda()

    dilate_t = dilate_cont(img_t, KERNELS[thickness], 1)
    dilate_img = Image.fromarray(to_numpy(dilate_t))
    dilate_img.save("test-dilate.png")

    erode_t = erode_cont(img_t, KERNELS[thickness], 1)
    erode_img = Image.fromarray(to_numpy(erode_t))
    erode_img.save("test-erode.png")

    oe_t, w = outline_expansion(img_t, thickness, thickness, patch_size, 10, 3)
    oe = Image.fromarray(to_numpy(oe_t))
    oe.save("test-oe.png")

    pixel_art_t = pixelize_pytorch(
        img_t,
        target_size=target_size,
        patch_size=patch_size,
        thickness=thickness,
        do_color_match=False,
    )
    pixel_art = Image.fromarray(to_numpy(pixel_art_t))
    pixel_art.save("test-pixel.png")
    print("Pixelated image saved as test-pixel.png")

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
