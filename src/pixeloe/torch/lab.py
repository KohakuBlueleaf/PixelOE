import torch

try:
    from kornia.color import lab_to_rgb as _kornia_lab_to_rgb
    from kornia.color import rgb_to_lab as _kornia_rgb_to_lab
except Exception:
    _kornia_lab_to_rgb = None
    _kornia_rgb_to_lab = None


def _srgb_to_linear(rgb):
    threshold = rgb.new_tensor(0.04045)
    return torch.where(rgb > threshold, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)


def _linear_to_srgb(rgb):
    threshold = rgb.new_tensor(0.0031308)
    return torch.where(
        rgb > threshold,
        1.055 * torch.clamp(rgb, min=0) ** (1 / 2.4) - 0.055,
        rgb * 12.92,
    )


def _rgb_to_lab_fallback(img):
    rgb = _srgb_to_linear(img.clamp(0, 1))
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    epsilon = img.new_tensor(0.008856)
    kappa = img.new_tensor(7.787)
    offset = img.new_tensor(16 / 116)

    fx = torch.where(x > epsilon, x.clamp(min=0) ** (1 / 3), kappa * x + offset)
    fy = torch.where(y > epsilon, y.clamp(min=0) ** (1 / 3), kappa * y + offset)
    fz = torch.where(z > epsilon, z.clamp(min=0) ** (1 / 3), kappa * z + offset)

    lab_l = 116 * fy - 16
    lab_a = 500 * (fx - fy)
    lab_b = 200 * (fy - fz)
    return torch.cat([lab_l, lab_a, lab_b], dim=1)


def _lab_to_rgb_fallback(lab):
    lab_l, lab_a, lab_b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (lab_l + 16) / 116
    fx = lab_a / 500 + fy
    fz = fy - lab_b / 200

    epsilon = lab.new_tensor(0.008856)
    kappa = lab.new_tensor(7.787)
    offset = lab.new_tensor(16 / 116)

    xr = torch.where(fx**3 > epsilon, fx**3, (fx - offset) / kappa)
    yr = torch.where(lab_l > 7.9996, fy**3, lab_l / 903.3)
    zr = torch.where(fz**3 > epsilon, fz**3, (fz - offset) / kappa)

    x = xr * 0.95047
    y = yr
    z = zr * 1.08883

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb = torch.cat([r, g, b], dim=1)
    return _linear_to_srgb(rgb).clamp(0, 1)


def rgb_to_lab(img):
    if _kornia_rgb_to_lab is not None:
        return _kornia_rgb_to_lab(img)
    return _rgb_to_lab_fallback(img)


def lab_to_rgb(lab):
    if _kornia_lab_to_rgb is not None:
        return _kornia_lab_to_rgb(lab)
    return _lab_to_rgb_fallback(lab)
