// sRGB <-> CIE Lab (D65, 2 deg observer), matching kornia.color.rgb_to_lab /
// lab_to_rgb(clip=True) operation by operation.

module lab;

import sel;

static const float LAB_WHITE_X = 0.95047;
static const float LAB_WHITE_Y = 1.0;
static const float LAB_WHITE_Z = 1.08883;
static const float LAB_EPS = 0.008856;
static const float LAB_KAPPA = 7.787;
static const float LAB_OFFSET = 0.13793103448275862; // 4 / 29
static const float LAB_INV_EPS = 0.2068966;
static const float SRGB_EPS = 0.04045;
static const float LINEAR_EPS = 0.0031308;
static const float ONE_THIRD = 0.3333333333333333;
static const float INV_GAMMA = 0.4166666666666667; // 1 / 2.4

public float srgb_to_linear(float v)
{
    return v > SRGB_EPS ? pow((v + 0.055) / 1.055, 2.4) : v / 12.92;
}

public float linear_to_srgb(float v)
{
    return v > LINEAR_EPS ? 1.055 * pow(fmax_s(v, LINEAR_EPS), INV_GAMMA) - 0.055 : 12.92 * v;
}

float lab_f(float v)
{
    return v > LAB_EPS ? pow(fmax_s(v, LAB_EPS), ONE_THIRD) : LAB_KAPPA * v + LAB_OFFSET;
}

float lab_f_inv(float f)
{
    return f > LAB_INV_EPS ? f * f * f : (f - LAB_OFFSET) / LAB_KAPPA;
}

public float3 rgb_to_lab(float3 rgb)
{
    float r = srgb_to_linear(rgb.x);
    float g = srgb_to_linear(rgb.y);
    float b = srgb_to_linear(rgb.z);

    float x = 0.412453 * r + 0.357580 * g + 0.180423 * b;
    float y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
    float z = 0.019334 * r + 0.119193 * g + 0.950227 * b;

    float fx = lab_f(x / LAB_WHITE_X);
    float fy = lab_f(y / LAB_WHITE_Y);
    float fz = lab_f(z / LAB_WHITE_Z);

    return float3(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

// rgb_to_lab(rgb).x / 100 (the luminance map of the outline statistics)
public float rgb_to_lum(float3 rgb)
{
    return rgb_to_lab(rgb).x / 100.0;
}

public float3 lab_to_rgb(float3 lab)
{
    float fy = (lab.x + 16.0) / 116.0;
    float fx = lab.y / 500.0 + fy;
    float fz = fmax_s(fy - lab.z / 200.0, 0.0);

    float x = lab_f_inv(fx) * LAB_WHITE_X;
    float y = lab_f_inv(fy) * LAB_WHITE_Y;
    float z = lab_f_inv(fz) * LAB_WHITE_Z;

    float r = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z;
    float g = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z;
    float b = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z;

    return fclamp_s(
        float3(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)), 0.0, 1.0);
}
