// Contrast-based downscale over p x p blocks of a planar Lab image.
// L: mid = element p*p/2 (row-major), med = lower median, mu = mean;
//    med < mu and (max - med) > (med - min)  -> min
//    med > mu and (max - med) < (med - min)  -> max
//    otherwise                               -> mid
// a, b: lower median. Output converted back to RGB.

import lab;
import order;
import kselect;

struct BlockKeys : IKeySource
{
    StructuredBuffer<float> src;
    uint base; // plane base + block origin
    uint p;
    uint width;

    uint count() { return p * p; }

    uint key(uint i)
    {
        uint dy = i / p;
        return float_to_ordered(src[base + dy * width + (i - dy * p)]);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void contrast_downscale(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lab_img,
    uniform RWStructuredBuffer<float> dst,
    uniform uint height,
    uniform uint width,
    uniform uint p,
    uniform uint out_h,
    uniform uint out_w)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint b = tid.z;
    if (ox >= out_w || oy >= out_h)
        return;
    uint hw = height * width;
    uint origin = (oy * p) * width + ox * p;
    uint plane_l = (b * 3) * hw;

    BlockKeys keys;
    keys.src = lab_img;
    keys.p = p;
    keys.width = width;

    keys.base = plane_l + origin;
    float med = radix_median(keys);
    keys.base = plane_l + hw + origin;
    float med_a = radix_median(keys);
    keys.base = plane_l + 2 * hw + origin;
    float med_b = radix_median(keys);

    float sum = 0.0;
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    for (uint dy = 0; dy < p; dy++)
    {
        uint row = plane_l + origin + dy * width;
        for (uint dx = 0; dx < p; dx++)
        {
            float v = lab_img[row + dx];
            sum += v;
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    float mu = sum / float(p * p);
    uint mid_i = (p * p) / 2;
    uint mid_y = mid_i / p;
    float mid = lab_img[plane_l + origin + mid_y * width + (mid_i - mid_y * p)];

    float l_out = mid;
    if (med < mu && (mx - med) > (med - mn))
        l_out = mn;
    if (med > mu && (mx - med) < (med - mn))
        l_out = mx;

    float3 rgb = lab_to_rgb(float3(l_out, med_a, med_b));
    uint ohw = out_h * out_w;
    uint o = oy * out_w + ox;
    dst[(b * 3) * ohw + o] = rgb.x;
    dst[(b * 3 + 1) * ohw + o] = rgb.y;
    dst[(b * 3 + 2) * ohw + o] = rgb.z;
}

// contrast_downscale for a compile-time block size P: the block's L, a and b
// values are loaded once into registers; medians by register selection.
// from_rgb: the source is sRGB, converted to Lab here (blocks do not
// overlap, so each pixel is converted once).
void contrast_regs<let P : int>(
    uint3 tid,
    StructuredBuffer<float> lab_img,
    RWStructuredBuffer<float> dst,
    uint height,
    uint width,
    uint out_h,
    uint out_w,
    bool from_rgb,
    uint up)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint b = tid.z;
    if (ox >= out_w || oy >= out_h)
        return;
    uint hw = height * width;
    uint origin = (b * 3) * hw + (oy * uint(P)) * width + ox * uint(P);
    uint kl[P * P];
    uint ka[P * P];
    uint kb[P * P];
    float sum = 0.0;
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    [unroll]
    for (int dy = 0; dy < P; dy++)
    {
        [unroll]
        for (int dx = 0; dx < P; dx++)
        {
            uint at = origin + uint(dy) * width + uint(dx);
            float3 v = float3(lab_img[at], lab_img[at + hw], lab_img[at + 2 * hw]);
            if (from_rgb)
                v = rgb_to_lab(v);
            sum += v.x;
            mn = min(mn, v.x);
            mx = max(mx, v.x);
            kl[dy * P + dx] = float_to_ordered(v.x);
            ka[dy * P + dx] = float_to_ordered(v.y);
            kb[dy * P + dx] = float_to_ordered(v.z);
        }
    }
    float med = register_median<P * P>(kl);
    float mu = sum / float(P * P);
    float l_out = ordered_to_float(kl[(P * P) / 2]);
    if (med < mu && (mx - med) > (med - mn))
        l_out = mn;
    if (med > mu && (mx - med) < (med - mn))
        l_out = mx;
    float3 rgb = lab_to_rgb(float3(l_out, register_median<P * P>(ka), register_median<P * P>(kb)));
    if (up != 0)
    {
        // nearest-exact upscale by P: the block's P x P pixels take the value
        uint uw = out_w * uint(P);
        uint uhw = out_h * uint(P) * uw;
        [unroll]
        for (int dy = 0; dy < P; dy++)
        {
            uint row = (oy * uint(P) + uint(dy)) * uw + ox * uint(P);
            [unroll]
            for (int dx = 0; dx < P; dx++)
            {
                dst[(b * 3) * uhw + row + uint(dx)] = rgb.x;
                dst[(b * 3 + 1) * uhw + row + uint(dx)] = rgb.y;
                dst[(b * 3 + 2) * uhw + row + uint(dx)] = rgb.z;
            }
        }
        return;
    }
    uint ohw = out_h * out_w;
    uint o = oy * out_w + ox;
    dst[(b * 3) * ohw + o] = rgb.x;
    dst[(b * 3 + 1) * ohw + o] = rgb.y;
    dst[(b * 3 + 2) * ohw + o] = rgb.z;
}

// lab_img holds Lab (contrast_downscale_p*) or sRGB (contrast_rgb_p*);
// up != 0: dst is the nearest-exact upscale (out_h P x out_w P)
#define CONTRAST_P(NAME, P, FROM_RGB)                                       \
    [shader("compute")]                                                     \
    [numthreads(16, 16, 1)]                                                 \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> lab_img,                            \
        uniform RWStructuredBuffer<float> dst,                              \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint out_h,                                                 \
        uniform uint out_w,                                                 \
        uniform uint up)                                                    \
    {                                                                       \
        contrast_regs<P>(tid, lab_img, dst, height, width, out_h, out_w,    \
                         FROM_RGB, up);                                     \
    }

CONTRAST_P(contrast_downscale_p2, 2, false)
CONTRAST_P(contrast_downscale_p3, 3, false)
CONTRAST_P(contrast_downscale_p4, 4, false)
CONTRAST_P(contrast_downscale_p5, 5, false)
CONTRAST_P(contrast_downscale_p6, 6, false)
CONTRAST_P(contrast_downscale_p8, 8, false)
CONTRAST_P(contrast_rgb_p2, 2, true)
CONTRAST_P(contrast_rgb_p3, 3, true)
CONTRAST_P(contrast_rgb_p4, 4, true)
CONTRAST_P(contrast_rgb_p5, 5, true)
CONTRAST_P(contrast_rgb_p6, 6, true)
CONTRAST_P(contrast_rgb_p8, 8, true)
