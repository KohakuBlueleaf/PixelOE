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
