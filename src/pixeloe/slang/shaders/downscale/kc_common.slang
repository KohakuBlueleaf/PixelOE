// One k-centroid iteration of one p x p block (shared by kcentroid.slang and
// kcentroid_atomic.slang): reassign the block's pixels to the nearer of
// c0, c1, replace each centroid by its members' exact mean (unchanged when
// empty), store (c0, c1, n0, n1) and return max |new - old|.

module kc_common;

import fixed;

public static const uint CENT_STRIDE = 8;

public float3 load_px(StructuredBuffer<float> img, uint base, uint hw, uint idx)
{
    return float3(img[base + idx], img[base + hw + idx], img[base + 2 * hw + idx]);
}

public float sqdist(float3 a, float3 c)
{
    float3 d = a - c;
    d = d * d;
    return d.x + d.y + d.z;
}

public float kc_update_block(
    StructuredBuffer<float> img,
    RWStructuredBuffer<float> cent,
    uint b,
    uint oy,
    uint ox,
    uint height,
    uint width,
    uint p,
    uint out_h,
    uint out_w)
{
    uint hw = height * width;
    uint base = b * 3 * hw;
    uint blk = (b * out_h + oy) * out_w + ox;
    uint o = blk * CENT_STRIDE;
    float3 c0 = float3(cent[o + 0], cent[o + 1], cent[o + 2]);
    float3 c1 = float3(cent[o + 3], cent[o + 4], cent[o + 5]);
    Fixed3 s0 = Fixed3();
    Fixed3 s1 = Fixed3();
    float n0 = 0.0;
    float n1 = 0.0;
    for (uint dy = 0; dy < p; dy++)
        for (uint dx = 0; dx < p; dx++)
        {
            float3 v = load_px(img, base, hw, (oy * p + dy) * width + ox * p + dx);
            if (sqdist(v, c1) < sqdist(v, c0))
            {
                s1.add(v, 1);
                n1 += 1.0;
            }
            else
            {
                s0.add(v, 1);
                n0 += 1.0;
            }
        }
    float3 new0 = n0 > 0.0 ? s0.mean(n0) : c0;
    float3 new1 = n1 > 0.0 ? s1.mean(n1) : c1;
    float3 d = max(abs(new0 - c0), abs(new1 - c1));
    cent[o + 0] = new0.x;
    cent[o + 1] = new0.y;
    cent[o + 2] = new0.z;
    cent[o + 3] = new1.x;
    cent[o + 4] = new1.y;
    cent[o + 5] = new1.z;
    cent[o + 6] = n0;
    cent[o + 7] = n1;
    return max(d.x, max(d.y, d.z));
}
