// k-centroid downscale (k = 2) over p x p RGB blocks.
// cent[blk] = (c0.rgb, c1.rgb, n0, n1); init c0 = per-channel max, c1 = min.
// Iterations stop globally as in the reference (see stopdiff.slang); each
// iteration writes its block's max |new - old| to item_diff[blk].
// Member sums are exact (fixed.slang), so means are correctly rounded
// regardless of summation order.
//
// Output: argmin_k sum_x |x - c_k|^2 (ties -> c0). When c0, c1 are the means
// of the last partition (sizes n0, n1 > 0), sum_x |x - c_k|^2 =
// S + n |m - c_k|^2 with m = (n0 c0 + n1 c1) / n, so the SSE of c0 is not
// larger iff n0 >= n1 (or c0 == c1): the majority cluster wins, exact ties
// go to c0. The reference evaluates this with float sums whose rounding
// decides every n0 == n1 block arbitrarily (torch CPU and CUDA disagree on
// such blocks); this kernel evaluates the exact criterion.

import fixed;
import stop;
import kc_common;

[shader("compute")]
[numthreads(16, 16, 1)]
void kc_init(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform RWStructuredBuffer<float> cent,
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
    uint base = b * 3 * hw;
    float3 mn = asfloat(0x7f800000u);
    float3 mx = -asfloat(0x7f800000u);
    for (uint dy = 0; dy < p; dy++)
        for (uint dx = 0; dx < p; dx++)
        {
            float3 v = load_px(img, base, hw, (oy * p + dy) * width + ox * p + dx);
            mn = min(mn, v);
            mx = max(mx, v);
        }
    uint o = ((b * out_h + oy) * out_w + ox) * CENT_STRIDE;
    cent[o + 0] = mx.x;
    cent[o + 1] = mx.y;
    cent[o + 2] = mx.z;
    cent[o + 3] = mn.x;
    cent[o + 4] = mn.y;
    cent[o + 5] = mn.z;
    cent[o + 6] = 0.0;
    cent[o + 7] = 0.0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void kc_iter(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform RWStructuredBuffer<float> cent,
    uniform RWStructuredBuffer<float> item_diff,
    uniform RWStructuredBuffer<float> diff,
    uniform uint it,
    uniform uint height,
    uniform uint width,
    uniform uint p,
    uniform uint out_h,
    uniform uint out_w)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint b = tid.z;
    if (ox >= out_w || oy >= out_h || stopped(diff, it))
        return;
    uint blk = (b * out_h + oy) * out_w + ox;
    item_diff[blk] = kc_update_block(img, cent, b, oy, ox, height, width, p, out_h, out_w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void kc_final(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform StructuredBuffer<float> cent,
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
    uint base = b * 3 * hw;
    uint o = ((b * out_h + oy) * out_w + ox) * CENT_STRIDE;
    float3 c0 = float3(cent[o + 0], cent[o + 1], cent[o + 2]);
    float3 c1 = float3(cent[o + 3], cent[o + 4], cent[o + 5]);
    float n0 = cent[o + 6];
    float n1 = cent[o + 7];
    bool pick1;
    if (n0 > 0.0 && n1 > 0.0)
    {
        pick1 = n1 > n0;
    }
    else
    {
        // a stale (empty-cluster) centroid breaks the mean identity: sum up
        float d0 = 0.0;
        float d1 = 0.0;
        for (uint dy = 0; dy < p; dy++)
            for (uint dx = 0; dx < p; dx++)
            {
                float3 v = load_px(img, base, hw, (oy * p + dy) * width + ox * p + dx);
                d0 += sqdist(v, c0);
                d1 += sqdist(v, c1);
            }
        pick1 = d1 < d0;
    }
    float3 c = pick1 ? c1 : c0;
    uint ohw = out_h * out_w;
    uint px = oy * out_w + ox;
    dst[(b * 3) * ohw + px] = c.x;
    dst[(b * 3 + 1) * ohw + px] = c.y;
    dst[(b * 3 + 2) * ohw + px] = c.z;
}
