// GPU-only (integer atomics) k-centroid iteration: each block folds its
// max |new - old| into diff_bits[it] with InterlockedMax. The differences are
// non-negative floats, whose bit patterns order like their values, so the
// result is the exact maximum in any order and replaces the separate
// diff_partial / diff_final reduction. diff_bits[] starts zeroed; iteration
// it runs while diff[it - 1] >= 1/256 (a skipped iteration leaves 0).

import stop;
import kc_common;

[shader("compute")]
[numthreads(16, 16, 1)]
void kc_iter_max(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform RWStructuredBuffer<float> cent,
    uniform RWStructuredBuffer<uint> diff_bits,
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
    if (ox >= out_w || oy >= out_h)
        return;
    if (it > 0 && asfloat(diff_bits[it - 1]) < STOP_EPS)
        return;
    float d = kc_update_block(img, cent, b, oy, ox, height, width, p, out_h, out_w);
    InterlockedMax(diff_bits[it], asuint(d));
}
