// Batched k-means palette over planar pixels pix[b, c, n] (c = 3).
// cent[b, k, c]; labels[b, n]; optional per-pixel weight (weighted
// assignment, reference semantics: each channel's squared difference is
// multiplied by the weight before summing) or integer count (repeat mode:
// the centroid update counts pixel n count[b, n] times).
//
// Global early stop as in the reference (`if max|new - old| < 1/256: break`
// over the whole batch): each update writes its centroid's |new - old| max
// to item_diff[b * K + k]; the next iteration's km_assign threads each fold
// those B*K values (a small, cached broadcast read) and agree on
//   run[it] = it == 0 or (run[it - 1] and max item_diff >= 1/256),
// which km_partial / km_update of the same iteration then read. This keeps
// the loop at three dispatches per iteration. run[] starts zeroed.
// Member sums are exact (fixed.slang): the means feed back into the next
// assignment, so they must not depend on summation order.

import fixed;
import stop;

bool decide_run(RWStructuredBuffer<uint> run, StructuredBuffer<float> item_diff, uint it, uint items)
{
    if (it == 0)
        return true;
    if (run[it - 1] == 0)
        return false;
    float m = 0.0;
    for (uint i = 0; i < items; i++)
        m = max(m, item_diff[i]);
    return m >= STOP_EPS;
}


float3 load_px(StructuredBuffer<float> pix, uint b, uint n, uint count)
{
    uint base = b * 3 * count + n;
    return float3(pix[base], pix[base + count], pix[base + 2 * count]);
}

float3 load_cent(RWStructuredBuffer<float> cent, uint b, uint k, uint K)
{
    uint o = (b * K + k) * 3;
    return float3(cent[o], cent[o + 1], cent[o + 2]);
}

float3 load_cent(StructuredBuffer<float> cent, uint b, uint k, uint K)
{
    uint o = (b * K + k) * 3;
    return float3(cent[o], cent[o + 1], cent[o + 2]);
}

// cent[b, k, c] = interp[k, c] * min[b, c] + (1 - interp[k, c]) * max[b, c]
[shader("compute")]
[numthreads(64, 1, 1)]
void km_init(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> interp,
    uniform StructuredBuffer<float> ch_min, // [b * 3 + c]
    uniform StructuredBuffer<float> ch_max,
    uniform RWStructuredBuffer<float> cent,
    uniform uint K,
    uniform uint batch)
{
    uint t = tid.x;
    if (t >= batch * K)
        return;
    uint b = t / K;
    uint k = t - b * K;
    for (uint c = 0; c < 3; c++)
    {
        float a = interp[k * 3 + c];
        cent[t * 3 + c] = a * ch_min[b * 3 + c] + (1.0 - a) * ch_max[b * 3 + c];
    }
}

uint nearest(float3 v, StructuredBuffer<float> cent, uint b, uint K, float w, bool weighted)
{
    uint best = 0;
    float best_d = asfloat(0x7f800000u);
    for (uint k = 0; k < K; k++)
    {
        float3 d = v - load_cent(cent, b, k, K);
        d = d * d;
        if (weighted)
            d = d * w;
        float s = d.x + d.y + d.z;
        if (s < best_d)
        {
            best_d = s;
            best = k;
        }
    }
    return best;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void km_assign(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> pix,
    uniform StructuredBuffer<float> weight,
    uniform StructuredBuffer<float> cent,
    uniform RWStructuredBuffer<uint> labels,
    uniform StructuredBuffer<float> item_diff,
    uniform RWStructuredBuffer<uint> run,
    uniform uint it,
    uniform uint K,
    uniform uint count,
    uniform uint batch,
    uniform uint weighted)
{
    uint t = tid.x;
    if (t >= batch * count)
        return;
    bool running = decide_run(run, item_diff, it, batch * K);
    if (t == 0)
        run[it] = running ? 1u : 0u;
    if (!running)
        return;
    uint b = t / count;
    uint n = t - b * count;
    float w = weighted != 0 ? weight[t] : 1.0;
    labels[t] = nearest(load_px(pix, b, n, count), cent, b, K, w, weighted != 0);
}

// One thread per (b, chunk, cluster k): part[b, chunk, k] = (fixed sums of
// r, g, b over the chunk's members of k, member count). Consecutive threads
// share the chunk, so each label load is uniform across the warp; the
// accumulators stay in registers.
[shader("compute")]
[numthreads(64, 1, 1)]
void km_partial(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> pix,
    uniform StructuredBuffer<uint> repeat,
    uniform StructuredBuffer<uint> labels,
    uniform RWStructuredBuffer<int64_t> part,
    uniform StructuredBuffer<uint> run,
    uniform uint it,
    uniform uint K,
    uniform uint count,
    uniform uint batch,
    uniform uint chunk,
    uniform uint chunks,
    uniform uint use_repeat)
{
    uint t = tid.x;
    if (t >= batch * chunks * K || run[it] == 0)
        return;
    uint k = t % K;
    uint bc = t / K;
    uint b = bc / chunks;
    uint c = bc - b * chunks;
    Fixed3 s = Fixed3();
    int64_t members = 0;
    uint begin = c * chunk;
    uint end = min(begin + chunk, count);
    for (uint n = begin; n < end; n++)
    {
        if (labels[b * count + n] == k)
        {
            int64_t m = use_repeat != 0 ? int64_t(repeat[b * count + n]) : 1;
            s.add(load_px(pix, b, n, count), m);
            members += m;
        }
    }
    uint o = t * 4;
    part[o] = s.x;
    part[o + 1] = s.y;
    part[o + 2] = s.z;
    part[o + 3] = members;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void km_update(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<int64_t> part,
    uniform RWStructuredBuffer<float> cent,
    uniform RWStructuredBuffer<float> item_diff,
    uniform StructuredBuffer<uint> run,
    uniform uint it,
    uniform uint K,
    uniform uint batch,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= batch * K || run[it] == 0)
        return;
    uint b = t / K;
    uint k = t - b * K;
    Fixed3 s = Fixed3();
    int64_t n = 0;
    for (uint c = 0; c < chunks; c++)
    {
        uint o = ((b * chunks + c) * K + k) * 4;
        s.x += part[o];
        s.y += part[o + 1];
        s.z += part[o + 2];
        n += part[o + 3];
    }
    float3 old = load_cent(cent, b, k, K);
    float3 upd = n > 0 ? s.mean(float(n)) : old;
    float3 d = abs(upd - old);
    item_diff[t] = max(d.x, max(d.y, d.z));
    cent[t * 3] = upd.x;
    cent[t * 3 + 1] = upd.y;
    cent[t * 3 + 2] = upd.z;
}

// Final unweighted assignment: labels and quantised pixels
[shader("compute")]
[numthreads(256, 1, 1)]
void km_final(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> pix,
    uniform StructuredBuffer<float> cent,
    uniform RWStructuredBuffer<uint> labels,
    uniform RWStructuredBuffer<float> quant,
    uniform uint K,
    uniform uint count,
    uniform uint batch)
{
    uint t = tid.x;
    if (t >= batch * count)
        return;
    uint b = t / count;
    uint n = t - b * count;
    uint l = nearest(load_px(pix, b, n, count), cent, b, K, 1.0, false);
    labels[t] = l;
    float3 c = load_cent(cent, b, l, K);
    uint base = b * 3 * count + n;
    quant[base] = c.x;
    quant[base + count] = c.y;
    quant[base + 2 * count] = c.z;
}
