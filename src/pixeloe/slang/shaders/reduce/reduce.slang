// Deterministic segmented tree reductions (no float atomics, fixed order).
// One level: thread (s, c) folds elements [c * chunk, (c + 1) * chunk) of
// segment s (s * seg_len + src_offset ..) into part[s * chunks + c]. The
// host repeats levels over the partials until one value per segment is
// left, so every level has plenty of threads and short serial loops.

import kahan;
import lab;

float f32_inf()
{
    return asfloat(0x7f800000u);
}

// min over src_min, max over src_max (the same buffer on the first level)
[shader("compute")]
[numthreads(256, 1, 1)]
void seg_minmax_level(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src_min,
    uniform StructuredBuffer<float> src_max,
    uniform RWStructuredBuffer<float> part_min,
    uniform RWStructuredBuffer<float> part_max,
    uniform uint src_offset,
    uniform uint seg_count,
    uniform uint seg_len,
    uniform uint chunk,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= seg_count * chunks)
        return;
    uint seg = t / chunks;
    uint begin = (t - seg * chunks) * chunk;
    uint end = min(begin + chunk, seg_len);
    uint base = src_offset + seg * seg_len;
    float mn = f32_inf();
    float mx = -f32_inf();
    for (uint i = begin; i < end; i++)
    {
        mn = min(mn, src_min[base + i]);
        mx = max(mx, src_max[base + i]);
    }
    part_min[t] = mn;
    part_max[t] = mx;
}

// part = scale * sum (scale = 1 except on the last level)
[shader("compute")]
[numthreads(256, 1, 1)]
void seg_sum_level(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> part,
    uniform uint src_offset,
    uniform uint seg_count,
    uniform uint seg_len,
    uniform uint chunk,
    uniform uint chunks,
    uniform float scale)
{
    uint t = tid.x;
    if (t >= seg_count * chunks)
        return;
    uint seg = t / chunks;
    uint begin = (t - seg * chunks) * chunk;
    uint end = min(begin + chunk, seg_len);
    uint base = src_offset + seg * seg_len;
    Kahan1 acc = Kahan1();
    for (uint i = begin; i < end; i++)
        acc.add(src[base + i]);
    part[t] = acc.sum * scale;
}

// ---------------------------------------------------------------------------
// Moments of Lab values pooled over the whole batch and all three channels
// (torch.mean / torch.std of an rgb_to_lab tensor). Welford per thread on
// the first level, Chan merges above it; std is unbiased (n - 1).
// Partials are (n, mean, m2) triplets.

struct Moments
{
    float n;
    float mean;
    float m2;

    [mutating]
    void add(float v)
    {
        n += 1.0;
        float d = v - mean;
        mean += d / n;
        m2 += d * (v - mean);
    }

    [mutating]
    void merge(float nb, float meanb, float m2b)
    {
        if (nb == 0.0)
            return;
        float total = n + nb;
        float d = meanb - mean;
        mean += d * (nb / total);
        m2 += m2b + d * d * (n * nb / total);
        n = total;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lab_moments_partial(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform RWStructuredBuffer<float> part,
    uniform uint pixels, // B * H * W
    uniform uint hw,
    uniform uint chunk,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= chunks)
        return;
    uint begin = t * chunk;
    uint end = min(begin + chunk, pixels);
    Moments m = { 0.0, 0.0, 0.0 };
    for (uint q = begin; q < end; q++)
    {
        uint b = q / hw;
        uint base = b * 3 * hw + (q - b * hw);
        float3 lab = rgb_to_lab(float3(img[base], img[base + hw], img[base + 2 * hw]));
        m.add(lab.x);
        m.add(lab.y);
        m.add(lab.z);
    }
    part[t * 3 + 0] = m.n;
    part[t * 3 + 1] = m.mean;
    part[t * 3 + 2] = m.m2;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void moments_merge(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> part,
    uniform uint count, // partials in src
    uniform uint chunk,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= chunks)
        return;
    uint end = min((t + 1) * chunk, count);
    Moments m = { 0.0, 0.0, 0.0 };
    for (uint c = t * chunk; c < end; c++)
        m.merge(src[c * 3 + 0], src[c * 3 + 1], src[c * 3 + 2]);
    part[t * 3 + 0] = m.n;
    part[t * 3 + 1] = m.mean;
    part[t * 3 + 2] = m.m2;
}

// stats[slot * 2 + 0] = mean, stats[slot * 2 + 1] = unbiased std
[shader("compute")]
[numthreads(1, 1, 1)]
void moments_finalize(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> part,
    uniform RWStructuredBuffer<float> stats,
    uniform uint slot)
{
    stats[slot * 2 + 0] = part[1];
    stats[slot * 2 + 1] = sqrt(part[2] / (part[0] - 1.0));
}
