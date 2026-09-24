// GPU-only (groupshared + barrier) two-level reductions: a level folds each
// segment into `groups` partials, one per workgroup of 256 threads; thread
// t of workgroup g reads elements g * 256 + t + k * groups * 256
// (coalesced), then the workgroup combines its 256 values with a fixed
// tree. Level 1 over the data, level 2 (groups = 1) over the partials: two
// dispatches, deterministic order.

import lab;

static const uint GROUP = 256;
groupshared float g_x[GROUP];
groupshared float g_y[GROUP];
groupshared float g_z[GROUP];

// part_min / part_max[seg * groups + g]; src_min / src_max may alias
[shader("compute")]
[numthreads(256, 1, 1)]
void seg_minmax_group(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform StructuredBuffer<float> src_min,
    uniform StructuredBuffer<float> src_max,
    uniform RWStructuredBuffer<float> part_min,
    uniform RWStructuredBuffer<float> part_max,
    uniform uint seg_len,
    uniform uint groups)
{
    uint seg = gid.y;
    uint base = seg * seg_len;
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    for (uint i = gid.x * GROUP + lid.x; i < seg_len; i += groups * GROUP)
    {
        mn = min(mn, src_min[base + i]);
        mx = max(mx, src_max[base + i]);
    }
    g_x[lid.x] = mn;
    g_y[lid.x] = mx;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = GROUP / 2; s > 0; s >>= 1)
    {
        if (lid.x < s)
        {
            g_x[lid.x] = min(g_x[lid.x], g_x[lid.x + s]);
            g_y[lid.x] = max(g_y[lid.x], g_y[lid.x + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid.x == 0)
    {
        part_min[seg * groups + gid.x] = g_x[0];
        part_max[seg * groups + gid.x] = g_y[0];
    }
}

// Moments (n, mean, m2) of Lab values pooled over pixels and channels
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

Moments tree_moments(Moments m, uint t)
{
    g_x[t] = m.n;
    g_y[t] = m.mean;
    g_z[t] = m.m2;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = GROUP / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            Moments a = { g_x[t], g_y[t], g_z[t] };
            a.merge(g_x[t + s], g_y[t + s], g_z[t + s]);
            g_x[t] = a.n;
            g_y[t] = a.mean;
            g_z[t] = a.m2;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    Moments r = { g_x[0], g_y[0], g_z[0] };
    return r;
}

// level 1, both images of a colour match at once (workgroup row y = image):
// part[(y * groups + g) * 3 ..] = moments of Lab(img_y) over workgroup g's pixels
[shader("compute")]
[numthreads(256, 1, 1)]
void lab_moments_group(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform StructuredBuffer<float> img0,
    uniform StructuredBuffer<float> img1,
    uniform RWStructuredBuffer<float> part,
    uniform uint pixels,
    uniform uint hw,
    uniform uint groups)
{
    Moments m = { 0.0, 0.0, 0.0 };
    for (uint q = gid.x * GROUP + lid.x; q < pixels; q += groups * GROUP)
    {
        uint b = q / hw;
        uint base = b * 3 * hw + (q - b * hw);
        float3 rgb = gid.y == 0
            ? float3(img0[base], img0[base + hw], img0[base + 2 * hw])
            : float3(img1[base], img1[base + hw], img1[base + 2 * hw]);
        float3 lab = rgb_to_lab(rgb);
        m.add(lab.x);
        m.add(lab.y);
        m.add(lab.z);
    }
    Moments r = tree_moments(m, lid.x);
    if (lid.x == 0)
    {
        uint o = (gid.y * groups + gid.x) * 3;
        part[o + 0] = r.n;
        part[o + 1] = r.mean;
        part[o + 2] = r.m2;
    }
}

// level 2, one workgroup per image y:
// stats[y * 2] = mean, stats[y * 2 + 1] = unbiased std
[shader("compute")]
[numthreads(256, 1, 1)]
void moments_final_group(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform StructuredBuffer<float> part,
    uniform RWStructuredBuffer<float> stats,
    uniform uint count)
{
    Moments m = { 0.0, 0.0, 0.0 };
    uint base = gid.y * count;
    for (uint c = lid.x; c < count; c += GROUP)
        m.merge(part[(base + c) * 3 + 0], part[(base + c) * 3 + 1], part[(base + c) * 3 + 2]);
    Moments r = tree_moments(m, lid.x);
    if (lid.x == 0)
    {
        stats[gid.y * 2 + 0] = r.mean;
        stats[gid.y * 2 + 1] = sqrt(r.m2 / (r.n - 1.0));
    }
}
