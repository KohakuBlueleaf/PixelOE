// GPU-only (groupshared + barrier) centroid update of kmeans.slang: one
// workgroup of 256 threads per (b, k) sums the chunk partials grid-stride
// and combines them with a tree. The sums are exact 64-bit integers
// (fixed.slang), so the result equals km_update's serial sum bit for bit.

import fixed;
import stop;

static const uint GROUP = 256;
groupshared int64_t g_sx[GROUP];
groupshared int64_t g_sy[GROUP];
groupshared int64_t g_sz[GROUP];
groupshared int64_t g_n[GROUP];

// Assignment and per-chunk member sums of one iteration in one pass: the
// workgroup of chunk (b, c) labels its CHUNK pixels into groupshared memory,
// then thread k (and k + CHUNK, ...) sums the members of cluster k. Same
// run-flag protocol and part layout as km_assign + km_partial.
static const uint CHUNK = 64;
groupshared uint g_label[CHUNK];

bool decide_run_group(RWStructuredBuffer<uint> run, StructuredBuffer<float> item_diff, uint it, uint items)
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

// workgroup gid.x = b * chunks + c
[shader("compute")]
[numthreads(64, 1, 1)]
void km_step_group(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform StructuredBuffer<float> pix,
    uniform StructuredBuffer<float> weight,
    uniform StructuredBuffer<float> cent,
    uniform StructuredBuffer<uint> repeat,
    uniform RWStructuredBuffer<int64_t> part,
    uniform StructuredBuffer<float> item_diff,
    uniform RWStructuredBuffer<uint> run,
    uniform uint it,
    uniform uint K,
    uniform uint count,
    uniform uint batch,
    uniform uint chunks,
    uniform uint weighted,
    uniform uint use_repeat)
{
    bool running = decide_run_group(run, item_diff, it, batch * K);
    if (gid.x == 0 && lid.x == 0)
        run[it] = running ? 1u : 0u;
    if (!running)
        return;
    uint b = gid.x / chunks;
    uint c = gid.x - b * chunks;
    uint n0 = c * CHUNK;
    uint n = n0 + lid.x;
    uint label = K; // no pixel
    if (n < count)
    {
        uint base = b * 3 * count + n;
        float3 v = float3(pix[base], pix[base + count], pix[base + 2 * count]);
        float w = weighted != 0 ? weight[b * count + n] : 1.0;
        float best_d = asfloat(0x7f800000u);
        label = 0;
        for (uint k = 0; k < K; k++)
        {
            uint o = (b * K + k) * 3;
            float3 d = v - float3(cent[o], cent[o + 1], cent[o + 2]);
            d = d * d;
            if (weighted != 0)
                d = d * w;
            float s = d.x + d.y + d.z;
            if (s < best_d)
            {
                best_d = s;
                label = k;
            }
        }
    }
    g_label[lid.x] = label;
    GroupMemoryBarrierWithGroupSync();

    uint end = min(CHUNK, count - n0);
    for (uint k = lid.x; k < K; k += CHUNK)
    {
        Fixed3 s = Fixed3();
        int64_t members = 0;
        for (uint j = 0; j < end; j++)
        {
            if (g_label[j] == k)
            {
                uint nj = n0 + j;
                uint base = b * 3 * count + nj;
                int64_t m = use_repeat != 0 ? int64_t(repeat[b * count + nj]) : 1;
                s.add(float3(pix[base], pix[base + count], pix[base + 2 * count]), m);
                members += m;
            }
        }
        uint o = (gid.x * K + k) * 4;
        part[o] = s.x;
        part[o + 1] = s.y;
        part[o + 2] = s.z;
        part[o + 3] = members;
    }
}

// workgroup t = b * K + k
[shader("compute")]
[numthreads(256, 1, 1)]
void km_update_group(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform RWStructuredBuffer<int64_t> part,
    uniform RWStructuredBuffer<float> cent,
    uniform RWStructuredBuffer<float> item_diff,
    uniform StructuredBuffer<uint> run,
    uniform uint it,
    uniform uint K,
    uniform uint chunks)
{
    if (run[it] == 0)
        return;
    uint t = gid.x;
    uint b = t / K;
    uint k = t - b * K;
    int64_t sx = 0;
    int64_t sy = 0;
    int64_t sz = 0;
    int64_t n = 0;
    for (uint c = lid.x; c < chunks; c += GROUP)
    {
        uint o = ((b * chunks + c) * K + k) * 4;
        sx += part[o];
        sy += part[o + 1];
        sz += part[o + 2];
        n += part[o + 3];
    }
    g_sx[lid.x] = sx;
    g_sy[lid.x] = sy;
    g_sz[lid.x] = sz;
    g_n[lid.x] = n;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = GROUP / 2; s > 0; s >>= 1)
    {
        if (lid.x < s)
        {
            g_sx[lid.x] += g_sx[lid.x + s];
            g_sy[lid.x] += g_sy[lid.x + s];
            g_sz[lid.x] += g_sz[lid.x + s];
            g_n[lid.x] += g_n[lid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid.x != 0)
        return;
    Fixed3 total = Fixed3();
    total.x = g_sx[0];
    total.y = g_sy[0];
    total.z = g_sz[0];
    int64_t members = g_n[0];
    uint o = t * 3;
    float3 old = float3(cent[o], cent[o + 1], cent[o + 2]);
    float3 upd = members > 0 ? total.mean(float(members)) : old;
    float3 d = abs(upd - old);
    item_diff[t] = max(d.x, max(d.y, d.z));
    cent[o] = upd.x;
    cent[o + 1] = upd.y;
    cent[o + 2] = upd.z;
}
