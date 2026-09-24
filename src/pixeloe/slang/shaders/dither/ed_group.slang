// GPU-only (barrier) single-dispatch variant of dither.slang's error
// diffusion: one 256-thread workgroup per image runs every row-pair step,
// with device-memory barriers between the error and update phases instead
// of two dispatches per step. Same arithmetic as ed_errors / ed_apply.

import palette;

static const uint GROUP = 256;

// Shared-memory form (1024 threads per image). Step y needs rows y, y+1,
// y+2: row y is the one the previous step updated last (kept in groupshared
// `carry`), rows y+1 and y+2 are still the input values. After the step,
// row y+1 is final (written out) and the updated row y+2 becomes the next
// carry; errors and palette live in groupshared too, so every barrier is
// workgroup-local. Needs 12 W + 3 K floats <= SHARED_CAP.
static const uint ED_THREADS = 1024;
static const uint SHARED_CAP = 8000;
groupshared float g_ed[SHARED_CAP];

[shader("compute")]
[numthreads(1024, 1, 1)]
void ed_rows_shared(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> pal,
    uniform uint K,
    uniform uint height,
    uniform uint width)
{
    uint b = gid.x;
    uint plane = height * width;
    uint W = width;
    uint carry0 = 0;     // [3 ch][W]
    uint err0 = 3 * W;   // [3 rows][3 ch][W]
    uint pal0 = 12 * W;  // [K][3]
    for (uint i = lid.x; i < K * 3; i += ED_THREADS)
        g_ed[pal0 + i] = pal[b * K * 3 + i];
    GroupMemoryBarrierWithGroupSync();

    uint img0 = b * 3 * plane;
    for (uint y = 0; y + 2 < height; y += 2)
    {
        for (uint i = lid.x; i < 3 * W; i += ED_THREADS)
        {
            uint r = i / W;
            uint x = i - r * W;
            float3 v;
            if (r == 0 && y > 0)
            {
                v = float3(g_ed[carry0 + x], g_ed[carry0 + W + x], g_ed[carry0 + 2 * W + x]);
                uint at = img0 + y * W + x; // row y is final: write it out
                img[at] = v.x;
                img[at + plane] = v.y;
                img[at + 2 * plane] = v.z;
            }
            else
            {
                uint at = img0 + (y + r) * W + x;
                v = float3(img[at], img[at + plane], img[at + 2 * plane]);
            }
            uint best = 0;
            float best_d = asfloat(0x7f800000u);
            for (uint k = 0; k < K; k++)
            {
                float3 p = float3(g_ed[pal0 + k * 3], g_ed[pal0 + k * 3 + 1], g_ed[pal0 + k * 3 + 2]);
                float d = dist2(v, p);
                if (d < best_d)
                {
                    best_d = d;
                    best = k;
                }
            }
            float3 e = v - float3(g_ed[pal0 + best * 3], g_ed[pal0 + best * 3 + 1], g_ed[pal0 + best * 3 + 2]);
            uint o = err0 + r * 3 * W + x;
            g_ed[o] = e.x;
            g_ed[o + W] = e.y;
            g_ed[o + 2 * W] = e.z;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = lid.x; i < 2 * W; i += ED_THREADS)
        {
            uint r = i / W;
            uint x = i - r * W;
            uint at = img0 + (y + 1 + r) * W + x;
            for (uint c = 0; c < 3; c++)
            {
                uint e0 = err0 + (r * 3 + c) * W;
                uint e1 = err0 + ((r + 1) * 3 + c) * W;
                float acc = 0.0;
                for (int j = 0; j < 3; j++)
                    acc += ED_K0[j] * g_ed[e0 + uint(reflect_col(int(x) + j - 1, int(W)))];
                for (int j = 0; j < 3; j++)
                    acc += ED_K1[j] * g_ed[e1 + uint(reflect_col(int(x) + j - 1, int(W)))];
                float v = img[at + c * plane] + acc;
                if (r == 0)
                    img[at + c * plane] = v; // row y + 1 is final
                else
                    g_ed[carry0 + c * W + x] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (height >= 3)
    {
        uint last = ((height - 3) / 2) * 2 + 2; // row held in carry
        for (uint i = lid.x; i < 3 * W; i += ED_THREADS)
        {
            uint c = i / W;
            uint x = i - c * W;
            img[img0 + c * plane + last * W + x] = g_ed[carry0 + i];
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ed_all_steps(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> pal,
    uniform RWStructuredBuffer<float> err, // [b, 3 rows, 3 ch, width]
    uniform uint K,
    uniform uint height,
    uniform uint width)
{
    uint b = gid.x;
    uint plane = height * width;
    for (uint y = 0; y + 2 < height; y += 2)
    {
        for (uint i = lid.x; i < 3 * width; i += GROUP)
        {
            uint r = i / width;
            uint x = i - r * width;
            uint base = b * 3 * plane + (y + r) * width + x;
            float3 v = load_px(img, base, plane);
            float3 e = v - load_pal(pal, b, nearest(v, pal, b, K), K);
            uint o = ((b * 3 + r) * 3) * width + x;
            err[o] = e.x;
            err[o + width] = e.y;
            err[o + 2 * width] = e.z;
        }
        AllMemoryBarrierWithGroupSync();
        for (uint i = lid.x; i < 2 * width; i += GROUP)
        {
            uint r = i / width;
            uint x = i - r * width;
            for (uint c = 0; c < 3; c++)
            {
                uint row0 = ((b * 3 + r) * 3 + c) * width;
                uint row1 = ((b * 3 + r + 1) * 3 + c) * width;
                float acc = 0.0;
                for (int j = 0; j < 3; j++)
                    acc += ED_K0[j] * err[row0 + uint(reflect_col(int(x) + j - 1, int(width)))];
                for (int j = 0; j < 3; j++)
                    acc += ED_K1[j] * err[row1 + uint(reflect_col(int(x) + j - 1, int(width)))];
                img[(b * 3 + c) * plane + (y + 1 + r) * width + x] += acc;
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
}
