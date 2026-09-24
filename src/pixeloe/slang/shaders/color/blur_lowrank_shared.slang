// GPU-only (groupshared + barrier) passes of blur_lowrank.slang with a
// compile-time radius R. A thread owns OUT consecutive outputs along the pass
// axis and streams the OUT + 2R inputs they share from groupshared memory
// once, adding each into every output it reaches: OUT + 2R reads and
// OUT (2R + 1) FMAs per thread, with only OUT accumulators and the 2R + 1
// taps in registers.
//
// rows: 32 x 4 threads, a warp per image row; a row segment of 32 OUT
//       outputs + 2R halo is staged with coalesced loads, padded by one
//       float per 32 so that threads reading at stride OUT hit distinct banks.
// cols: 32 x 8 threads, 32 columns x 8 OUT rows of outputs; the
//       (8 OUT + 2R) x 32 strip is staged with coalesced loads, adjacent
//       lanes read adjacent columns, stores are coalesced.

import blur_pad;

static const int OUT = 8;
static const int ROW_SEG = 32 * OUT;
static const int ROWS_PER_GROUP = 4;
static const int ROW_CAP = ROW_SEG + 64 + (ROW_SEG + 64) / 32 + 1; // R <= 32
static const int COL_SEG = 8 * OUT;
static const int COL_CAP = (COL_SEG + 64) * 32; // R <= 32
groupshared float g_rows[ROWS_PER_GROUP * ROW_CAP];
groupshared float g_cols[COL_CAP];

int padded(int i)
{
    return i + i / 32;
}

void rows_stream<let R : int>(
    uint3 gid,
    uint3 lid,
    StructuredBuffer<float> src,
    RWStructuredBuffer<float> tmp,
    StructuredBuffer<float> tables,
    uint offset,
    uint height,
    uint width,
    uint reflect)
{
    uint y = gid.y * ROWS_PER_GROUP + lid.y;
    bool row_ok = y < height;
    uint row = gid.z * height * width + min(y, height - 1) * width;
    int x0 = int(gid.x) * ROW_SEG;
    int sbase = int(lid.y) * ROW_CAP;
    for (int i = int(lid.x); i < ROW_SEG + 2 * R; i += 32)
        g_rows[sbase + padded(i)] = src[row + uint(pad_index(x0 - R + i, int(width), reflect))];
    GroupMemoryBarrierWithGroupSync();

    float t[2 * R + 1];
    [unroll]
    for (int k = 0; k <= 2 * R; k++)
        t[k] = tables[offset + uint(k)];
    float acc[OUT];
    [unroll]
    for (int o = 0; o < OUT; o++)
        acc[o] = 0.0;
    int s0 = int(lid.x) * OUT;
    [unroll]
    for (int i = 0; i < OUT + 2 * R; i++)
    {
        float v = g_rows[sbase + padded(s0 + i)];
        [unroll]
        for (int o = 0; o < OUT; o++)
        {
            if (i - o >= 0 && i - o <= 2 * R)
                acc[o] += t[i - o] * v;
        }
    }
    // results back through the staged row (each warp owns its row), then
    // lane-contiguous stores
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (int o = 0; o < OUT; o++)
        g_rows[sbase + padded(s0 + o)] = acc[o];
    GroupMemoryBarrierWithGroupSync();
    if (!row_ok)
        return;
    for (int i = int(lid.x); i < ROW_SEG; i += 32)
    {
        int x = x0 + i;
        if (x < int(width))
            tmp[row + uint(x)] = g_rows[sbase + padded(i)];
    }
}

// mode 0: dst = acc; 1: dst = add_src + acc; 2: dst += acc
void cols_stream<let R : int>(
    uint3 gid,
    uint3 lid,
    StructuredBuffer<float> tmp,
    StructuredBuffer<float> add_src,
    RWStructuredBuffer<float> dst,
    StructuredBuffer<float> tables,
    uint offset,
    uint height,
    uint width,
    uint reflect,
    uint mode)
{
    uint base = gid.z * height * width;
    uint x = gid.x * 32 + lid.x;
    uint xc = min(x, width - 1);
    int y0 = int(gid.y) * COL_SEG;
    for (int i = int(lid.y); i < COL_SEG + 2 * R; i += 8)
        g_cols[i * 32 + int(lid.x)] = tmp[base + uint(pad_index(y0 - R + i, int(height), reflect)) * width + xc];
    GroupMemoryBarrierWithGroupSync();
    if (x >= width)
        return;

    float t[2 * R + 1];
    [unroll]
    for (int k = 0; k <= 2 * R; k++)
        t[k] = tables[offset + uint(k)];
    float acc[OUT];
    [unroll]
    for (int o = 0; o < OUT; o++)
        acc[o] = 0.0;
    int s0 = int(lid.y) * OUT;
    [unroll]
    for (int i = 0; i < OUT + 2 * R; i++)
    {
        float v = g_cols[(s0 + i) * 32 + int(lid.x)];
        [unroll]
        for (int o = 0; o < OUT; o++)
        {
            if (i - o >= 0 && i - o <= 2 * R)
                acc[o] += t[i - o] * v;
        }
    }
    [unroll]
    for (int o = 0; o < OUT; o++)
    {
        int y = y0 + s0 + o;
        if (y < int(height))
        {
            uint idx = base + uint(y) * width + x;
            float r = acc[o];
            if (mode == 1)
                r += add_src[idx];
            else if (mode == 2)
                r += dst[idx];
            dst[idx] = r;
        }
    }
}

// Host dispatches rows: (ceil(W / ROW_SEG) * 32, ceil(H / 4) * 4, planes),
// cols: (ceil(W / 32) * 32, ceil(H / COL_SEG) * 8, planes).
#define LR_ROWS_SHARED(NAME, RADIUS)                                        \
    [shader("compute")]                                                     \
    [numthreads(32, 4, 1)]                                                  \
    void NAME(                                                              \
        uint3 gid: SV_GroupID,                                              \
        uint3 lid: SV_GroupThreadID,                                        \
        uniform StructuredBuffer<float> src,                                \
        uniform RWStructuredBuffer<float> tmp,                              \
        uniform StructuredBuffer<float> tables,                             \
        uniform uint offset,                                                \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint reflect)                                               \
    {                                                                       \
        rows_stream<RADIUS>(gid, lid, src, tmp, tables, offset, height,     \
                            width, reflect);                                \
    }

#define LR_COLS_SHARED(NAME, RADIUS)                                        \
    [shader("compute")]                                                     \
    [numthreads(32, 8, 1)]                                                  \
    void NAME(                                                              \
        uint3 gid: SV_GroupID,                                              \
        uint3 lid: SV_GroupThreadID,                                        \
        uniform StructuredBuffer<float> tmp,                                \
        uniform StructuredBuffer<float> add_src,                            \
        uniform RWStructuredBuffer<float> dst,                              \
        uniform StructuredBuffer<float> tables,                             \
        uniform uint offset,                                                \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint reflect,                                               \
        uniform uint mode)                                                  \
    {                                                                       \
        cols_stream<RADIUS>(gid, lid, tmp, add_src, dst, tables, offset,    \
                            height, width, reflect, mode);                  \
    }

LR_ROWS_SHARED(lr_rows_shared_r2, 2)
LR_ROWS_SHARED(lr_rows_shared_r4, 4)
LR_ROWS_SHARED(lr_rows_shared_r8, 8)
LR_ROWS_SHARED(lr_rows_shared_r16, 16)
LR_ROWS_SHARED(lr_rows_shared_r32, 32)
LR_COLS_SHARED(lr_cols_shared_r2, 2)
LR_COLS_SHARED(lr_cols_shared_r4, 4)
LR_COLS_SHARED(lr_cols_shared_r8, 8)
LR_COLS_SHARED(lr_cols_shared_r16, 16)
LR_COLS_SHARED(lr_cols_shared_r32, 32)
