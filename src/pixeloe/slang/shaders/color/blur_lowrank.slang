// Low-rank form of the exact colour-fix blur (every target, CPU included):
//
//     K = sum_{c < rank} col_c row_c^T      (SVD of the (2r+1)^2 table)
//     dst = add_src + sum_c cols_c(rows_c(pad(src)))
//
// One component is a row pass into tmp and a column pass into dst; the
// column pass of component 0 writes add_src + acc, later ones add to dst.
// tables: [rank, 2r+1] row vectors, then [rank, 2r+1] column vectors
// (singular value folded in).

import blur_pad;

[shader("compute")]
[numthreads(64, 4, 1)]
void lr_rows(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> tmp,
    uniform StructuredBuffer<float> tables,
    uniform uint offset, // start of this component's row vector
    uniform uint radius,
    uniform uint height,
    uniform uint width,
    uniform uint reflect)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= width || y >= height)
        return;
    int r = int(radius);
    uint row = plane * height * width + y * width;
    float acc = 0.0;
    for (int kx = 0; kx <= 2 * r; kx++)
        acc += tables[offset + uint(kx)] * src[row + uint(pad_index(int(x) + kx - r, int(width), reflect))];
    tmp[row + x] = acc;
}

// Register-blocked passes with a compile-time radius: a thread owns OUT
// consecutive outputs along the pass axis and loads the OUT + 2R inputs they
// share once, so each output costs (OUT + 2R) / OUT loads and 2R + 1 FMAs.
static const int OUT = 8;

void rows_blocked<let R : int>(
    uint3 tid,
    StructuredBuffer<float> src,
    RWStructuredBuffer<float> tmp,
    StructuredBuffer<float> tables,
    uint offset,
    uint height,
    uint width,
    uint reflect)
{
    int x0 = int(tid.x) * OUT;
    uint y = tid.y;
    if (x0 >= int(width) || y >= height)
        return;
    uint row = tid.z * height * width + y * width;
    float v[OUT + 2 * R];
    [unroll]
    for (int i = 0; i < OUT + 2 * R; i++)
        v[i] = src[row + uint(pad_index(x0 - R + i, int(width), reflect))];
    float t[2 * R + 1];
    [unroll]
    for (int k = 0; k <= 2 * R; k++)
        t[k] = tables[offset + uint(k)];
    [unroll]
    for (int o = 0; o < OUT; o++)
    {
        float acc = 0.0;
        [unroll]
        for (int k = 0; k <= 2 * R; k++)
            acc += t[k] * v[o + k];
        if (x0 + o < int(width))
            tmp[row + uint(x0 + o)] = acc;
    }
}

void cols_blocked<let R : int>(
    uint3 tid,
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
    uint x = tid.x;
    int y0 = int(tid.y) * OUT;
    if (x >= width || y0 >= int(height))
        return;
    uint base = tid.z * height * width;
    float v[OUT + 2 * R];
    [unroll]
    for (int i = 0; i < OUT + 2 * R; i++)
        v[i] = tmp[base + uint(pad_index(y0 - R + i, int(height), reflect)) * width + x];
    float t[2 * R + 1];
    [unroll]
    for (int k = 0; k <= 2 * R; k++)
        t[k] = tables[offset + uint(k)];
    [unroll]
    for (int o = 0; o < OUT; o++)
    {
        if (y0 + o >= int(height))
            break;
        float acc = 0.0;
        [unroll]
        for (int k = 0; k <= 2 * R; k++)
            acc += t[k] * v[o + k];
        uint idx = base + uint(y0 + o) * width + x;
        if (mode == 1)
            acc += add_src[idx];
        else if (mode == 2)
            acc += dst[idx];
        dst[idx] = acc;
    }
}

// Host dispatches rows: (ceil(W / OUT), H, planes), cols: (W, ceil(H / OUT), planes).
#define LR_ROWS(NAME, RADIUS)                                               \
    [shader("compute")]                                                     \
    [numthreads(32, 4, 1)]                                                  \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> src,                                \
        uniform RWStructuredBuffer<float> tmp,                              \
        uniform StructuredBuffer<float> tables,                             \
        uniform uint offset,                                                \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint reflect)                                               \
    {                                                                       \
        rows_blocked<RADIUS>(tid, src, tmp, tables, offset, height, width,  \
                             reflect);                                      \
    }

#define LR_COLS(NAME, RADIUS)                                               \
    [shader("compute")]                                                     \
    [numthreads(32, 4, 1)]                                                  \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
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
        cols_blocked<RADIUS>(tid, tmp, add_src, dst, tables, offset, height,\
                             width, reflect, mode);                         \
    }

LR_ROWS(lr_rows_r2, 2)
LR_ROWS(lr_rows_r4, 4)
LR_ROWS(lr_rows_r8, 8)
LR_ROWS(lr_rows_r16, 16)
LR_ROWS(lr_rows_r32, 32)
LR_COLS(lr_cols_r2, 2)
LR_COLS(lr_cols_r4, 4)
LR_COLS(lr_cols_r8, 8)
LR_COLS(lr_cols_r16, 16)
LR_COLS(lr_cols_r32, 32)

// mode 0: dst = acc; 1: dst = add_src + acc; 2: dst += acc
[shader("compute")]
[numthreads(64, 4, 1)]
void lr_cols(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> tmp,
    uniform StructuredBuffer<float> add_src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<float> tables,
    uniform uint offset, // start of this component's column vector
    uniform uint radius,
    uniform uint height,
    uniform uint width,
    uniform uint reflect,
    uniform uint mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= width || y >= height)
        return;
    int r = int(radius);
    uint base = plane * height * width;
    float acc = 0.0;
    for (int ky = 0; ky <= 2 * r; ky++)
        acc += tables[offset + uint(ky)] * tmp[base + uint(pad_index(int(y) + ky - r, int(height), reflect)) * width + x];
    uint idx = base + y * width + x;
    if (mode == 1)
        acc += add_src[idx];
    else if (mode == 2)
        acc += dst[idx];
    dst[idx] = acc;
}
