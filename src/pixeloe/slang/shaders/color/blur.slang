// Wavelet colour-fix blur levels over planar [planes, H, W] images,
// portable implementations (every target, CPU included):
//
// dst = add_src (when has_add) + conv(pad(src), weights)
//
//   blur2d          2-D direct convolution with the (2r+1)^2 table
//   blur2d_sym_rN   the same, folding the kernel's symmetry (radius N)
//   blur_rows/cols  separable passes with a 1-D table
//
// blur_tiled.slang holds the GPU groupshared variant of blur2d.

import blur_pad;

[shader("compute")]
[numthreads(16, 16, 1)]
void blur2d(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform StructuredBuffer<float> add_src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<float> weights, // (2r+1)^2 row-major
    uniform uint radius,
    uniform uint height,
    uniform uint width,
    uniform uint reflect,
    uniform uint has_add)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= width || y >= height)
        return;
    int r = int(radius);
    int k = 2 * r + 1;
    uint base = plane * height * width;
    float acc = 0.0;
    for (int ky = 0; ky < k; ky++)
    {
        uint row = base + uint(pad_index(int(y) + ky - r, int(height), reflect)) * width;
        for (int kx = 0; kx < k; kx++)
        {
            int ix = pad_index(int(x) + kx - r, int(width), reflect);
            acc += weights[ky * k + kx] * src[row + uint(ix)];
        }
    }
    uint idx = base + y * width + x;
    dst[idx] = has_add != 0 ? add_src[idx] + acc : acc;
}

// Symmetric-fold direct convolution (same folding as blur_tiled.slang, read
// straight from memory): the reference kernel satisfies
// w[dy][dx] = w[-dy][dx] = w[dy][-dx], so each thread owns ROWS vertically
// adjacent outputs and, per column offset dx, sums the mirrored columns once
// into registers and applies the R+1 weights of that column to all ROWS
// outputs: (R+1)^2 multiplies per output instead of (2R+1)^2.
void blur_sym_direct<let R : int, let ROWS : int>(
    uint3 tid,
    StructuredBuffer<float> src,
    StructuredBuffer<float> add_src,
    RWStructuredBuffer<float> dst,
    StructuredBuffer<float> weights,
    uint height,
    uint width,
    uint reflect,
    uint has_add)
{
    const int K = 2 * R + 1;
    uint x = tid.x;
    int y0 = int(tid.y) * ROWS;
    uint plane = tid.z;
    if (x >= width || y0 >= int(height))
        return;
    uint base = plane * height * width;
    uint rows[2 * R + ROWS];
    [unroll]
    for (int k = 0; k < 2 * R + ROWS; k++)
        rows[k] = base + uint(pad_index(y0 - R + k, int(height), reflect)) * width;
    float acc[ROWS];
    [unroll]
    for (int o = 0; o < ROWS; o++)
        acc[o] = 0.0;
    for (int dx = 0; dx <= R; dx++)
    {
        uint xp = uint(pad_index(int(x) + dx, int(width), reflect));
        uint xm = uint(pad_index(int(x) - dx, int(width), reflect));
        float s[2 * R + ROWS];
        [unroll]
        for (int k = 0; k < 2 * R + ROWS; k++)
        {
            float v = src[rows[k] + xp];
            if (dx > 0)
                v += src[rows[k] + xm];
            s[k] = v;
        }
        float w[R + 1];
        [unroll]
        for (int dy = 0; dy <= R; dy++)
            w[dy] = weights[(R + dy) * K + R + dx];
        [unroll]
        for (int o = 0; o < ROWS; o++)
        {
            float a = w[0] * s[R + o];
            [unroll]
            for (int dy = 1; dy <= R; dy++)
                a += w[dy] * (s[R + o + dy] + s[R + o - dy]);
            acc[o] += a;
        }
    }
    [unroll]
    for (int o = 0; o < ROWS; o++)
    {
        int y = y0 + o;
        if (y < int(height))
        {
            uint idx = base + uint(y) * width + x;
            dst[idx] = has_add != 0 ? add_src[idx] + acc[o] : acc[o];
        }
    }
}

// Host dispatches (W, ceil(H / 4), planes) threads.
#define BLUR_SYM_ENTRY(NAME, RADIUS)                                      \
    [shader("compute")]                                                   \
    [numthreads(64, 1, 1)]                                                \
    void NAME(                                                            \
        uint3 tid: SV_DispatchThreadID,                                   \
        uniform StructuredBuffer<float> src,                              \
        uniform StructuredBuffer<float> add_src,                          \
        uniform RWStructuredBuffer<float> dst,                            \
        uniform StructuredBuffer<float> weights,                          \
        uniform uint height,                                              \
        uniform uint width,                                               \
        uniform uint reflect,                                             \
        uniform uint has_add)                                             \
    {                                                                     \
        blur_sym_direct<RADIUS, 4>(tid, src, add_src, dst, weights,       \
                                   height, width, reflect, has_add);      \
    }

BLUR_SYM_ENTRY(blur2d_sym_r2, 2)
BLUR_SYM_ENTRY(blur2d_sym_r4, 4)
BLUR_SYM_ENTRY(blur2d_sym_r8, 8)
BLUR_SYM_ENTRY(blur2d_sym_r16, 16)
BLUR_SYM_ENTRY(blur2d_sym_r32, 32)

// Separable passes: rows (along W) then cols (along H, with the add).
[shader("compute")]
[numthreads(16, 16, 1)]
void blur_rows(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<float> weights, // 2r+1
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
    for (int kx = 0; kx < 2 * r + 1; kx++)
        acc += weights[kx] * src[row + uint(pad_index(int(x) + kx - r, int(width), reflect))];
    dst[row + x] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blur_cols(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform StructuredBuffer<float> add_src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<float> weights,
    uniform uint radius,
    uniform uint height,
    uniform uint width,
    uniform uint reflect,
    uniform uint has_add)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= width || y >= height)
        return;
    int r = int(radius);
    uint base = plane * height * width;
    float acc = 0.0;
    for (int ky = 0; ky < 2 * r + 1; ky++)
        acc += weights[ky] * src[base + uint(pad_index(int(y) + ky - r, int(height), reflect)) * width + x];
    uint idx = base + y * width + x;
    dst[idx] = has_add != 0 ? add_src[idx] + acc : acc;
}
