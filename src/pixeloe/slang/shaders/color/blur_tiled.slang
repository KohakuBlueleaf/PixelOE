// GPU-only (groupshared + barrier) variant of blur.slang's blur2d:
// dst = add_src (when has_add) + conv(pad(src), weights)
//
// The reference kernel is symmetric: w[dy][dx] = w[-dy][dx] = w[dy][-dx]
// (its taps come from x^2). So
//   out(y, x) = sum_{dx=0..R} sum_{dy=0..R} w[dy][dx] * S(dy, dx),
//   S = sum of the 1, 2 or 4 samples at (y +- dy, x +- dx),
// i.e. (R+1)^2 multiplies instead of (2R+1)^2. Each thread owns ROWS
// vertically adjacent outputs of one column; per column offset dx it sums
// the two mirrored tile columns once into registers (2R+ROWS values) and
// reuses them, with the R+1 weights of that column, for all ROWS outputs.
// R, ROWS and the thread rows TY are compile-time (one entry point per
// wavelet level), so the register arrays are fully unrolled.
//
// Tile: 32 x (ROWS * TY) outputs, 32 x TY threads, groupshared
// (32 + 2R) x (ROWS * TY + 2R) floats (<= 30 KiB, under the D3D12 32 KiB).

import blur_pad;

static const uint TILE_W = 32;
groupshared float g_tile[7680];

void blur_sym<let R : int, let ROWS : int, let TY : int>(
    uint3 gid,
    uint3 lid,
    StructuredBuffer<float> src,
    StructuredBuffer<float> add_src,
    RWStructuredBuffer<float> dst,
    StructuredBuffer<float> weights, // (2R+1)^2 row-major
    uint height,
    uint width,
    uint reflect,
    uint has_add)
{
    const int K = 2 * R + 1;
    const uint TILE_H = uint(ROWS * TY);
    const uint TW = TILE_W + 2 * R;
    const uint TH = TILE_H + 2 * R;
    int x0 = int(gid.x * TILE_W) - R;
    int y0 = int(gid.y * TILE_H) - R;
    uint base = gid.z * height * width;
    for (uint e = lid.y * TILE_W + lid.x; e < TW * TH; e += TILE_W * uint(TY))
    {
        uint ty = e / TW;
        uint tx = e - ty * TW;
        int sy = pad_index(y0 + int(ty), int(height), reflect);
        int sx = pad_index(x0 + int(tx), int(width), reflect);
        g_tile[ty * TW + tx] = src[base + uint(sy) * width + uint(sx)];
    }
    GroupMemoryBarrierWithGroupSync();

    uint row0 = lid.y * uint(ROWS); // tile row of this thread's first output - R
    uint cx = lid.x + R;            // tile column of this thread's outputs
    float acc[ROWS];
    [unroll]
    for (int o = 0; o < ROWS; o++)
        acc[o] = 0.0;
    for (int dx = 0; dx <= R; dx++)
    {
        float s[2 * R + ROWS];
        [unroll]
        for (int k = 0; k < 2 * R + ROWS; k++)
        {
            uint row = (row0 + uint(k)) * TW;
            float v = g_tile[row + cx + uint(dx)];
            if (dx > 0)
                v += g_tile[row + cx - uint(dx)];
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

    uint x = gid.x * TILE_W + lid.x;
    if (x >= width)
        return;
    [unroll]
    for (int o = 0; o < ROWS; o++)
    {
        uint y = gid.y * TILE_H + row0 + uint(o);
        if (y < height)
        {
            uint idx = base + y * width + x;
            dst[idx] = has_add != 0 ? add_src[idx] + acc[o] : acc[o];
        }
    }
}

// One entry point per wavelet level radius; the host dispatches
// (ceil(W / 32) * 32, ceil(H / (ROWS * TY)) * TY, planes) threads.
#define BLUR_ENTRY(NAME, RADIUS, ROWS, TY)                                \
    [shader("compute")]                                                   \
    [numthreads(32, TY, 1)]                                               \
    void NAME(                                                            \
        uint3 gid: SV_GroupID,                                            \
        uint3 lid: SV_GroupThreadID,                                      \
        uniform StructuredBuffer<float> src,                              \
        uniform StructuredBuffer<float> add_src,                          \
        uniform RWStructuredBuffer<float> dst,                            \
        uniform StructuredBuffer<float> weights,                          \
        uniform uint height,                                              \
        uniform uint width,                                               \
        uniform uint reflect,                                             \
        uniform uint has_add)                                             \
    {                                                                     \
        blur_sym<RADIUS, ROWS, TY>(gid, lid, src, add_src, dst, weights,  \
                                   height, width, reflect, has_add);      \
    }

BLUR_ENTRY(blur2d_tiled_r2, 2, 4, 4)
BLUR_ENTRY(blur2d_tiled_r4, 4, 4, 4)
BLUR_ENTRY(blur2d_tiled_r8, 8, 4, 4)
BLUR_ENTRY(blur2d_tiled_r16, 16, 8, 4)
BLUR_ENTRY(blur2d_tiled_r32, 32, 8, 2)
