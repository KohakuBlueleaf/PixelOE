// Sliding-window local statistics of the luminance map: one window per
// pixel (stride 1) instead of lattice patches + overlap-add. A K x K window
// covers [y - K/2, y - K/2 + K) x [x - K/2, x - K/2 + K) (the unfold window
// of stride 1, padding K/2). Outside the image: pad_mode 0 = zeros (as the
// lattice statistics), 1 = replicate the border pixel.

import order;
import kselect;

float fetch(StructuredBuffer<float> lum, uint base, int y, int x, uint height, uint width, uint pad_mode)
{
    if (pad_mode == 1)
    {
        y = clamp(y, 0, int(height) - 1);
        x = clamp(x, 0, int(width) - 1);
    }
    else if (y < 0 || x < 0 || y >= int(height) || x >= int(width))
    {
        return 0.0;
    }
    return lum[base + uint(y) * width + uint(x)];
}

struct WindowKeys : IKeySource
{
    StructuredBuffer<float> lum;
    uint base;
    int y0;
    int x0;
    uint k;
    uint height;
    uint width;
    uint pad_mode;

    uint count() { return k * k; }

    uint key(uint i)
    {
        uint dy = i / k;
        return float_to_ordered(
            fetch(lum, base, y0 + int(dy), x0 + int(i - dy * k), height, width, pad_mode));
    }
}

// out[b, y, x] = lower median of the K x K window at (y, x)
[shader("compute")]
[numthreads(16, 16, 1)]
void sliding_median(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lum,
    uniform RWStructuredBuffer<float> out_stat,
    uniform uint height,
    uniform uint width,
    uniform uint ksize,
    uniform uint pad_mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    WindowKeys keys;
    keys.lum = lum;
    keys.base = b * height * width;
    keys.y0 = int(y) - int(ksize / 2);
    keys.x0 = int(x) - int(ksize / 2);
    keys.k = ksize;
    keys.height = height;
    keys.width = width;
    keys.pad_mode = pad_mode;
    out_stat[keys.base + y * width + x] = radix_median(keys);
}

// sliding_median for a compile-time window size K: keys in registers
void sliding_median_regs<let K : int>(
    uint3 tid,
    StructuredBuffer<float> lum,
    RWStructuredBuffer<float> out_stat,
    uint height,
    uint width,
    uint pad_mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    uint base = b * height * width;
    int y0 = int(y) - K / 2;
    int x0 = int(x) - K / 2;
    uint keys[K * K];
    [unroll]
    for (int dy = 0; dy < K; dy++)
    {
        [unroll]
        for (int dx = 0; dx < K; dx++)
            keys[dy * K + dx] = float_to_ordered(
                fetch(lum, base, y0 + dy, x0 + dx, height, width, pad_mode));
    }
    out_stat[base + y * width + x] = register_median<K * K>(keys);
}

#define SLIDING_MEDIAN_K(NAME, K)                                           \
    [shader("compute")]                                                     \
    [numthreads(16, 16, 1)]                                                 \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> lum,                                \
        uniform RWStructuredBuffer<float> out_stat,                         \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint pad_mode)                                              \
    {                                                                       \
        sliding_median_regs<K>(tid, lum, out_stat, height, width, pad_mode);\
    }

SLIDING_MEDIAN_K(sliding_median_k4, 4)
SLIDING_MEDIAN_K(sliding_median_k6, 6)
SLIDING_MEDIAN_K(sliding_median_k8, 8)
SLIDING_MEDIAN_K(sliding_median_k10, 10)
SLIDING_MEDIAN_K(sliding_median_k12, 12)

// Separable window min / max. Row pass over columns [x - K/2, x - K/2 + K)
// of the padded image; column pass over rows of the row-pass result, where
// an out-of-image row is a row of padding (zeros, or the replicated edge).
[shader("compute")]
[numthreads(16, 16, 1)]
void sliding_minmax_rows(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lum,
    uniform RWStructuredBuffer<float> row_min,
    uniform RWStructuredBuffer<float> row_max,
    uniform uint height,
    uniform uint width,
    uniform uint ksize,
    uniform uint pad_mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    uint base = b * height * width;
    int x0 = int(x) - int(ksize / 2);
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    for (uint i = 0; i < ksize; i++)
    {
        float v = fetch(lum, base, int(y), x0 + int(i), height, width, pad_mode);
        mn = min(mn, v);
        mx = max(mx, v);
    }
    row_min[base + y * width + x] = mn;
    row_max[base + y * width + x] = mx;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sliding_minmax_cols(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> row_min,
    uniform StructuredBuffer<float> row_max,
    uniform RWStructuredBuffer<float> out_min,
    uniform RWStructuredBuffer<float> out_max,
    uniform uint height,
    uniform uint width,
    uniform uint ksize,
    uniform uint pad_mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    uint base = b * height * width;
    int y0 = int(y) - int(ksize / 2);
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    for (uint i = 0; i < ksize; i++)
    {
        int yy = y0 + int(i);
        if (pad_mode == 1)
            yy = clamp(yy, 0, int(height) - 1);
        if (yy < 0 || yy >= int(height))
        {
            mn = min(mn, 0.0);
            mx = max(mx, 0.0);
        }
        else
        {
            mn = min(mn, row_min[base + uint(yy) * width + x]);
            mx = max(mx, row_max[base + uint(yy) * width + x]);
        }
    }
    out_min[base + y * width + x] = mn;
    out_max[base + y * width + x] = mx;
}
