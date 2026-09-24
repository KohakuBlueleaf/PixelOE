// Lattice local statistics of the luminance map (torch F.unfold semantics):
// patch (i, j) of size K x K has origin (i * S - pad, j * S - pad), zero
// padding outside the image, lattice size n = (H + 2 pad - K) / S + 1.

import order;
import kselect;
import sel;

struct PatchKeys : IKeySource
{
    StructuredBuffer<float> lum;
    uint base;
    int y0;
    int x0;
    uint k;
    uint height;
    uint width;

    uint count() { return k * k; }

    uint key(uint i)
    {
        uint dy = i / k;
        int y = y0 + int(dy);
        int x = x0 + int(i - dy * k);
        float v = 0.0;
        if (y >= 0 && x >= 0 && y < int(height) && x < int(width))
            v = lum[base + uint(y) * width + uint(x)];
        return float_to_ordered(v);
    }
}

// out[b, i, j] = lower median of patch (i, j)
[shader("compute")]
[numthreads(16, 16, 1)]
void lattice_median(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lum,
    uniform RWStructuredBuffer<float> out_stat,
    uniform uint height,
    uniform uint width,
    uniform uint ksize,
    uniform uint stride,
    uniform uint pad,
    uniform uint lat_h,
    uniform uint lat_w)
{
    uint j = tid.x;
    uint i = tid.y;
    uint b = tid.z;
    if (i >= lat_h || j >= lat_w)
        return;
    PatchKeys keys;
    keys.lum = lum;
    keys.base = b * height * width;
    keys.y0 = int(i * stride) - int(pad);
    keys.x0 = int(j * stride) - int(pad);
    keys.k = ksize;
    keys.height = height;
    keys.width = width;
    out_stat[(b * lat_h + i) * lat_w + j] = radix_median(keys);
}

// lattice_median for a compile-time patch size K: the K^2 keys are loaded
// once into registers and selected there (kselect.register_median).
void lattice_median_regs<let K : int>(
    uint3 tid,
    StructuredBuffer<float> lum,
    RWStructuredBuffer<float> out_stat,
    uint height,
    uint width,
    uint stride,
    uint pad,
    uint lat_h,
    uint lat_w)
{
    uint j = tid.x;
    uint i = tid.y;
    uint b = tid.z;
    if (i >= lat_h || j >= lat_w)
        return;
    int y0 = int(i * stride) - int(pad);
    int x0 = int(j * stride) - int(pad);
    uint base = b * height * width;
    uint keys[K * K];
    [unroll]
    for (int dy = 0; dy < K; dy++)
    {
        int y = y0 + dy;
        bool row_in = y >= 0 && y < int(height);
        [unroll]
        for (int dx = 0; dx < K; dx++)
        {
            int x = x0 + dx;
            float v = 0.0;
            if (row_in && x >= 0 && x < int(width))
                v = lum[base + uint(y) * width + uint(x)];
            keys[dy * K + dx] = float_to_ordered(v);
        }
    }
    out_stat[(b * lat_h + i) * lat_w + j] = register_median<K * K>(keys);
}

#define LATTICE_MEDIAN_K(NAME, K)                                           \
    [shader("compute")]                                                     \
    [numthreads(16, 16, 1)]                                                 \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> lum,                                \
        uniform RWStructuredBuffer<float> out_stat,                         \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint stride,                                                \
        uniform uint pad,                                                   \
        uniform uint lat_h,                                                 \
        uniform uint lat_w)                                                 \
    {                                                                       \
        lattice_median_regs<K>(tid, lum, out_stat, height, width, stride,   \
                               pad, lat_h, lat_w);                          \
    }

LATTICE_MEDIAN_K(lattice_median_k4, 4)
LATTICE_MEDIAN_K(lattice_median_k6, 6)
LATTICE_MEDIAN_K(lattice_median_k8, 8)
LATTICE_MEDIAN_K(lattice_median_k10, 10)
LATTICE_MEDIAN_K(lattice_median_k12, 12)

// All three lattice statistics of the outline weight in one pass, for an
// even half-size H (patch 2H, compile-time):
//   median over the 2H x 2H patch at origin (i S - H)
//   min / max over the H x H patch at origin (i S - H / 2), the centre of
//   the median patch; both lattices have the same (lat_h, lat_w).
void lattice_stats<let H : int>(
    uint3 tid,
    StructuredBuffer<float> lum,
    RWStructuredBuffer<float> out_med,
    RWStructuredBuffer<float> out_min,
    RWStructuredBuffer<float> out_max,
    uint height,
    uint width,
    uint stride,
    uint lat_h,
    uint lat_w)
{
    const int K = 2 * H;
    uint j = tid.x;
    uint i = tid.y;
    uint b = tid.z;
    if (i >= lat_h || j >= lat_w)
        return;
    int y0 = int(i * stride) - H;
    int x0 = int(j * stride) - H;
    uint base = b * height * width;
    uint keys[K * K];
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    [unroll]
    for (int dy = 0; dy < K; dy++)
    {
        int y = y0 + dy;
        bool row_in = y >= 0 && y < int(height);
        [unroll]
        for (int dx = 0; dx < K; dx++)
        {
            int x = x0 + dx;
            float v = 0.0;
            if (row_in && x >= 0 && x < int(width))
                v = lum[base + uint(y) * width + uint(x)];
            keys[dy * K + dx] = float_to_ordered(v);
            if (dy >= H / 2 && dy < H / 2 + H && dx >= H / 2 && dx < H / 2 + H)
            {
                mn = fmin_s(mn, v);
                mx = fmax_s(mx, v);
            }
        }
    }
    uint o = (b * lat_h + i) * lat_w + j;
    out_med[o] = register_median<K * K>(keys);
    out_min[o] = mn;
    out_max[o] = mx;
}

#define LATTICE_STATS(NAME, H)                                              \
    [shader("compute")]                                                     \
    [numthreads(16, 16, 1)]                                                 \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> lum,                                \
        uniform RWStructuredBuffer<float> out_med,                          \
        uniform RWStructuredBuffer<float> out_min,                          \
        uniform RWStructuredBuffer<float> out_max,                          \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint stride,                                                \
        uniform uint lat_h,                                                 \
        uniform uint lat_w)                                                 \
    {                                                                       \
        lattice_stats<H>(tid, lum, out_med, out_min, out_max, height,       \
                         width, stride, lat_h, lat_w);                      \
    }

LATTICE_STATS(lattice_stats_h2, 2)
LATTICE_STATS(lattice_stats_h4, 4)
LATTICE_STATS(lattice_stats_h6, 6)

// out_min / out_max[b, i, j] over patch (i, j), zero padding included
[shader("compute")]
[numthreads(16, 16, 1)]
void lattice_minmax(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lum,
    uniform RWStructuredBuffer<float> out_min,
    uniform RWStructuredBuffer<float> out_max,
    uniform uint height,
    uniform uint width,
    uniform uint ksize,
    uniform uint stride,
    uniform uint pad,
    uniform uint lat_h,
    uniform uint lat_w)
{
    uint j = tid.x;
    uint i = tid.y;
    uint b = tid.z;
    if (i >= lat_h || j >= lat_w)
        return;
    int y0 = int(i * stride) - int(pad);
    int x0 = int(j * stride) - int(pad);
    uint base = b * height * width;
    float mn = asfloat(0x7f800000u);
    float mx = -asfloat(0x7f800000u);
    for (uint dy = 0; dy < ksize; dy++)
    {
        int y = y0 + int(dy);
        bool row_in = y >= 0 && y < int(height);
        for (uint dx = 0; dx < ksize; dx++)
        {
            int x = x0 + int(dx);
            float v = 0.0;
            if (row_in && x >= 0 && x < int(width))
                v = lum[base + uint(y) * width + uint(x)];
            mn = fmin_s(mn, v);
            mx = fmax_s(mx, v);
        }
    }
    uint o = (b * lat_h + i) * lat_w + j;
    out_min[o] = mn;
    out_max[o] = mx;
}
