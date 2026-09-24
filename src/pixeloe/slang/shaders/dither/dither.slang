// Palette dithering over planar pixels img[b, c, n], palette pal[b, k, c]
// (portable kernels; ed_group.slang holds the GPU single-dispatch variant).
//
// ordered: two nearest colours d1 <= d2 (ties: lowest index first),
//   r = d1 / (d1 + d2 + 1e-6); bayer8[y % 8][x % 8] > r -> nearest, else second.
// error diffusion (reference row-pair scheme), per step y (y + 2 < h):
//   e = x - nearest(x) for rows y, y+1, y+2 from the current state, then
//   x[y+1] += K0 * e[y]   + K1 * e[y+1]
//   x[y+2] += K0 * e[y+1] + K1 * e[y+2]
//   K0 = [0, 0, 7] / 16, K1 = [3, 5, 1] / 16 over columns x-1..x+1
//   (reflect at the borders); finally every pixel -> nearest colour.

import palette;

[shader("compute")]
[numthreads(16, 16, 1)]
void dither_ordered(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> pal,
    uniform StructuredBuffer<float> bayer, // 8 x 8, already / 64
    uniform RWStructuredBuffer<float> dst,
    uniform uint K,
    uniform uint height,
    uniform uint width)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    uint plane = height * width;
    uint base = b * 3 * plane + y * width + x;
    float3 v = load_px(img, base, plane);
    uint i1 = 0;
    uint i2 = 0;
    float d1 = asfloat(0x7f800000u);
    float d2 = asfloat(0x7f800000u);
    for (uint k = 0; k < K; k++)
    {
        float d = dist2(v, load_pal(pal, b, k, K));
        if (d < d1)
        {
            d2 = d1;
            i2 = i1;
            d1 = d;
            i1 = k;
        }
        else if (d < d2)
        {
            d2 = d;
            i2 = k;
        }
    }
    float ratio = d1 / (d1 + d2 + 1e-6);
    float3 c = bayer[(y % 8) * 8 + (x % 8)] > ratio ? load_pal(pal, b, i1, K) : load_pal(pal, b, i2, K);
    dst[base] = c.x;
    dst[base + plane] = c.y;
    dst[base + 2 * plane] = c.z;
}

// err[b, r, c, x] for rows y + r, r = 0..2
[shader("compute")]
[numthreads(64, 3, 1)]
void ed_errors(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> pal,
    uniform RWStructuredBuffer<float> err,
    uniform uint y,
    uniform uint K,
    uniform uint height,
    uniform uint width)
{
    uint x = tid.x;
    uint r = tid.y;
    uint b = tid.z;
    if (x >= width || r >= 3)
        return;
    uint plane = height * width;
    uint base = b * 3 * plane + (y + r) * width + x;
    float3 v = load_px(img, base, plane);
    float3 e = v - load_pal(pal, b, nearest(v, pal, b, K), K);
    uint o = ((b * 3 + r) * 3) * width + x;
    err[o] = e.x;
    err[o + width] = e.y;
    err[o + 2 * width] = e.z;
}

[shader("compute")]
[numthreads(64, 2, 1)]
void ed_apply(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> err,
    uniform uint y,
    uniform uint height,
    uniform uint width)
{
    uint x = tid.x;
    uint r = tid.y; // target row y + 1 + r uses error rows r (K0) and r + 1 (K1)
    uint b = tid.z;
    if (x >= width || r >= 2)
        return;
    uint plane = height * width;
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

[shader("compute")]
[numthreads(16, 16, 1)]
void palette_map(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> pal,
    uniform RWStructuredBuffer<float> dst,
    uniform uint K,
    uniform uint height,
    uniform uint width)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    uint plane = height * width;
    uint base = b * 3 * plane + y * width + x;
    float3 c = load_pal(pal, b, nearest(load_px(img, base, plane), pal, b, K), K);
    dst[base] = c.x;
    dst[base + plane] = c.y;
    dst[base + 2 * plane] = c.z;
}
