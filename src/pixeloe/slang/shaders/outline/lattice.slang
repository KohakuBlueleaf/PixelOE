// Lattice local statistics of the luminance map (torch F.unfold semantics):
// patch (i, j) of size K x K has origin (i * S - pad, j * S - pad), zero
// padding outside the image, lattice size n = (H + 2 pad - K) / S + 1.

import order;
import kselect;

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
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    uint o = (b * lat_h + i) * lat_w + j;
    out_min[o] = mn;
    out_max[o] = mx;
}
