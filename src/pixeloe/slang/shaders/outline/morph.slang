// Grey-scale morphology with a non-flat structuring element k (values in
// [0, 1], size ks x ks) over a zero-padded neighbourhood (torch F.unfold):
//     dilate(f)(x) = max_y (f(x + y) + k(y)) - 1
//     erode(f)(x)  = min_y (f(x + y) - k(y)) + 1
// dilate_cont clamps to [0, 1] after its iterations, erode_cont never clamps.

float morph_at(
    StructuredBuffer<float> src,
    StructuredBuffer<float> se,
    uint ks,
    uint base,
    int y,
    int x,
    uint height,
    uint width,
    bool dilate)
{
    int r = int(ks / 2);
    float acc = dilate ? -asfloat(0x7f800000u) : asfloat(0x7f800000u);
    for (uint dy = 0; dy < ks; dy++)
    {
        int sy = y + int(dy) - r;
        bool row_in = sy >= 0 && sy < int(height);
        for (uint dx = 0; dx < ks; dx++)
        {
            int sx = x + int(dx) - r;
            float v = 0.0;
            if (row_in && sx >= 0 && sx < int(width))
                v = src[base + uint(sy) * width + uint(sx)];
            float k = se[dy * ks + dx];
            if (dilate)
                acc = max(acc, v + k - 1.0);
            else
                acc = min(acc, v - k + 1.0);
        }
    }
    return acc;
}

// mode 0 erode, 1 dilate; do_clamp clamps the result to [0, 1]
[shader("compute")]
[numthreads(16, 16, 1)]
void morph(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<float> se,
    uniform uint ks,
    uniform uint height,
    uniform uint width,
    uniform uint mode,
    uniform uint do_clamp)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= width || y >= height)
        return;
    uint base = plane * height * width;
    float v = morph_at(src, se, ks, base, int(y), int(x), height, width, mode == 1);
    if (do_clamp != 0)
        v = clamp(v, 0.0, 1.0);
    dst[base + y * width + x] = v;
}

// Weight normalisation: norm_mode 0 none, 1 global (slot 0), 2 per image
// (slot b); (w - min) / (max - min + 1e-8).
float normalize_weight(float w, StructuredBuffer<float> wmin, StructuredBuffer<float> wmax, uint norm_mode, uint b)
{
    if (norm_mode == 0)
        return w;
    uint s = norm_mode == 1 ? 0 : b;
    float mn = wmin[s];
    return (w - mn) / (wmax[s] - mn + 1e-8);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void normalize_map(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> w_raw,
    uniform StructuredBuffer<float> wmin,
    uniform StructuredBuffer<float> wmax,
    uniform RWStructuredBuffer<float> w_out,
    uniform uint hw,
    uniform uint count,
    uniform uint norm_mode,
    uniform uint row_len)
{
    uint q = tid.y * row_len + tid.x;
    if (q >= count)
        return;
    w_out[q] = normalize_weight(w_raw[q], wmin, wmax, norm_mode, q / hw);
}

// out = erode(img) * w + clamp(dilate(img)) * (1 - w), w normalised.
// Channel 0 also writes the normalised weight map.
[shader("compute")]
[numthreads(16, 16, 1)]
void oe_blend(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform StructuredBuffer<float> w_raw,
    uniform StructuredBuffer<float> wmin,
    uniform StructuredBuffer<float> wmax,
    uniform StructuredBuffer<float> se_erode,
    uniform StructuredBuffer<float> se_dilate,
    uniform RWStructuredBuffer<float> dst,
    uniform RWStructuredBuffer<float> w_out,
    uniform uint ks_erode,
    uniform uint ks_dilate,
    uniform uint height,
    uniform uint width,
    uniform uint norm_mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z; // b * 3 + c
    if (x >= width || y >= height)
        return;
    uint b = plane / 3;
    uint hw = height * width;
    uint base = plane * hw;
    uint pix = y * width + x;
    float w = normalize_weight(w_raw[b * hw + pix], wmin, wmax, norm_mode, b);
    float e = morph_at(img, se_erode, ks_erode, base, int(y), int(x), height, width, false);
    float d = clamp(morph_at(img, se_dilate, ks_dilate, base, int(y), int(x), height, width, true), 0.0, 1.0);
    dst[base + pix] = e * w + d * (1.0 - w);
    if (plane - b * 3 == 0)
        w_out[b * hw + pix] = w;
}
