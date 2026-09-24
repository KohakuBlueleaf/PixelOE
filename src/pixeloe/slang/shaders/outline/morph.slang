// Grey-scale morphology with a non-flat structuring element k (values in
// [0, 1], size ks x ks) over a zero-padded neighbourhood (torch F.unfold):
//     dilate(f)(x) = max_y (f(x + y) + k(y)) - 1
//     erode(f)(x)  = min_y (f(x + y) - k(y)) + 1
// dilate_cont clamps to [0, 1] after its iterations, erode_cont never clamps.

import sel;

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
                acc = fmax_s(acc, v + k - 1.0);
            else
                acc = fmin_s(acc, v - k + 1.0);
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
        v = fclamp_s(v, 0.0, 1.0);
    dst[base + y * width + x] = v;
}

// Compile-time element size KS: taps unrolled, element in registers.
float morph_fixed<let KS : int>(
    StructuredBuffer<float> src,
    float se[KS * KS],
    uint base,
    int y,
    int x,
    uint height,
    uint width,
    bool dilate)
{
    const int R = KS / 2;
    float acc = dilate ? -asfloat(0x7f800000u) : asfloat(0x7f800000u);
    [unroll]
    for (int dy = 0; dy < KS; dy++)
    {
        int sy = y + dy - R;
        bool row_in = sy >= 0 && sy < int(height);
        [unroll]
        for (int dx = 0; dx < KS; dx++)
        {
            int sx = x + dx - R;
            float v = 0.0;
            if (row_in && sx >= 0 && sx < int(width))
                v = src[base + uint(sy) * width + uint(sx)];
            if (dilate)
                acc = fmax_s(acc, v + se[dy * KS + dx] - 1.0);
            else
                acc = fmin_s(acc, v - se[dy * KS + dx] + 1.0);
        }
    }
    return acc;
}

// A thread owns OUT consecutive outputs of a row: the (OUT + 2R) x KS input
// window is loaded once (zero outside the image) and every output's taps
// read it from registers.
static const int MORPH_OUT = 8;

void morph_k<let KS : int>(
    uint3 tid,
    StructuredBuffer<float> src,
    RWStructuredBuffer<float> dst,
    StructuredBuffer<float> se,
    uint height,
    uint width,
    uint mode,
    uint do_clamp)
{
    const int R = KS / 2;
    const int NW = MORPH_OUT + 2 * R;
    int x0 = int(tid.x) * MORPH_OUT;
    int y = int(tid.y);
    if (x0 >= int(width) || y >= int(height))
        return;
    float k[KS * KS];
    [unroll]
    for (int i = 0; i < KS * KS; i++)
        k[i] = se[i];
    uint base = tid.z * height * width;
    float win[KS * NW];
    [unroll]
    for (int dy = 0; dy < KS; dy++)
    {
        int sy = y + dy - R;
        bool row_in = sy >= 0 && sy < int(height);
        [unroll]
        for (int i = 0; i < NW; i++)
        {
            int sx = x0 - R + i;
            float v = 0.0;
            if (row_in && sx >= 0 && sx < int(width))
                v = src[base + uint(sy) * width + uint(sx)];
            win[dy * NW + i] = v;
        }
    }
    bool dilate = mode == 1;
    [unroll]
    for (int o = 0; o < MORPH_OUT; o++)
    {
        if (x0 + o >= int(width))
            break;
        float acc = dilate ? -asfloat(0x7f800000u) : asfloat(0x7f800000u);
        [unroll]
        for (int dy = 0; dy < KS; dy++)
        {
            [unroll]
            for (int dx = 0; dx < KS; dx++)
            {
                float v = win[dy * NW + o + dx];
                if (dilate)
                    acc = fmax_s(acc, v + k[dy * KS + dx] - 1.0);
                else
                    acc = fmin_s(acc, v - k[dy * KS + dx] + 1.0);
            }
        }
        if (do_clamp != 0)
            acc = fclamp_s(acc, 0.0, 1.0);
        dst[base + uint(y) * width + uint(x0 + o)] = acc;
    }
}

// Host dispatches (ceil(W / MORPH_OUT), H, planes) threads.
#define MORPH_K(NAME, KS)                                                   \
    [shader("compute")]                                                     \
    [numthreads(16, 16, 1)]                                                 \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> src,                                \
        uniform RWStructuredBuffer<float> dst,                              \
        uniform StructuredBuffer<float> se,                                 \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint mode,                                                  \
        uniform uint do_clamp)                                              \
    {                                                                       \
        morph_k<KS>(tid, src, dst, se, height, width, mode, do_clamp);      \
    }

MORPH_K(morph_k3, 3)
MORPH_K(morph_k5, 5)
MORPH_K(morph_k7, 7)

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
    float d = fclamp_s(morph_at(img, se_dilate, ks_dilate, base, int(y), int(x), height, width, true), 0.0, 1.0);
    dst[base + pix] = e * w + d * (1.0 - w);
    if (plane - b * 3 == 0)
        w_out[b * hw + pix] = w;
}

// oe_blend with equal compile-time erode / dilate element sizes KS
void oe_blend_k<let KS : int>(
    uint3 tid,
    StructuredBuffer<float> img,
    StructuredBuffer<float> w_raw,
    StructuredBuffer<float> wmin,
    StructuredBuffer<float> wmax,
    StructuredBuffer<float> se_erode,
    StructuredBuffer<float> se_dilate,
    RWStructuredBuffer<float> dst,
    RWStructuredBuffer<float> w_out,
    uint height,
    uint width,
    uint norm_mode)
{
    // MORPH_OUT consecutive outputs per thread from one register window
    const int R = KS / 2;
    const int NW = MORPH_OUT + 2 * R;
    int x0 = int(tid.x) * MORPH_OUT;
    int y = int(tid.y);
    uint plane = tid.z;
    if (x0 >= int(width) || y >= int(height))
        return;
    float ke[KS * KS];
    float kd[KS * KS];
    [unroll]
    for (int i = 0; i < KS * KS; i++)
    {
        ke[i] = se_erode[i];
        kd[i] = se_dilate[i];
    }
    uint b = plane / 3;
    uint hw = height * width;
    uint base = plane * hw;
    float win[KS * NW];
    [unroll]
    for (int dy = 0; dy < KS; dy++)
    {
        int sy = y + dy - R;
        bool row_in = sy >= 0 && sy < int(height);
        [unroll]
        for (int i = 0; i < NW; i++)
        {
            int sx = x0 - R + i;
            float v = 0.0;
            if (row_in && sx >= 0 && sx < int(width))
                v = img[base + uint(sy) * width + uint(sx)];
            win[dy * NW + i] = v;
        }
    }
    [unroll]
    for (int o = 0; o < MORPH_OUT; o++)
    {
        if (x0 + o >= int(width))
            break;
        float e = asfloat(0x7f800000u);
        float d = -asfloat(0x7f800000u);
        [unroll]
        for (int dy = 0; dy < KS; dy++)
        {
            [unroll]
            for (int dx = 0; dx < KS; dx++)
            {
                float v = win[dy * NW + o + dx];
                e = fmin_s(e, v - ke[dy * KS + dx] + 1.0);
                d = fmax_s(d, v + kd[dy * KS + dx] - 1.0);
            }
        }
        d = fclamp_s(d, 0.0, 1.0);
        uint pix = uint(y) * width + uint(x0 + o);
        float w = normalize_weight(w_raw[b * hw + pix], wmin, wmax, norm_mode, b);
        dst[base + pix] = e * w + d * (1.0 - w);
        if (plane - b * 3 == 0)
            w_out[b * hw + pix] = w;
    }
}

#define OE_BLEND_K(NAME, KS)                                                \
    [shader("compute")]                                                     \
    [numthreads(16, 16, 1)]                                                 \
    void NAME(                                                              \
        uint3 tid: SV_DispatchThreadID,                                     \
        uniform StructuredBuffer<float> img,                                \
        uniform StructuredBuffer<float> w_raw,                              \
        uniform StructuredBuffer<float> wmin,                               \
        uniform StructuredBuffer<float> wmax,                               \
        uniform StructuredBuffer<float> se_erode,                           \
        uniform StructuredBuffer<float> se_dilate,                          \
        uniform RWStructuredBuffer<float> dst,                              \
        uniform RWStructuredBuffer<float> w_out,                            \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint norm_mode)                                             \
    {                                                                       \
        oe_blend_k<KS>(tid, img, w_raw, wmin, wmax, se_erode, se_dilate,    \
                       dst, w_out, height, width, norm_mode);               \
    }

OE_BLEND_K(oe_blend_k3, 3)
OE_BLEND_K(oe_blend_k5, 5)
OE_BLEND_K(oe_blend_k7, 7)
