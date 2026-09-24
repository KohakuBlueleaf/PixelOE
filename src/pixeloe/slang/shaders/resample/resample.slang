// Resampling over planar [planes, H, W] images, torch semantics
// (align_corners=False, antialias=False).

// Sparse 1-D resampling pass (Lanczos): dst[o] = sum_k w_k src[idx_k] along
// one axis; row o of the CSR table lists (idx, w) pairs. axis 0 = along H.
[shader("compute")]
[numthreads(16, 16, 1)]
void csr_pass(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<uint> offsets,
    uniform StructuredBuffer<uint> indices,
    uniform StructuredBuffer<float> weights,
    uniform uint axis,
    uniform uint in_h,
    uniform uint in_w,
    uniform uint out_h,
    uniform uint out_w,
    uniform uint do_clamp)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= out_w || y >= out_h)
        return;
    uint src_base = plane * in_h * in_w;
    float acc = 0.0;
    if (axis == 0)
    {
        for (uint k = offsets[y]; k < offsets[y + 1]; k++)
            acc += weights[k] * src[src_base + indices[k] * in_w + x];
    }
    else
    {
        uint row = src_base + y * in_w;
        for (uint k = offsets[x]; k < offsets[x + 1]; k++)
            acc += weights[k] * src[row + indices[k]];
    }
    if (do_clamp != 0)
        acc = clamp(acc, 0.0, 1.0);
    dst[plane * out_h * out_w + y * out_w + x] = acc;
}

float cubic_conv1(float x, float a)
{
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

float cubic_conv2(float x, float a)
{
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

float cubic_interp(float x0, float x1, float x2, float x3, float t)
{
    const float a = -0.75;
    float c0 = cubic_conv2(t + 1.0, a);
    float c1 = cubic_conv1(t, a);
    float u = 1.0 - t;
    float c2 = cubic_conv1(u, a);
    float c3 = cubic_conv2(u + 1.0, a);
    return x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3;
}

float fetch(StructuredBuffer<float> src, uint base, int y, int x, uint in_h, uint in_w)
{
    y = clamp(y, 0, int(in_h) - 1);
    x = clamp(x, 0, int(in_w) - 1);
    return src[base + uint(y) * in_w + uint(x)];
}

// mode 0 nearest, 1 nearest-exact, 2 bilinear, 3 bicubic, 4 area
// scale_h = in_h / out_h, scale_w = in_w / out_w (float32)
[shader("compute")]
[numthreads(16, 16, 1)]
void interpolate(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform uint mode,
    uniform uint in_h,
    uniform uint in_w,
    uniform uint out_h,
    uniform uint out_w,
    uniform float scale_h,
    uniform float scale_w)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= out_w || y >= out_h)
        return;
    uint base = plane * in_h * in_w;
    float v;
    if (mode == 0 || mode == 1)
    {
        float off = mode == 1 ? 0.5 : 0.0;
        uint sy = min(uint(floor((float(y) + off) * scale_h)), in_h - 1);
        uint sx = min(uint(floor((float(x) + off) * scale_w)), in_w - 1);
        v = src[base + sy * in_w + sx];
    }
    else if (mode == 2)
    {
        float ry = max(scale_h * (float(y) + 0.5) - 0.5, 0.0);
        float rx = max(scale_w * (float(x) + 0.5) - 0.5, 0.0);
        uint y0 = uint(ry);
        uint x0 = uint(rx);
        uint y1 = y0 + (y0 < in_h - 1 ? 1 : 0);
        uint x1 = x0 + (x0 < in_w - 1 ? 1 : 0);
        float ly = ry - float(y0);
        float lx = rx - float(x0);
        float hy = 1.0 - ly;
        float hx = 1.0 - lx;
        v = hy * (hx * src[base + y0 * in_w + x0] + lx * src[base + y0 * in_w + x1])
          + ly * (hx * src[base + y1 * in_w + x0] + lx * src[base + y1 * in_w + x1]);
    }
    else if (mode == 3)
    {
        float ry = scale_h * (float(y) + 0.5) - 0.5;
        float rx = scale_w * (float(x) + 0.5) - 0.5;
        float fy = floor(ry);
        float fx = floor(rx);
        float ty = ry - fy;
        float tx = rx - fx;
        int iy = int(fy);
        int ix = int(fx);
        float rows[4];
        for (int k = 0; k < 4; k++)
        {
            int sy = iy - 1 + k;
            rows[k] = cubic_interp(
                fetch(src, base, sy, ix - 1, in_h, in_w),
                fetch(src, base, sy, ix, in_h, in_w),
                fetch(src, base, sy, ix + 1, in_h, in_w),
                fetch(src, base, sy, ix + 2, in_h, in_w),
                tx);
        }
        v = cubic_interp(rows[0], rows[1], rows[2], rows[3], ty);
    }
    else
    {
        uint y0 = (y * in_h) / out_h;
        uint y1 = ((y + 1) * in_h + out_h - 1) / out_h;
        uint x0 = (x * in_w) / out_w;
        uint x1 = ((x + 1) * in_w + out_w - 1) / out_w;
        float sum = 0.0;
        for (uint sy = y0; sy < y1; sy++)
            for (uint sx = x0; sx < x1; sx++)
                sum += src[base + sy * in_w + sx];
        v = sum / float(y1 - y0) / float(x1 - x0);
    }
    dst[plane * out_h * out_w + y * out_w + x] = v;
}

// Replicate padding: dst[y, x] = src[clamp(y - top), clamp(x - left)]
[shader("compute")]
[numthreads(16, 16, 1)]
void pad_replicate(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform uint in_h,
    uniform uint in_w,
    uniform uint out_h,
    uniform uint out_w,
    uniform uint top,
    uniform uint left)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= out_w || y >= out_h)
        return;
    int sy = clamp(int(y) - int(top), 0, int(in_h) - 1);
    int sx = clamp(int(x) - int(left), 0, int(in_w) - 1);
    dst[plane * out_h * out_w + y * out_w + x] = src[plane * in_h * in_w + uint(sy) * in_w + uint(sx)];
}

// F.interpolate(scale_factor=p, mode="nearest-exact"); inv_scale = float(1 / p)
[shader("compute")]
[numthreads(16, 16, 1)]
void upscale_nearest_exact(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform uint in_h,
    uniform uint in_w,
    uniform uint out_h,
    uniform uint out_w,
    uniform float inv_scale)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= out_w || y >= out_h)
        return;
    uint sy = min(uint(floor((float(y) + 0.5) * inv_scale)), in_h - 1);
    uint sx = min(uint(floor((float(x) + 0.5) * inv_scale)), in_w - 1);
    dst[plane * out_h * out_w + y * out_w + x] = src[plane * in_h * in_w + sy * in_w + sx];
}
