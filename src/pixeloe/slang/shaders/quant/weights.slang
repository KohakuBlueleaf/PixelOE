// Quantisation weights: bilinear resize (torch semantics) of |2w - 1| * w to
// the output grid, then w ** gamma with torch's pow special cases
// (gamma_mode 0 pow, 1 sqrt (0.5), 2 identity (1), 3 square (2)).

float transform(StructuredBuffer<float> w, uint idx)
{
    float v = w[idx];
    return abs(v * 2.0 - 1.0) * v;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void quant_weights(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> w_full,
    uniform RWStructuredBuffer<float> w_out,
    uniform uint in_h,
    uniform uint in_w,
    uniform uint out_h,
    uniform uint out_w,
    uniform float scale_h,
    uniform float scale_w,
    uniform float gamma,
    uniform uint gamma_mode)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= out_w || y >= out_h)
        return;
    uint base = b * in_h * in_w;
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
    float v = hy * (hx * transform(w_full, base + y0 * in_w + x0) + lx * transform(w_full, base + y0 * in_w + x1))
            + ly * (hx * transform(w_full, base + y1 * in_w + x0) + lx * transform(w_full, base + y1 * in_w + x1));
    if (gamma_mode == 0)
        v = pow(v, gamma);
    else if (gamma_mode == 1)
        v = sqrt(v);
    else if (gamma_mode == 3)
        v = v * v;
    w_out[(b * out_h + y) * out_w + x] = v;
}
