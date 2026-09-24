// 3x3 sharpening filters, zero padding (F.conv2d padding=1), clamp [0, 1].
//   unsharp:   mask = x - gauss3(x); |mask| < threshold -> 0; x + amount mask
//   laplacian: x + amount * conv([[0,-1,0],[-1,4,-1],[0,-1,0]])

float conv3(StructuredBuffer<float> src, StructuredBuffer<float> k, uint base, int y, int x, uint height, uint width)
{
    float acc = 0.0;
    for (int dy = 0; dy < 3; dy++)
    {
        int sy = y + dy - 1;
        for (int dx = 0; dx < 3; dx++)
        {
            int sx = x + dx - 1;
            float v = 0.0;
            if (sy >= 0 && sx >= 0 && sy < int(height) && sx < int(width))
                v = src[base + uint(sy) * width + uint(sx)];
            acc += k[dy * 3 + dx] * v;
        }
    }
    return acc;
}

// mode 0 unsharp (kernel = normalised Gaussian), 1 laplacian (kernel = Laplacian)
[shader("compute")]
[numthreads(16, 16, 1)]
void sharpen(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform RWStructuredBuffer<float> dst,
    uniform StructuredBuffer<float> taps,
    uniform uint height,
    uniform uint width,
    uniform uint mode,
    uniform float amount,
    uniform float threshold)
{
    uint x = tid.x;
    uint y = tid.y;
    uint plane = tid.z;
    if (x >= width || y >= height)
        return;
    uint base = plane * height * width;
    uint idx = base + y * width + x;
    float v = src[idx];
    float c = conv3(src, taps, base, int(y), int(x), height, width);
    float outv;
    if (mode == 0)
    {
        float mask = v - c;
        if (threshold > 0.0 && abs(mask) < threshold)
            mask = 0.0;
        outv = v + amount * mask;
    }
    else
    {
        outv = v + amount * c;
    }
    dst[idx] = clamp(outv, 0.0, 1.0);
}
