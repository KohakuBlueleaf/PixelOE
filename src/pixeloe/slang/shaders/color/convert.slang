// Planar [B, 3, H, W] sRGB -> Lab conversions.

import lab;

// lum[b, pix] = L / 100
[shader("compute")]
[numthreads(256, 1, 1)]
void luminance(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform RWStructuredBuffer<float> lum,
    uniform uint pixels, // B * H * W
    uniform uint hw,
    uniform uint row_len)
{
    uint q = tid.y * row_len + tid.x;
    if (q >= pixels)
        return;
    uint b = q / hw;
    uint base = b * 3 * hw + (q - b * hw);
    float3 lab = rgb_to_lab(float3(img[base], img[base + hw], img[base + 2 * hw]));
    lum[q] = lab.x / 100.0;
}

// lab[b, c, pix] = rgb_to_lab(img[b, :, pix])[c]
[shader("compute")]
[numthreads(256, 1, 1)]
void lab_image(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> img,
    uniform RWStructuredBuffer<float> lab_out,
    uniform uint pixels,
    uniform uint hw,
    uniform uint row_len)
{
    uint q = tid.y * row_len + tid.x;
    if (q >= pixels)
        return;
    uint b = q / hw;
    uint base = b * 3 * hw + (q - b * hw);
    float3 lab = rgb_to_lab(float3(img[base], img[base + hw], img[base + 2 * hw]));
    lab_out[base] = lab.x;
    lab_out[base + hw] = lab.y;
    lab_out[base + 2 * hw] = lab.z;
}
