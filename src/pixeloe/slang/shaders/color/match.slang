// match_color: global Lab mean/std transfer followed by the wavelet colour
// fix. With B the (linear) 5-level blur chain,
//     inp - B(inp) + B(tgt) == inp + B(tgt - inp),
// so the colour fix blurs one difference image instead of two images.

import lab;

// stats = [mean_src, std_src, mean_tgt, std_tgt]
// inp  = lab_to_rgb((lab(src) - mean_src) / std_src * std_tgt + mean_tgt)
// diff = tgt - inp
[shader("compute")]
[numthreads(256, 1, 1)]
void match_apply(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> src,
    uniform StructuredBuffer<float> tgt,
    uniform StructuredBuffer<float> stats,
    uniform RWStructuredBuffer<float> inp,
    uniform RWStructuredBuffer<float> diff,
    uniform uint pixels,
    uniform uint hw,
    uniform uint row_len)
{
    uint q = tid.y * row_len + tid.x;
    if (q >= pixels)
        return;
    uint b = q / hw;
    uint base = b * 3 * hw + (q - b * hw);
    float mean_s = stats[0];
    float std_s = stats[1];
    float mean_t = stats[2];
    float std_t = stats[3];
    float3 lab = rgb_to_lab(float3(src[base], src[base + hw], src[base + 2 * hw]));
    lab = (lab - mean_s) / std_s;
    lab = lab * std_t + mean_t;
    float3 rgb = lab_to_rgb(lab);
    inp[base] = rgb.x;
    inp[base + hw] = rgb.y;
    inp[base + 2 * hw] = rgb.z;
    diff[base] = tgt[base] - rgb.x;
    diff[base + hw] = tgt[base + hw] - rgb.y;
    diff[base + 2 * hw] = tgt[base + 2 * hw] - rgb.z;
}
