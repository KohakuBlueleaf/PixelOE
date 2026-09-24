// GPU-only (barrier) single-dispatch variant of dither.slang's error
// diffusion: one 256-thread workgroup per image runs every row-pair step,
// with device-memory barriers between the error and update phases instead
// of two dispatches per step. Same arithmetic as ed_errors / ed_apply.

import palette;

static const uint GROUP = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void ed_all_steps(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform RWStructuredBuffer<float> img,
    uniform StructuredBuffer<float> pal,
    uniform RWStructuredBuffer<float> err, // [b, 3 rows, 3 ch, width]
    uniform uint K,
    uniform uint height,
    uniform uint width)
{
    uint b = gid.x;
    uint plane = height * width;
    for (uint y = 0; y + 2 < height; y += 2)
    {
        for (uint i = lid.x; i < 3 * width; i += GROUP)
        {
            uint r = i / width;
            uint x = i - r * width;
            uint base = b * 3 * plane + (y + r) * width + x;
            float3 v = load_px(img, base, plane);
            float3 e = v - load_pal(pal, b, nearest(v, pal, b, K), K);
            uint o = ((b * 3 + r) * 3) * width + x;
            err[o] = e.x;
            err[o + width] = e.y;
            err[o + 2 * width] = e.z;
        }
        AllMemoryBarrierWithGroupSync();
        for (uint i = lid.x; i < 2 * width; i += GROUP)
        {
            uint r = i / width;
            uint x = i - r * width;
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
        AllMemoryBarrierWithGroupSync();
    }
}
