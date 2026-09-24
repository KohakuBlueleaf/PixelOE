// Palette helpers: planar pixels img[b, c, n], palette pal[b, k, c].

module palette;

public float3 load_px(RWStructuredBuffer<float> img, uint base, uint plane)
{
    return float3(img[base], img[base + plane], img[base + 2 * plane]);
}

public float3 load_pal(StructuredBuffer<float> pal, uint b, uint k, uint K)
{
    uint o = (b * K + k) * 3;
    return float3(pal[o], pal[o + 1], pal[o + 2]);
}

// (palette - pixel)^2 summed over channels, as the reference computes it
public float dist2(float3 a, float3 c)
{
    float3 d = c - a;
    d = d * d;
    return d.x + d.y + d.z;
}

// nearest palette index, lowest index on ties (torch argmin)
public uint nearest(float3 v, StructuredBuffer<float> pal, uint b, uint K)
{
    uint best = 0;
    float best_d = asfloat(0x7f800000u);
    for (uint k = 0; k < K; k++)
    {
        float d = dist2(v, load_pal(pal, b, k, K));
        if (d < best_d)
        {
            best_d = d;
            best = k;
        }
    }
    return best;
}

// reflect padding of one column index (F.pad mode="reflect", pad 1)
public int reflect_col(int x, int width)
{
    if (x < 0)
        return -x;
    if (x >= width)
        return 2 * (width - 1) - x;
    return x;
}

public static const float ED_K0[3] = { 0.0, 0.0, 0.4375 };
public static const float ED_K1[3] = { 0.1875, 0.3125, 0.0625 };
