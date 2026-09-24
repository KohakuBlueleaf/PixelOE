// Float min / max / clamp as compare-select. The CPU target lowers the
// min() / max() / clamp() intrinsics to NaN-aware library calls (several
// times slower than a compare); every operand in these kernels is finite,
// where both forms agree. GPU targets compile either form to one instruction.

module sel;

public float fmin_s(float a, float b)
{
    return b < a ? b : a;
}

public float fmax_s(float a, float b)
{
    return b > a ? b : a;
}

public float fclamp_s(float v, float lo, float hi)
{
    return fmin_s(fmax_s(v, lo), hi);
}

public float3 fmin_s(float3 a, float3 b)
{
    return float3(fmin_s(a.x, b.x), fmin_s(a.y, b.y), fmin_s(a.z, b.z));
}

public float3 fmax_s(float3 a, float3 b)
{
    return float3(fmax_s(a.x, b.x), fmax_s(a.y, b.y), fmax_s(a.z, b.z));
}

public float3 fclamp_s(float3 v, float lo, float hi)
{
    return float3(fclamp_s(v.x, lo, hi), fclamp_s(v.y, lo, hi), fclamp_s(v.z, lo, hi));
}
