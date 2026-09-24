// Order-preserving float <-> uint mapping. For any floats a < b (no NaN),
// float_to_ordered(a) < float_to_ordered(b) as unsigned integers, so float
// min/max/selection can run on integer atomics and radix digits.

module order;

public uint float_to_ordered(float f)
{
    uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

public float ordered_to_float(uint k)
{
    uint u = (k & 0x80000000u) != 0 ? (k & 0x7fffffffu) : ~k;
    return asfloat(u);
}

public static const uint ORDERED_POS_INF = 0xff800000u; // float_to_ordered(+inf)
public static const uint ORDERED_NEG_INF = 0x007fffffu; // float_to_ordered(-inf)
