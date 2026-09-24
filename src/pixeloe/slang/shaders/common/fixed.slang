// Exact, order-independent sums: values are converted to 32.32 fixed point
// (truncation error < 2^-32 per value) and added in int64, so every target
// and every summation order produces the same bits. Range: |sum| < 2^31.

module fixed;

static const float FIXED_ONE = 4294967296.0; // 2^32
static const float FIXED_INV = 2.3283064365386963e-10; // 2^-32

public int64_t to_fixed(float v)
{
    return int64_t(v * FIXED_ONE);
}

public float from_fixed(int64_t s)
{
    return float(s) * FIXED_INV;
}

public struct Fixed3
{
    public int64_t x;
    public int64_t y;
    public int64_t z;

    public __init() { x = 0; y = 0; z = 0; }

    [mutating]
    public void add(float3 v, int64_t m)
    {
        x += to_fixed(v.x) * m;
        y += to_fixed(v.y) * m;
        z += to_fixed(v.z) * m;
    }

    // sum / n, n > 0
    public float3 mean(float n)
    {
        return float3(from_fixed(x), from_fixed(y), from_fixed(z)) / n;
    }
}
