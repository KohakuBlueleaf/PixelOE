// Kahan-compensated float accumulators. Exact only where the target compiler
// keeps IEEE ordering (CUDA, CPU); a compiler that reassociates degrades them
// to a plain sum. Sums whose rounding feeds back into a decision (k-means
// means) use fixed.slang instead, which is exact on every target.

module kahan;

public struct Kahan1
{
    public float sum;
    public float comp;

    public __init() { sum = 0.0; comp = 0.0; }

    [mutating]
    public void add(float v)
    {
        float y = v - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
}
