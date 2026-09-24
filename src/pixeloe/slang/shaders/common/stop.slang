// Global early stop of the reference k-means loops: after an iteration,
// `if max|new - old| < 1/256: break` over the whole batch.

module stop;

public static const float STOP_EPS = 0.00390625; // 1 / 256

public bool stopped(RWStructuredBuffer<float> diff, uint it)
{
    return it > 0 && diff[it - 1] < STOP_EPS;
}
