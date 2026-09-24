// Early-stop bookkeeping shared by the iterative kernels (k-means palette,
// k-centroid). Iteration `it` writes one |new - old| max per work item into
// item_diff[]; diff_partial / diff_final fold them into diff[it]. Iteration
// it runs only while diff[it - 1] >= 1/256; a skipped iteration leaves its
// slot at 0, which stops every later one. diff[] starts zeroed.
// (Portable replacement of an atomic max: the CPU target has no atomics.)

import stop;

[shader("compute")]
[numthreads(64, 1, 1)]
void diff_partial(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> item_diff,
    uniform RWStructuredBuffer<float> part,
    uniform RWStructuredBuffer<float> diff,
    uniform uint it,
    uniform uint items,
    uniform uint chunk,
    uniform uint chunks)
{
    uint c = tid.x;
    if (c >= chunks || stopped(diff, it))
        return;
    uint end = min((c + 1) * chunk, items);
    float m = 0.0;
    for (uint i = c * chunk; i < end; i++)
        m = max(m, item_diff[i]);
    part[c] = m;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void diff_final(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> part,
    uniform RWStructuredBuffer<float> diff,
    uniform uint it,
    uniform uint chunks)
{
    if (stopped(diff, it))
        return;
    float m = 0.0;
    for (uint c = 0; c < chunks; c++)
        m = max(m, part[c]);
    diff[it] = m;
}
