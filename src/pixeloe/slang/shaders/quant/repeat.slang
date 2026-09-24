// Repeat table for repeat-kmeans (generate_repeat_table semantics), per image
// b over n = 0 .. count-1, with N2 = 4 count:
//   lw = log(max(w, 1e-8)); lse = log(sum exp(lw - max lw)) + max lw
//   rc = exp(lw - lse) * (N2 - count); fl = floor(rc); rem = rc - fl
//   repeat = 1 + fl, plus 1 for the `need = (N2 - count) - sum fl` largest
//   remainders (ties: lowest index first).
// The largest-remainder threshold is found by an 8-pass radix select over
// order-preserving keys, with per-chunk histograms (no atomics).

import order;

[shader("compute")]
[numthreads(256, 1, 1)]
void rp_log(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> w,
    uniform RWStructuredBuffer<float> lw,
    uniform uint total)
{
    uint t = tid.x;
    if (t >= total)
        return;
    lw[t] = log(max(w[t], 1e-8));
}

// e = exp(lw - lmax[b])
[shader("compute")]
[numthreads(256, 1, 1)]
void rp_shift_exp(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lw,
    uniform StructuredBuffer<float> lmax,
    uniform RWStructuredBuffer<float> e,
    uniform uint count,
    uniform uint total)
{
    uint t = tid.x;
    if (t >= total)
        return;
    e[t] = exp(lw[t] - lmax[t / count]);
}

// repeat = 1 + fl, rem, fl (as float, for the per-image sum)
[shader("compute")]
[numthreads(256, 1, 1)]
void rp_counts(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> lw,
    uniform StructuredBuffer<float> lmax,
    uniform StructuredBuffer<float> sumexp,
    uniform RWStructuredBuffer<uint> repeat,
    uniform RWStructuredBuffer<float> rem,
    uniform RWStructuredBuffer<float> fl_out,
    uniform uint count,
    uniform uint total,
    uniform float extra) // N2 - count
{
    uint t = tid.x;
    if (t >= total)
        return;
    uint b = t / count;
    float lse = log(sumexp[b]) + lmax[b];
    float rc = exp(lw[t] - lse) * extra;
    float fl = floor(rc);
    rem[t] = rc - fl;
    fl_out[t] = fl;
    repeat[t] = 1 + uint(fl);
}

// state[b * 4 + 0] = prefix, 1 = rank (k-th smallest wanted), 2 = need, 3 = active
// need = extra - sum fl; the need largest = rank k = count - need.
[shader("compute")]
[numthreads(64, 1, 1)]
void rp_select_init(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> fl_sum,
    uniform RWStructuredBuffer<uint> state,
    uniform uint count,
    uniform uint batch,
    uniform float extra)
{
    uint b = tid.x;
    if (b >= batch)
        return;
    float left = extra - fl_sum[b];
    uint need = left > 0.0 ? uint(left) : 0u;
    state[b * 4 + 0] = 0;
    state[b * 4 + 1] = need > 0 ? count - need : 0;
    state[b * 4 + 2] = need;
    state[b * 4 + 3] = need > 0 ? 1 : 0;
}

uint digit_mask(uint shift)
{
    return shift == 28 ? 0u : ~((1u << (shift + 4)) - 1u);
}

// hist[b, chunk, d] for keys matching the current prefix
[shader("compute")]
[numthreads(64, 1, 1)]
void rp_hist(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> rem,
    uniform StructuredBuffer<uint> state,
    uniform RWStructuredBuffer<uint> hist,
    uniform uint shift,
    uniform uint count,
    uniform uint batch,
    uniform uint chunk,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= batch * chunks)
        return;
    uint b = t / chunks;
    uint h[16];
    for (uint d = 0; d < 16; d++)
        h[d] = 0;
    if (state[b * 4 + 3] != 0)
    {
        uint prefix = state[b * 4 + 0];
        uint mask = digit_mask(shift);
        uint begin = (t - b * chunks) * chunk;
        uint end = min(begin + chunk, count);
        for (uint n = begin; n < end; n++)
        {
            uint key = float_to_ordered(rem[b * count + n]);
            if ((key & mask) == prefix)
                h[(key >> shift) & 15u] += 1;
        }
    }
    for (uint d = 0; d < 16; d++)
        hist[t * 16 + d] = h[d];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rp_select_step(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<uint> hist,
    uniform RWStructuredBuffer<uint> state,
    uniform uint shift,
    uniform uint batch,
    uniform uint chunks)
{
    uint b = tid.x;
    if (b >= batch || state[b * 4 + 3] == 0)
        return;
    uint rank = state[b * 4 + 1];
    uint cum = 0;
    for (uint d = 0; d < 16; d++)
    {
        uint h = 0;
        for (uint c = 0; c < chunks; c++)
            h += hist[((b * chunks + c) * 16) + d];
        if (rank < cum + h)
        {
            state[b * 4 + 0] |= d << shift;
            state[b * 4 + 1] = rank - cum;
            return;
        }
        cum += h;
    }
}

// Per chunk: number of keys > T and == T (T = state prefix)
[shader("compute")]
[numthreads(64, 1, 1)]
void rp_tie_count(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> rem,
    uniform StructuredBuffer<uint> state,
    uniform RWStructuredBuffer<uint> chunk_gt,
    uniform RWStructuredBuffer<uint> chunk_eq,
    uniform uint count,
    uniform uint batch,
    uniform uint chunk,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= batch * chunks)
        return;
    uint b = t / chunks;
    uint gt = 0;
    uint eq = 0;
    if (state[b * 4 + 3] != 0)
    {
        uint thr = state[b * 4 + 0];
        uint begin = (t - b * chunks) * chunk;
        uint end = min(begin + chunk, count);
        for (uint n = begin; n < end; n++)
        {
            uint key = float_to_ordered(rem[b * count + n]);
            gt += key > thr ? 1 : 0;
            eq += key == thr ? 1 : 0;
        }
    }
    chunk_gt[t] = gt;
    chunk_eq[t] = eq;
}

// Exclusive prefix of ties over chunks; tie budget = need - total gt
[shader("compute")]
[numthreads(64, 1, 1)]
void rp_tie_scan(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<uint> chunk_gt,
    uniform RWStructuredBuffer<uint> chunk_eq, // becomes exclusive prefix
    uniform RWStructuredBuffer<uint> state,    // [b*4+2] becomes tie budget
    uniform uint batch,
    uniform uint chunks)
{
    uint b = tid.x;
    if (b >= batch || state[b * 4 + 3] == 0)
        return;
    uint gt = 0;
    uint run = 0;
    for (uint c = 0; c < chunks; c++)
    {
        uint i = b * chunks + c;
        gt += chunk_gt[i];
        uint e = chunk_eq[i];
        chunk_eq[i] = run;
        run += e;
    }
    state[b * 4 + 2] = state[b * 4 + 2] - gt;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rp_mark(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> rem,
    uniform StructuredBuffer<uint> state,
    uniform StructuredBuffer<uint> tie_prefix,
    uniform RWStructuredBuffer<uint> repeat,
    uniform uint count,
    uniform uint batch,
    uniform uint chunk,
    uniform uint chunks)
{
    uint t = tid.x;
    if (t >= batch * chunks)
        return;
    uint b = t / chunks;
    if (state[b * 4 + 3] == 0)
        return;
    uint thr = state[b * 4 + 0];
    uint budget = state[b * 4 + 2];
    uint seen = tie_prefix[t];
    uint begin = (t - b * chunks) * chunk;
    uint end = min(begin + chunk, count);
    for (uint n = begin; n < end; n++)
    {
        uint i = b * count + n;
        uint key = float_to_ordered(rem[i]);
        if (key > thr)
            repeat[i] += 1;
        else if (key == thr)
        {
            if (seen < budget)
                repeat[i] += 1;
            seen += 1;
        }
    }
}
