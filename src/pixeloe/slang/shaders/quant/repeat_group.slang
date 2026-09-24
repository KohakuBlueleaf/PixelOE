// GPU-only (groupshared + barrier) form of repeat.slang's largest-remainder
// selection: one 1024-thread workgroup per image does rp_select_init, the
// eight radix passes, the tie count / scan and rp_mark in one dispatch.
// Thread t owns the contiguous index range [t span, (t + 1) span), so the
// exclusive scan of per-thread tie counts keeps "lowest index first".
// Digit and > counts use integer groupshared atomics (exact, any order).

import order;

static const uint THREADS = 1024;
groupshared uint g_hist[16];
groupshared uint g_state[4]; // prefix, rank, need, greater-than total
groupshared uint g_eq[THREADS];

uint digit_mask(uint shift)
{
    return shift == 28 ? 0u : ~((1u << (shift + 4)) - 1u);
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void rp_select_group(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform StructuredBuffer<float> rem,
    uniform StructuredBuffer<float> fl_sum,
    uniform RWStructuredBuffer<uint> repeat,
    uniform uint count,
    uniform float extra) // N2 - count
{
    uint b = gid.x;
    uint t = lid.x;
    float left = extra - fl_sum[b];
    uint need = left > 0.0 ? uint(left) : 0u;
    if (need == 0)
        return;
    uint span = (count + THREADS - 1) / THREADS;
    uint begin = min(t * span, count);
    uint end = min(begin + span, count);
    uint base = b * count;
    if (t == 0)
    {
        g_state[0] = 0;
        g_state[1] = count - need;
        g_state[3] = 0;
    }
    for (int shift = 28; shift >= 0; shift -= 4)
    {
        if (t < 16)
            g_hist[t] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint prefix = g_state[0];
        uint mask = digit_mask(uint(shift));
        for (uint n = begin; n < end; n++)
        {
            uint key = float_to_ordered(rem[base + n]);
            if ((key & mask) == prefix)
                InterlockedAdd(g_hist[(key >> uint(shift)) & 15u], 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0)
        {
            uint rank = g_state[1];
            uint cum = 0;
            for (uint d = 0; d < 16; d++)
            {
                uint h = g_hist[d];
                if (rank < cum + h)
                {
                    g_state[0] = prefix | (d << uint(shift));
                    g_state[1] = rank - cum;
                    break;
                }
                cum += h;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint thr = g_state[0];
    uint gt = 0;
    uint eq = 0;
    for (uint n = begin; n < end; n++)
    {
        uint key = float_to_ordered(rem[base + n]);
        gt += key > thr ? 1u : 0u;
        eq += key == thr ? 1u : 0u;
    }
    g_eq[t] = eq;
    InterlockedAdd(g_state[3], gt);
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        uint run = 0;
        for (uint i = 0; i < THREADS; i++)
        {
            uint e = g_eq[i];
            g_eq[i] = run;
            run += e;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint budget = need - g_state[3];
    uint seen = g_eq[t];
    for (uint n = begin; n < end; n++)
    {
        uint i = base + n;
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
