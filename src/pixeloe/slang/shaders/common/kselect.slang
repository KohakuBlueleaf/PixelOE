// Per-thread order statistics over an abstract key source. Barrier-free, so
// the same code runs on every Slang target, CPU included.
//
// Radix select over order-preserving uint keys (order.slang), 4-bit digits,
// most significant first: 8 passes over the n keys, O(n) work and O(1)
// storage, exact for any n (ties are values, as in torch.median/min/max).

module kselect;

import order;

public interface IKeySource
{
    public uint count();
    public uint key(uint i);
}

uint digit_mask(int shift)
{
    return shift == 28 ? 0u : ~((1u << uint(shift + 4)) - 1u);
}

// k-th smallest (0-based) key of the source. Once the selected bucket holds
// a single key, one scan fetches it instead of resolving the remaining
// digits (continuous data isolates the target after ~2 passes).
public uint radix_select<S : IKeySource>(S src, uint k)
{
    uint n = src.count();
    uint prefix = 0;
    uint rank = k;
    for (int shift = 28; shift >= 0; shift -= 4)
    {
        uint hist[16];
        for (uint d = 0; d < 16; d++)
            hist[d] = 0;
        uint mask = digit_mask(shift);
        for (uint i = 0; i < n; i++)
        {
            uint key = src.key(i);
            if ((key & mask) == prefix)
                hist[(key >> uint(shift)) & 15u] += 1;
        }
        uint cum = 0;
        uint picked = 0;
        for (uint d = 0; d < 16; d++)
        {
            uint h = hist[d];
            if (rank < cum + h)
            {
                prefix |= d << uint(shift);
                rank -= cum;
                picked = h;
                break;
            }
            cum += h;
        }
        if (picked == 1 && shift > 0)
        {
            uint done_mask = ~((1u << uint(shift)) - 1u);
            for (uint i = 0; i < n; i++)
            {
                uint key = src.key(i);
                if ((key & done_mask) == prefix)
                    return key;
            }
        }
    }
    return prefix;
}

// Lower median, matching torch.median (element (n - 1) / 2 of the sorted keys).
public float radix_median<S : IKeySource>(S src)
{
    return ordered_to_float(radix_select(src, (src.count() - 1) / 2));
}
