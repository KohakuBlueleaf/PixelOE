// Padding rule of the wavelet colour-fix blur (torch.nn.functional.pad per
// level): reflect when H > r and W > r, else replicate.

module blur_pad;

// Valid outputs only reach i in [-r, n - 1 + r] with r < n, where one
// reflection is exact; the final clamp keeps the halo loads of discarded
// out-of-image tile positions in bounds.
public int pad_index(int i, int n, uint reflect)
{
    if (reflect != 0)
    {
        if (i < 0)
            i = -i;
        else if (i >= n)
            i = 2 * (n - 1) - i;
    }
    return clamp(i, 0, n - 1);
}
