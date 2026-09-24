// GPU-only (groupshared + barrier) fusion of outline expansion's
// morphology chain (morph.slang semantics):
//     blended = erode(img, se_e) * w + clamp(dilate(img, se_d)) * (1 - w)
//     out     = erode(clamp(dilate(dilate(erode(blended)))))   (se_o each)
// A workgroup owns a TILE x TILE output tile of one plane. Each pass runs
// over a region that shrinks by the next pass's radius, ping-ponging
// between two groupshared buffers; cells outside the image hold 0 after
// every pass, which is the zero padding each separate pass applies.
// The host checks the regions fit (else morph.slang runs pass by pass).

static const int TILE = 32;
static const int THREADS = 256;
static const uint IMG_CAP = 3072;
static const uint BUF_CAP = 2400;
groupshared float g_img[IMG_CAP];
groupshared float g_a[BUF_CAP];
groupshared float g_b[BUF_CAP];

float normalized_w(StructuredBuffer<float> wmin, StructuredBuffer<float> wmax, float w, uint norm_mode, uint b)
{
    if (norm_mode == 0)
        return w;
    uint s = norm_mode == 1 ? 0 : b;
    float mn = wmin[s];
    return (w - mn) / (wmax[s] - mn + 1e-8);
}

// one morphology pass: dst region (side n, origin o) from src region
// (side n + 2 r, origin o - r); dilate: max(v + k - 1), erode: min(v - k + 1)
void pass_region(
    bool from_a,
    StructuredBuffer<float> se,
    uint ks,
    int n,
    int oy,
    int ox,
    uint height,
    uint width,
    bool dilate,
    bool do_clamp,
    uint flat)
{
    int r = int(ks / 2);
    int sn = n + 2 * r;
    for (int e = int(flat); e < n * n; e += THREADS)
    {
        int ty = e / n;
        int tx = e - ty * n;
        int gy = oy + ty;
        int gx = ox + tx;
        float acc = 0.0;
        if (gy >= 0 && gx >= 0 && gy < int(height) && gx < int(width))
        {
            acc = dilate ? -asfloat(0x7f800000u) : asfloat(0x7f800000u);
            for (int dy = 0; dy < int(ks); dy++)
            {
                for (int dx = 0; dx < int(ks); dx++)
                {
                    int si = (ty + dy) * sn + tx + dx;
                    float v = from_a ? g_a[si] : g_b[si];
                    float k = se[uint(dy) * ks + uint(dx)];
                    if (dilate)
                        acc = max(acc, v + k - 1.0);
                    else
                        acc = min(acc, v - k + 1.0);
                }
            }
            if (do_clamp)
                acc = clamp(acc, 0.0, 1.0);
        }
        if (from_a)
            g_b[e] = acc;
        else
            g_a[e] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
}

// Compile-time structuring-element sizes: blend KB x KB (erode and dilate),
// open/close KO x KO; elements held in registers, tap loops unrolled.
void pass_fixed<let KO : int>(
    bool from_a,
    float se[KO * KO],
    int n,
    int oy,
    int ox,
    uint height,
    uint width,
    bool dilate,
    bool do_clamp,
    uint flat)
{
    const int R = KO / 2;
    int sn = n + 2 * R;
    for (int e = int(flat); e < n * n; e += THREADS)
    {
        int ty = e / n;
        int tx = e - ty * n;
        int gy = oy + ty;
        int gx = ox + tx;
        float acc = 0.0;
        if (gy >= 0 && gx >= 0 && gy < int(height) && gx < int(width))
        {
            acc = dilate ? -asfloat(0x7f800000u) : asfloat(0x7f800000u);
            [unroll]
            for (int dy = 0; dy < KO; dy++)
            {
                [unroll]
                for (int dx = 0; dx < KO; dx++)
                {
                    int si = (ty + dy) * sn + tx + dx;
                    float v = from_a ? g_a[si] : g_b[si];
                    if (dilate)
                        acc = max(acc, v + se[dy * KO + dx] - 1.0);
                    else
                        acc = min(acc, v - se[dy * KO + dx] + 1.0);
                }
            }
            if (do_clamp)
                acc = clamp(acc, 0.0, 1.0);
        }
        if (from_a)
            g_b[e] = acc;
        else
            g_a[e] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
}

void oe_morph_fixed<let KB : int, let KO : int>(
    uint3 gid,
    uint3 lid,
    StructuredBuffer<float> img,
    StructuredBuffer<float> w_raw,
    StructuredBuffer<float> wmin,
    StructuredBuffer<float> wmax,
    StructuredBuffer<float> se_erode,
    StructuredBuffer<float> se_dilate,
    StructuredBuffer<float> se_open,
    RWStructuredBuffer<float> dst,
    RWStructuredBuffer<float> w_out,
    uint height,
    uint width,
    uint norm_mode)
{
    const int RO = KO / 2;
    const int RB = KB / 2;
    float se_e[KB * KB];
    float se_d[KB * KB];
    [unroll]
    for (int i = 0; i < KB * KB; i++)
    {
        se_e[i] = se_erode[i];
        se_d[i] = se_dilate[i];
    }
    float se_o[KO * KO];
    [unroll]
    for (int i = 0; i < KO * KO; i++)
        se_o[i] = se_open[i];

    uint flat = lid.y * 32 + lid.x;
    uint plane = gid.z;
    uint b = plane / 3;
    uint hw = height * width;
    uint base = plane * hw;
    int ty0 = int(gid.y) * TILE;
    int tx0 = int(gid.x) * TILE;
    const int HALO = 4 * RO + RB;
    const int NI = TILE + 2 * HALO;
    for (int e = int(flat); e < NI * NI; e += THREADS)
    {
        int y = ty0 - HALO + e / NI;
        int x = tx0 - HALO + e % NI;
        float v = 0.0;
        if (y >= 0 && x >= 0 && y < int(height) && x < int(width))
            v = img[base + uint(y) * width + uint(x)];
        g_img[e] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    const int NB = TILE + 8 * RO;
    for (int e = int(flat); e < NB * NB; e += THREADS)
    {
        int ty = e / NB;
        int tx = e - ty * NB;
        int gy = ty0 - 4 * RO + ty;
        int gx = tx0 - 4 * RO + tx;
        float out_v = 0.0;
        if (gy >= 0 && gx >= 0 && gy < int(height) && gx < int(width))
        {
            float ev = asfloat(0x7f800000u);
            float dv = -asfloat(0x7f800000u);
            [unroll]
            for (int dy = 0; dy < KB; dy++)
            {
                [unroll]
                for (int dx = 0; dx < KB; dx++)
                {
                    float v = g_img[(ty + dy) * NI + tx + dx];
                    ev = min(ev, v - se_e[dy * KB + dx] + 1.0);
                    dv = max(dv, v + se_d[dy * KB + dx] - 1.0);
                }
            }
            dv = clamp(dv, 0.0, 1.0);
            uint pix = uint(gy) * width + uint(gx);
            float w = normalized_w(wmin, wmax, w_raw[b * hw + pix], norm_mode, b);
            out_v = ev * w + dv * (1.0 - w);
            bool inside = ty >= 4 * RO && tx >= 4 * RO && ty < 4 * RO + TILE && tx < 4 * RO + TILE;
            if (inside && plane - b * 3 == 0)
                w_out[b * hw + pix] = w;
        }
        g_a[e] = out_v;
    }
    GroupMemoryBarrierWithGroupSync();

    pass_fixed<KO>(true, se_o, TILE + 6 * RO, ty0 - 3 * RO, tx0 - 3 * RO, height, width, false, false, flat);
    pass_fixed<KO>(false, se_o, TILE + 4 * RO, ty0 - 2 * RO, tx0 - 2 * RO, height, width, true, false, flat);
    pass_fixed<KO>(true, se_o, TILE + 2 * RO, ty0 - RO, tx0 - RO, height, width, true, true, flat);
    pass_fixed<KO>(false, se_o, TILE, ty0, tx0, height, width, false, false, flat);

    for (int e = int(flat); e < TILE * TILE; e += THREADS)
    {
        int y = ty0 + e / TILE;
        int x = tx0 + e % TILE;
        if (y < int(height) && x < int(width))
            dst[base + uint(y) * width + uint(x)] = g_a[e];
    }
}

#define OE_MORPH_FIXED(NAME, KB, KO)                                        \
    [shader("compute")]                                                     \
    [numthreads(32, 8, 1)]                                                  \
    void NAME(                                                              \
        uint3 gid: SV_GroupID,                                              \
        uint3 lid: SV_GroupThreadID,                                        \
        uniform StructuredBuffer<float> img,                                \
        uniform StructuredBuffer<float> w_raw,                              \
        uniform StructuredBuffer<float> wmin,                               \
        uniform StructuredBuffer<float> wmax,                               \
        uniform StructuredBuffer<float> se_erode,                           \
        uniform StructuredBuffer<float> se_dilate,                          \
        uniform StructuredBuffer<float> se_open,                            \
        uniform RWStructuredBuffer<float> dst,                              \
        uniform RWStructuredBuffer<float> w_out,                            \
        uniform uint height,                                                \
        uniform uint width,                                                 \
        uniform uint norm_mode)                                             \
    {                                                                       \
        oe_morph_fixed<KB, KO>(gid, lid, img, w_raw, wmin, wmax, se_erode,  \
                               se_dilate, se_open, dst, w_out, height,      \
                               width, norm_mode);                           \
    }

OE_MORPH_FIXED(oe_morph_fused_b3o3, 3, 3)
OE_MORPH_FIXED(oe_morph_fused_b5o3, 5, 3)
OE_MORPH_FIXED(oe_morph_fused_b5o5, 5, 5)
OE_MORPH_FIXED(oe_morph_fused_b7o5, 7, 5)

// Runtime structuring-element sizes (any erode / dilate / open sizes).
[shader("compute")]
[numthreads(32, 8, 1)]
void oe_morph_fused(
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    uniform StructuredBuffer<float> img,
    uniform StructuredBuffer<float> w_raw,
    uniform StructuredBuffer<float> wmin,
    uniform StructuredBuffer<float> wmax,
    uniform StructuredBuffer<float> se_erode,
    uniform StructuredBuffer<float> se_dilate,
    uniform StructuredBuffer<float> se_open,
    uniform RWStructuredBuffer<float> dst,
    uniform RWStructuredBuffer<float> w_out,
    uniform uint ks_erode,
    uniform uint ks_dilate,
    uniform uint ks_open,
    uniform uint height,
    uniform uint width,
    uniform uint norm_mode)
{
    uint flat = lid.y * 32 + lid.x;
    uint plane = gid.z;
    uint b = plane / 3;
    uint hw = height * width;
    uint base = plane * hw;
    int ro = int(ks_open / 2);
    int rb = int(max(ks_erode, ks_dilate) / 2);
    int ty0 = int(gid.y) * TILE;
    int tx0 = int(gid.x) * TILE;

    // image region: output tile + 4 ro (four passes) + rb (the blend)
    int halo = 4 * ro + rb;
    int ni = TILE + 2 * halo;
    for (int e = int(flat); e < ni * ni; e += THREADS)
    {
        int y = ty0 - halo + e / ni;
        int x = tx0 - halo + e % ni;
        float v = 0.0;
        if (y >= 0 && x >= 0 && y < int(height) && x < int(width))
            v = img[base + uint(y) * width + uint(x)];
        g_img[e] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    // blend over the tile + 4 ro into g_a
    int nb = TILE + 8 * ro;
    int re = int(ks_erode / 2);
    int rd = int(ks_dilate / 2);
    for (int e = int(flat); e < nb * nb; e += THREADS)
    {
        int ty = e / nb;
        int tx = e - ty * nb;
        int gy = ty0 - 4 * ro + ty;
        int gx = tx0 - 4 * ro + tx;
        float out_v = 0.0;
        if (gy >= 0 && gx >= 0 && gy < int(height) && gx < int(width))
        {
            float ev = asfloat(0x7f800000u);
            for (int dy = 0; dy < int(ks_erode); dy++)
                for (int dx = 0; dx < int(ks_erode); dx++)
                {
                    float v = g_img[(ty + rb - re + dy) * ni + tx + rb - re + dx];
                    ev = min(ev, v - se_erode[uint(dy) * ks_erode + uint(dx)] + 1.0);
                }
            float dv = -asfloat(0x7f800000u);
            for (int dy = 0; dy < int(ks_dilate); dy++)
                for (int dx = 0; dx < int(ks_dilate); dx++)
                {
                    float v = g_img[(ty + rb - rd + dy) * ni + tx + rb - rd + dx];
                    dv = max(dv, v + se_dilate[uint(dy) * ks_dilate + uint(dx)] - 1.0);
                }
            dv = clamp(dv, 0.0, 1.0);
            uint pix = uint(gy) * width + uint(gx);
            float w = normalized_w(wmin, wmax, w_raw[b * hw + pix], norm_mode, b);
            out_v = ev * w + dv * (1.0 - w);
            bool inside = ty >= 4 * ro && tx >= 4 * ro && ty < 4 * ro + TILE && tx < 4 * ro + TILE;
            if (inside && plane - b * 3 == 0)
                w_out[b * hw + pix] = w;
        }
        g_a[e] = out_v;
    }
    GroupMemoryBarrierWithGroupSync();

    pass_region(true, se_open, ks_open, TILE + 6 * ro, ty0 - 3 * ro, tx0 - 3 * ro, height, width, false, false, flat);
    pass_region(false, se_open, ks_open, TILE + 4 * ro, ty0 - 2 * ro, tx0 - 2 * ro, height, width, true, false, flat);
    pass_region(true, se_open, ks_open, TILE + 2 * ro, ty0 - ro, tx0 - ro, height, width, true, true, flat);
    pass_region(false, se_open, ks_open, TILE, ty0, tx0, height, width, false, false, flat);

    for (int e = int(flat); e < TILE * TILE; e += THREADS)
    {
        int y = ty0 + e / TILE;
        int x = tx0 + e % TILE;
        if (y < int(height) && x < int(width))
            dst[base + uint(y) * width + uint(x)] = g_a[e];
    }
}
