// Outline-expansion weight: fold (overlap-add average) of the lattice
// statistics back to full resolution, then the polarity mapping.
//
// mapping 0 (current / polarity):
//     sigmoid(avg (med - .5) + dist (dark - bright))
// mapping 1 (contrast_ratio):
//     sigmoid((med - .5) avg - ((bright - dark) / (lc + 1e-8)) dist)
// mapping 2 (contrast_gated), first half: w = sigmoid(current), lc_out = lc;
//     weight_gate then applies w = .5 + (w - .5) sigmoid((lc - mean lc) dist)

struct Lattice
{
    StructuredBuffer<float> stat;
    uint lat_h;
    uint lat_w;
    uint ksize;
    uint stride;
    uint pad;
}

uint first_patch(uint pos, uint ksize, uint stride, uint pad)
{
    int lo = int(pos) + int(pad) - int(ksize) + 1;
    return lo <= 0 ? 0u : (uint(lo) + stride - 1) / stride;
}

// sum of covering patch stats / (covering count + 1e-8), F.fold order
float fold_gather(Lattice lat, uint b, uint y, uint x)
{
    uint i0 = first_patch(y, lat.ksize, lat.stride, lat.pad);
    uint i1 = min((y + lat.pad) / lat.stride + 1, lat.lat_h);
    uint j0 = first_patch(x, lat.ksize, lat.stride, lat.pad);
    uint j1 = min((x + lat.pad) / lat.stride + 1, lat.lat_w);
    float acc = 0.0;
    float count = 0.0;
    for (uint i = i0; i < i1; i++)
    {
        uint row = (b * lat.lat_h + i) * lat.lat_w;
        for (uint j = j0; j < j1; j++)
        {
            acc += lat.stat[row + j];
            count += 1.0;
        }
    }
    return acc / (count + 1e-8);
}

float sigmoid(float v)
{
    return 1.0 / (1.0 + exp(-v));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void weight_raw(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> med_stat,
    uniform StructuredBuffer<float> min_stat,
    uniform StructuredBuffer<float> max_stat,
    uniform RWStructuredBuffer<float> w_out,
    uniform RWStructuredBuffer<float> lc_out,
    uniform uint height,
    uniform uint width,
    uniform uint med_lat_h,
    uniform uint med_lat_w,
    uniform uint med_ksize,
    uniform uint med_pad,
    uniform uint mm_lat_h,
    uniform uint mm_lat_w,
    uniform uint mm_ksize,
    uniform uint mm_pad,
    uniform uint stride,
    uniform uint mapping,
    uniform float avg_scale,
    uniform float dist_scale)
{
    uint x = tid.x;
    uint y = tid.y;
    uint b = tid.z;
    if (x >= width || y >= height)
        return;
    Lattice med_lat = { med_stat, med_lat_h, med_lat_w, med_ksize, stride, med_pad };
    Lattice min_lat = { min_stat, mm_lat_h, mm_lat_w, mm_ksize, stride, mm_pad };
    Lattice max_lat = { max_stat, mm_lat_h, mm_lat_w, mm_ksize, stride, mm_pad };
    float med = fold_gather(med_lat, b, y, x);
    float mn = fold_gather(min_lat, b, y, x);
    float mx = fold_gather(max_lat, b, y, x);
    uint o = (b * height + y) * width + x;
    map_weight(med, mn, mx, mapping, avg_scale, dist_scale, o, w_out, lc_out);
}

void map_weight(
    float med,
    float mn,
    float mx,
    uint mapping,
    float avg_scale,
    float dist_scale,
    uint o,
    RWStructuredBuffer<float> w_out,
    RWStructuredBuffer<float> lc_out)
{
    float bright = mx - med;
    float dark = med - mn;
    float lc = bright + dark;
    float w;
    if (mapping == 1)
    {
        float asym = (bright - dark) / (lc + 1e-8);
        w = sigmoid((med - 0.5) * avg_scale - asym * dist_scale);
    }
    else
    {
        w = sigmoid((med - 0.5) * avg_scale + (dark - bright) * dist_scale);
    }
    w_out[o] = w;
    if (mapping == 2)
        lc_out[o] = lc;
}

// Same mapping from per-pixel (sliding-window) statistic maps.
[shader("compute")]
[numthreads(256, 1, 1)]
void weight_dense(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<float> med,
    uniform StructuredBuffer<float> mn,
    uniform StructuredBuffer<float> mx,
    uniform RWStructuredBuffer<float> w_out,
    uniform RWStructuredBuffer<float> lc_out,
    uniform uint count,
    uniform uint mapping,
    uniform float avg_scale,
    uniform float dist_scale,
    uniform uint row_len)
{
    uint q = tid.y * row_len + tid.x;
    if (q >= count)
        return;
    map_weight(med[q], mn[q], mx[q], mapping, avg_scale, dist_scale, q, w_out, lc_out);
}

// contrast_gated second half; lc_mean[0] = mean(lc) over the whole batch
[shader("compute")]
[numthreads(256, 1, 1)]
void weight_gate(
    uint3 tid: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> w,
    uniform StructuredBuffer<float> lc,
    uniform StructuredBuffer<float> lc_mean,
    uniform uint count,
    uniform float dist_scale,
    uniform uint row_len)
{
    uint q = tid.y * row_len + tid.x;
    if (q >= count)
        return;
    float gate = sigmoid((lc[q] - lc_mean[0]) * dist_scale);
    w[q] = 0.5 + (w[q] - 0.5) * gate;
}
