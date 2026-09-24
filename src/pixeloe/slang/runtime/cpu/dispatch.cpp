// Parallel group dispatcher for Slang CPU-target kernels.
//
// A Slang kernel compiled for the CPU exports
//     void entry(ComputeVaryingInput* range, void* entryPointParams, void* globals)
// which runs every workgroup in [range.start, range.end). This library splits
// a dispatch's group grid across an OpenMP thread pool: one work item per
// (y, z) group row when there are enough rows, otherwise per slice of x.

#include <cstdint>
#include <omp.h>

struct uint3
{
    uint32_t x, y, z;
};

struct ComputeVaryingInput
{
    uint3 start;
    uint3 end;
};

typedef void (*ComputeFunc)(ComputeVaryingInput*, void*, void*);

extern "C" __declspec(dllexport) int pixeloe_max_threads()
{
    return omp_get_max_threads();
}

extern "C" __declspec(dllexport) void pixeloe_set_threads(int n)
{
    omp_set_num_threads(n);
}

extern "C" __declspec(dllexport) void pixeloe_dispatch(
    void* fn_ptr, uint32_t gx, uint32_t gy, uint32_t gz, void* params)
{
    ComputeFunc fn = reinterpret_cast<ComputeFunc>(fn_ptr);
    const int64_t rows = int64_t(gy) * int64_t(gz);
    const int threads = omp_get_max_threads();
    if (rows >= 2 * threads || gx == 1)
    {
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t r = 0; r < rows; r++)
        {
            uint32_t z = uint32_t(r / gy);
            uint32_t y = uint32_t(r - int64_t(z) * gy);
            ComputeVaryingInput vi = {{0, y, z}, {gx, y + 1, z + 1}};
            fn(&vi, params, nullptr);
        }
        return;
    }
    const int64_t slices = gx < uint32_t(4 * threads) ? int64_t(gx) : int64_t(4 * threads);
    const int64_t items = slices * rows;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < items; i++)
    {
        int64_t r = i / slices;
        int64_t s = i - r * slices;
        uint32_t z = uint32_t(r / gy);
        uint32_t y = uint32_t(r - int64_t(z) * gy);
        uint32_t x0 = uint32_t(int64_t(gx) * s / slices);
        uint32_t x1 = uint32_t(int64_t(gx) * (s + 1) / slices);
        ComputeVaryingInput vi = {{x0, y, z}, {x1, y + 1, z + 1}};
        fn(&vi, params, nullptr);
    }
}
