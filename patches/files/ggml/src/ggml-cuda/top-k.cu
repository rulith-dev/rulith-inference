#include "argsort.cuh"
#include "top-k.cuh"

#if defined(GGML_USE_HIP)
#include <hip/hip_version.h>
#endif
// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

template<int BLOCK_SIZE>
static __global__ void top_k_nary_search_cuda(
        const float * __restrict__ src,
        const int2 * __restrict__ src_pairs,
        int * __restrict__ dst,
        int2 * __restrict__ dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int k,
        int nrows,
        bool first_pass,
        bool last_pass) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int warp_count = BLOCK_SIZE / warpSize;

    __shared__ int2 candidates[BLOCK_SIZE];
    __shared__ uint32_t counts[64];
    __shared__ uint32_t selected_bucket;
    __shared__ uint32_t selected_total;
    __shared__ uint32_t warp_offsets[32];
    __shared__ uint32_t warp_equal_offsets[32];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col = blockIdx.x * BLOCK_SIZE + tid;
        const bool valid = col < ncols_input;
        int2 value;
        if (valid) {
            value = first_pass
                ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))
                : src_pairs[(size_t) row * ncols_input + col];
        } else {
            value = make_int2(original_ncols, (int) 0xff800000U);
        }
        candidates[tid] = value;
        __syncthreads();

        const int limit = min(k, ncols_input - blockIdx.x * BLOCK_SIZE);
        if (k == 1) {
#pragma unroll
            for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
                if (tid < stride) {
                    const int2 a = candidates[tid];
                    const int2 b = candidates[tid + stride];
                    if (a.x >= original_ncols ||
                        (b.x < original_ncols && __int_as_float(b.y) > __int_as_float(a.y))) {
                        candidates[tid] = b;
                    }
                }
                __syncthreads();
            }
        } else {
            constexpr int radix_bits = 6;
            constexpr int radix_size = 1 << radix_bits;
            int shift = 32 - radix_bits;
            uint32_t mask = ((1U << radix_bits) - 1) << shift;
            // strixllama: keys in [range_min, range_max), the bound 64-bit: +inf's key is 0xff800000 (a bound of that
            // left +inf uncounted and the selection took a wrong set when one was there), the top bucket's end 2^32
            uint32_t range_min = 0;
            uint64_t range_max = 0xff800001ULL;
            uint32_t total = 0;

            while (mask != 0) {
                __syncthreads();
                if (tid < radix_size) {
                    counts[tid] = 0;
                }
                __syncthreads();

                const uint32_t key = top_k_float_to_ordered(__int_as_float(value.y));
                if (valid && key >= range_min && key < range_max) {
                    atomicAdd(&counts[(key & mask) >> shift], 1U);
                }
                __syncthreads();

                if (tid < 32) {
                    uint32_t high_partial = counts[radix_size - 1 - tid];
                    uint32_t low_partial  = counts[31 - tid];
                    for (int offset = 1; offset < 32; offset *= 2) {
                        const uint32_t high_previous = __shfl_up(high_partial, offset, 32);
                        const uint32_t low_previous  = __shfl_up(low_partial, offset, 32);
                        if (tid >= offset) {
                            high_partial += high_previous;
                            low_partial += low_previous;
                        }
                    }
                    high_partial += total;
                    low_partial += __shfl(high_partial, 31, 32);
                    const unsigned long long high_selected = __ballot(high_partial >= (uint32_t) limit);
                    if (high_selected != 0) {
                        const int first = __ffsll(high_selected) - 1;
                        if (tid == first) {
                            selected_bucket = radix_size - 1 - first;
                            selected_total  = high_partial;
                        }
                    } else {
                        const unsigned long long low_selected = __ballot(low_partial >= (uint32_t) limit);
                        const int                first        = __ffsll(low_selected) - 1;
                        if (tid == first) {
                            selected_bucket = 31 - first;
                            selected_total  = low_partial;
                        }
                    }
                }
                __syncthreads();

                const uint32_t bucket = selected_bucket;
                total = selected_total;
                range_max = range_min + ((uint64_t) (bucket + 1) << shift);
                range_min = range_min + (bucket << shift);
                if (total == (uint32_t) limit) {
                    break;
                }
                total -= counts[bucket];
                mask >>= radix_bits;
                shift -= radix_bits;
                if (shift < 0) {
                    shift = 0;
                }
            }

            const uint32_t key = top_k_float_to_ordered(__int_as_float(value.y));
            const bool above = valid && key > range_min;
            const bool equal = valid && key == range_min;
            const unsigned long long above_mask = __ballot(above);
            const unsigned long long equal_mask = __ballot(equal);
            if (lane == 0) {
                warp_offsets[warp] = __popcll(above_mask);
                warp_equal_offsets[warp] = __popcll(equal_mask);
            }
            __syncthreads();

            uint32_t above_base = 0;
            uint32_t equal_base = 0;
            uint32_t above_total = 0;
            for (int i = 0; i < warp_count; ++i) {
                if (i < warp) {
                    above_base += warp_offsets[i];
                    equal_base += warp_equal_offsets[i];
                }
                above_total += warp_offsets[i];
            }
            equal_base += above_total;

            const unsigned long long lane_mask = lane == 0 ? 0 : (1ULL << lane) - 1;
            if (above) {
                candidates[above_base + __popcll(above_mask & lane_mask)] = value;
            }
            const uint32_t equal_index = equal_base + __popcll(equal_mask & lane_mask);
            if (equal && equal_index < (uint32_t) limit) {
                candidates[equal_index] = value;
            }
            __syncthreads();
        }

        if (tid < k) {
            if (last_pass) {
                dst[(size_t) row * k + tid] = candidates[tid].x;
            } else {
                const int output_col = blockIdx.x * k + tid;
                if (output_col < ncols_output) {
                    dst_pairs[(size_t) row * ncols_output + output_col] = candidates[tid];
                }
            }
        }
        __syncthreads();
    }
}

template <int BLOCK_SIZE, bool USE_SHUFFLE>
static __device__ __forceinline__ int2 top_k_one_reduce(int2 * candidates, int2 value, int original_ncols) {
    const int tid = threadIdx.x;
    if constexpr (USE_SHUFFLE) {
        const int     lane       = tid & 31;
        const int     wave       = tid >> 5;
        constexpr int wave_count = BLOCK_SIZE / 32;

#    pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            const int2 other = make_int2(__shfl_down(value.x, offset, 32), __shfl_down(value.y, offset, 32));
            if (lane + offset < 32 &&
                (value.x >= original_ncols ||
                 (other.x < original_ncols && __int_as_float(other.y) > __int_as_float(value.y)))) {
                value = other;
            }
        }
        if (lane == 0) {
            candidates[wave] = value;
        }
        __syncthreads();

        if (wave == 0) {
            value = lane < wave_count ? candidates[lane] : make_int2(original_ncols, (int) 0xff800000U);
#    pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                const int2 other = make_int2(__shfl_down(value.x, offset, 32), __shfl_down(value.y, offset, 32));
                if (lane + offset < 32 &&
                    (value.x >= original_ncols ||
                     (other.x < original_ncols && __int_as_float(other.y) > __int_as_float(value.y)))) {
                    value = other;
                }
            }
        }
        return value;
    } else {
        candidates[tid] = value;
        __syncthreads();

#    pragma unroll
        for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
            if (tid < stride) {
                const int2 a = candidates[tid];
                const int2 b = candidates[tid + stride];
                if (a.x >= original_ncols || (b.x < original_ncols && __int_as_float(b.y) > __int_as_float(a.y))) {
                    candidates[tid] = b;
                }
            }
            __syncthreads();
        }
        return candidates[0];
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, bool USE_SHUFFLE>
static __global__ void top_k_one_first_cuda(const float * __restrict__ src,
                                            int2 * __restrict__ dst_pairs,
                                            int original_ncols,
                                            int ncols_input,
                                            int ncols_output,
                                            int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[USE_SHUFFLE ? BLOCK_SIZE / 32 : BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col   = blockIdx.x * (ITEMS_PER_THREAD * BLOCK_SIZE) + tid;
        int2      value = col < ncols_input ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col])) :
                                              make_int2(original_ncols, (int) 0xff800000U);
#    pragma unroll
        for (int item = 1; item < ITEMS_PER_THREAD; ++item) {
            const int other_col = col + item * BLOCK_SIZE;
            if (other_col < ncols_input) {
                const int2 other = make_int2(other_col, __float_as_int(src[(size_t) row * ncols_input + other_col]));
                if (value.x >= original_ncols || __int_as_float(other.y) > __int_as_float(value.y)) {
                    value = other;
                }
            }
        }
        const int2 result = top_k_one_reduce<BLOCK_SIZE, USE_SHUFFLE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst_pairs[(size_t) row * ncols_output + blockIdx.x] = result;
        }
        __syncthreads();
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, bool USE_SHUFFLE>
static __global__ void top_k_one_first_last_cuda(const float * __restrict__ src,
                                                 int * __restrict__ dst,
                                                 int original_ncols,
                                                 int ncols_input,
                                                 int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[USE_SHUFFLE ? BLOCK_SIZE / 32 : BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col   = blockIdx.x * (ITEMS_PER_THREAD * BLOCK_SIZE) + tid;
        int2      value = col < ncols_input ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col])) :
                                              make_int2(original_ncols, (int) 0xff800000U);
#    pragma unroll
        for (int item = 1; item < ITEMS_PER_THREAD; ++item) {
            const int other_col = col + item * BLOCK_SIZE;
            if (other_col < ncols_input) {
                const int2 other = make_int2(other_col, __float_as_int(src[(size_t) row * ncols_input + other_col]));
                if (value.x >= original_ncols || __int_as_float(other.y) > __int_as_float(value.y)) {
                    value = other;
                }
            }
        }
        const int2 result = top_k_one_reduce<BLOCK_SIZE, USE_SHUFFLE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst[row] = result.x;
        }
        __syncthreads();
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, bool USE_SHUFFLE>
static __global__ void top_k_one_middle_cuda(const int2 * __restrict__ src_pairs,
                                             int2 * __restrict__ dst_pairs,
                                             int original_ncols,
                                             int ncols_input,
                                             int ncols_output,
                                             int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[USE_SHUFFLE ? BLOCK_SIZE / 32 : BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col   = blockIdx.x * (ITEMS_PER_THREAD * BLOCK_SIZE) + tid;
        int2      value = col < ncols_input ? src_pairs[(size_t) row * ncols_input + col] :
                                              make_int2(original_ncols, (int) 0xff800000U);
#    pragma unroll
        for (int item = 1; item < ITEMS_PER_THREAD; ++item) {
            const int other_col = col + item * BLOCK_SIZE;
            if (other_col < ncols_input) {
                const int2 other = src_pairs[(size_t) row * ncols_input + other_col];
                if (value.x >= original_ncols || __int_as_float(other.y) > __int_as_float(value.y)) {
                    value = other;
                }
            }
        }
        const int2 result = top_k_one_reduce<BLOCK_SIZE, USE_SHUFFLE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst_pairs[(size_t) row * ncols_output + blockIdx.x] = result;
        }
        __syncthreads();
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, bool USE_SHUFFLE>
static __global__ void top_k_one_last_cuda(const int2 * __restrict__ src_pairs,
                                           int * __restrict__ dst,
                                           int original_ncols,
                                           int ncols_input,
                                           int nrows) {
    const int tid = threadIdx.x;
    __shared__ int2 candidates[USE_SHUFFLE ? BLOCK_SIZE / 32 : BLOCK_SIZE];

    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {
        const int col   = blockIdx.x * (ITEMS_PER_THREAD * BLOCK_SIZE) + tid;
        int2      value = col < ncols_input ? src_pairs[(size_t) row * ncols_input + col] :
                                              make_int2(original_ncols, (int) 0xff800000U);
#    pragma unroll
        for (int item = 1; item < ITEMS_PER_THREAD; ++item) {
            const int other_col = col + item * BLOCK_SIZE;
            if (other_col < ncols_input) {
                const int2 other = src_pairs[(size_t) row * ncols_input + other_col];
                if (value.x >= original_ncols || __int_as_float(other.y) > __int_as_float(value.y)) {
                    value = other;
                }
            }
        }
        const int2 result = top_k_one_reduce<BLOCK_SIZE, USE_SHUFFLE>(candidates, value, original_ncols);
        if (tid == 0) {
            dst[row] = result.x;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __device__ void top_k_gather_equal(
        const float * src, int * dst, int ncols, uint32_t threshold, int limit, int offset) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int nwarps = BLOCK_SIZE / warpSize;
    __shared__ int warp_counts[32];
    int count = 0;

    for (int base = 0; base < ncols && count < limit; base += BLOCK_SIZE) {
        const int col = base + tid;
        const bool equal = col < ncols && top_k_float_to_ordered(src[col]) == threshold;
        const unsigned long long mask = __ballot(equal);
        if (lane == 0) {
            warp_counts[warp] = __popcll(mask);
        }
        __syncthreads();
        int before = count;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                before += warp_counts[w];
            }
            count += warp_counts[w];
        }
        const unsigned long long lane_mask = (1ULL << lane) - 1;
        const int pos = before + __popcll(mask & lane_mask);
        if (equal && pos < limit) {
            dst[offset + pos] = col;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_select_cuda(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int ncols,
        int nrows,
        int k) {
    constexpr int RADIX_BITS = 8;
    constexpr int RADIX_SIZE = 1 << RADIX_BITS;

    const int tid = threadIdx.x;
    __shared__ uint32_t histogram[RADIX_SIZE];
    __shared__ uint32_t selected_bucket;
    __shared__ uint32_t count_above;
    __shared__ uint32_t output_count;

    for (int row = blockIdx.x; row < nrows; row += gridDim.x) {
        const float * row_src = src + (size_t) row * ncols;
        int * row_dst = dst + (size_t) row * k;
        uint32_t prefix = 0;
        uint32_t desired = k;

#pragma unroll
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            for (int bin = tid; bin < RADIX_SIZE; bin += BLOCK_SIZE) {
                histogram[bin] = 0;
            }
            __syncthreads();

            const uint32_t high_mask = shift == 32 - RADIX_BITS ? 0 : 0xffffffffU << (shift + RADIX_BITS);
            const uint32_t prefix_high = prefix & high_mask;
            for (int col = tid; col < ncols; col += BLOCK_SIZE) {
                const uint32_t key = top_k_float_to_ordered(row_src[col]);
                if ((key & high_mask) == prefix_high) {
                    atomicAdd(&histogram[(key >> shift) & (RADIX_SIZE - 1)], 1U);
                }
            }
            __syncthreads();

            if (tid == 0) {
                uint32_t above = 0;
                uint32_t bucket = 0;
                for (int bin = RADIX_SIZE - 1; bin >= 0; --bin) {
                    const uint32_t count = histogram[bin];
                    if (above + count >= desired) {
                        bucket = bin;
                        break;
                    }
                    above += count;
                }
                selected_bucket = bucket;
                count_above = above;
            }
            __syncthreads();

            prefix |= selected_bucket << shift;
            desired -= count_above;
            __syncthreads();
        }

        if (tid == 0) {
            output_count = 0;
        }
        __syncthreads();

        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            if (top_k_float_to_ordered(row_src[col]) > prefix) {
                row_dst[atomicAdd(&output_count, 1U)] = col;
            }
        }
        __syncthreads();

        top_k_gather_equal<BLOCK_SIZE>(row_src, row_dst, ncols, prefix, desired, output_count);
    }
}

static void top_k_radix_select_cuda(
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 1024;
    const int grid_size = std::min(nrows, 65535);
    top_k_radix_select_cuda<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, nrows, k);
}

static int top_k_floor_log2(int value) {
    int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

static int top_k_ceil_log2(int value) {
    const int floor = top_k_floor_log2(value);
    return value == (1 << floor) ? floor : floor + 1;
}

static int top_k_nary_block_log2(int ncols, int k) {
    const int min_block = std::max(top_k_floor_log2(k) + 1, 6);
    if (min_block > 10) {
        return -1;
    }
    const int max_block = std::min(std::max(top_k_floor_log2(k) + 2, 8), 10);
    int block = std::min(std::max(top_k_ceil_log2(ncols), min_block), max_block);
    if (ncols > (1 << block)) {
        for (int candidate = block; candidate <= 10; ++candidate) {
            if (ncols <= (1 << candidate)) {
                block = candidate;
                break;
            }
        }
    }
    return block;
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, bool USE_SHUFFLE>
static void top_k_one_cuda_launch(const float * src,
                                  const int2 *  src_pairs,
                                  int *         dst,
                                  int2 *        dst_pairs,
                                  int           original_ncols,
                                  int           ncols_input,
                                  int           ncols_output,
                                  int           nrows,
                                  bool          first_pass,
                                  bool          last_pass,
                                  cudaStream_t  stream) {
    constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;
    const dim3    grid((ncols_input + TILE_SIZE - 1) / TILE_SIZE, std::min(nrows, 65535), 1);
    if (first_pass && last_pass) {
        top_k_one_first_last_cuda<BLOCK_SIZE, ITEMS_PER_THREAD, USE_SHUFFLE>
            <<<grid, BLOCK_SIZE, 0, stream>>>(src, dst, original_ncols, ncols_input, nrows);
    } else if (first_pass) {
        top_k_one_first_cuda<BLOCK_SIZE, ITEMS_PER_THREAD, USE_SHUFFLE>
            <<<grid, BLOCK_SIZE, 0, stream>>>(src, dst_pairs, original_ncols, ncols_input, ncols_output, nrows);
    } else if (last_pass) {
        top_k_one_last_cuda<BLOCK_SIZE, ITEMS_PER_THREAD, USE_SHUFFLE>
            <<<grid, BLOCK_SIZE, 0, stream>>>(src_pairs, dst, original_ncols, ncols_input, nrows);
    } else {
        top_k_one_middle_cuda<BLOCK_SIZE, ITEMS_PER_THREAD, USE_SHUFFLE>
            <<<grid, BLOCK_SIZE, 0, stream>>>(src_pairs, dst_pairs, original_ncols, ncols_input, ncols_output, nrows);
    }
}

template <bool USE_SHUFFLE>
static void top_k_one_cuda_launch(int           block_log2,
                                  const float * src,
                                  const int2 *  src_pairs,
                                  int *         dst,
                                  int2 *        dst_pairs,
                                  int           original_ncols,
                                  int           ncols_input,
                                  int           ncols_output,
                                  int           nrows,
                                  bool          first_pass,
                                  bool          last_pass,
                                  cudaStream_t  stream) {
    switch (block_log2) {
        case 6:
            top_k_one_cuda_launch<64, 1, USE_SHUFFLE>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                                      ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 7:
            top_k_one_cuda_launch<128, 1, USE_SHUFFLE>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                                       ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 8:
            top_k_one_cuda_launch<256, 1, USE_SHUFFLE>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                                       ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 9:
            top_k_one_cuda_launch<512, 1, USE_SHUFFLE>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                                       ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case 10:
            top_k_one_cuda_launch<1024, 1, USE_SHUFFLE>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                                        ncols_output, nrows, first_pass, last_pass, stream);
            break;
        default:
            GGML_ABORT("invalid HIP TOP_K block size");
    }
}

enum top_k_one_mode {
    TOP_K_ONE_SHARED,
    TOP_K_ONE_SHUFFLE,
    TOP_K_ONE_SHUFFLE_2,
    TOP_K_ONE_SHUFFLE_4,
};

static top_k_one_mode top_k_one_select_mode(int ncols, int nrows) {
    if (ncols <= 32 || (nrows == 1 && ncols > 131072)) {
        return TOP_K_ONE_SHARED;
    }
    if (ncols <= 4096) {
        return TOP_K_ONE_SHUFFLE;
    }
    return nrows > 1 ? TOP_K_ONE_SHUFFLE_4 : TOP_K_ONE_SHUFFLE_2;
}

static int top_k_one_items_per_thread(top_k_one_mode mode) {
    switch (mode) {
        case TOP_K_ONE_SHUFFLE_2:
            return 2;
        case TOP_K_ONE_SHUFFLE_4:
            return 4;
        default:
            return 1;
    }
}

static int top_k_one_block_log2(top_k_one_mode mode, int ncols) {
    switch (mode) {
        case TOP_K_ONE_SHUFFLE_2:
            return 7;
        case TOP_K_ONE_SHUFFLE_4:
            return 6;
        default:
            return top_k_nary_block_log2(ncols, 1);
    }
}

static void top_k_one_cuda_launch(top_k_one_mode mode,
                                  int            block_log2,
                                  const float *  src,
                                  const int2 *   src_pairs,
                                  int *          dst,
                                  int2 *         dst_pairs,
                                  int            original_ncols,
                                  int            ncols_input,
                                  int            ncols_output,
                                  int            nrows,
                                  bool           first_pass,
                                  bool           last_pass,
                                  cudaStream_t   stream) {
    switch (mode) {
        case TOP_K_ONE_SHARED:
            top_k_one_cuda_launch<false>(block_log2, src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                         ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case TOP_K_ONE_SHUFFLE:
            top_k_one_cuda_launch<true>(block_log2, src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                        ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case TOP_K_ONE_SHUFFLE_2:
            top_k_one_cuda_launch<128, 2, true>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                                ncols_output, nrows, first_pass, last_pass, stream);
            break;
        case TOP_K_ONE_SHUFFLE_4:
            top_k_one_cuda_launch<64, 4, true>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input,
                                               ncols_output, nrows, first_pass, last_pass, stream);
            break;
    }
}

template<int BLOCK_SIZE>
static void top_k_nary_search_cuda_launch(
        const float * src,
        const int2 * src_pairs,
        int * dst,
        int2 * dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int k,
        int nrows,
        bool first_pass,
        bool last_pass,
        cudaStream_t stream) {
    const dim3 grid((ncols_input + BLOCK_SIZE - 1) / BLOCK_SIZE, std::min(nrows, 65535), 1);
    top_k_nary_search_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
        src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output,
        k, nrows, first_pass, last_pass);
}

static void top_k_nary_search_cuda_launch(
        int block_log2,
        const float * src,
        const int2 * src_pairs,
        int * dst,
        int2 * dst_pairs,
        int original_ncols,
        int ncols_input,
        int ncols_output,
        int k,
        int nrows,
        bool first_pass,
        bool last_pass,
        cudaStream_t stream) {
    switch (block_log2) {
        case 6:
            top_k_nary_search_cuda_launch<64>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 7:
            top_k_nary_search_cuda_launch<128>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 8:
            top_k_nary_search_cuda_launch<256>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 9:
            top_k_nary_search_cuda_launch<512>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        case 10:
            top_k_nary_search_cuda_launch<1024>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);
            break;
        default:
            GGML_ABORT("invalid HIP TOP_K block size");
    }
}

static void top_k_small_cuda(
        ggml_cuda_pool & pool,
        const float * src,
        int * dst,
        int ncols,
        int nrows,
        int k,
        cudaStream_t stream) {
    const top_k_one_mode one_mode             = top_k_one_select_mode(ncols, nrows);
    const int            one_items_per_thread = top_k_one_items_per_thread(one_mode);
    int                  block_log2 = k == 1 ? top_k_one_block_log2(one_mode, ncols) : top_k_nary_block_log2(ncols, k);
    if (block_log2 < 0) {
        top_k_radix_select_cuda(src, dst, ncols, nrows, k, stream);
        return;
    }

    const int                  block_size       = (k == 1 ? one_items_per_thread : 1) << block_log2;
    const int first_output = (ncols / block_size) * k + std::min(k, ncols % block_size);
    const size_t scratch_elements = (size_t) first_output * nrows;
    ggml_cuda_pool_alloc<int2> scratch_alloc(pool, 2 * scratch_elements);
    int2 * scratch[2] = {scratch_alloc.get(), scratch_alloc.get() + scratch_elements};

    int ncols_input = ncols;
    int buffer = 0;
    bool first_pass = true;
    while (ncols_input > k || first_pass) {
        block_log2 = k == 1 ? top_k_one_block_log2(one_mode, ncols_input) : top_k_nary_block_log2(ncols_input, k);
        const int  current_block_size = (k == 1 ? one_items_per_thread : 1) << block_log2;
        const int ncols_output = (ncols_input / current_block_size) * k + std::min(k, ncols_input % current_block_size);
        const bool last_pass = ncols_output == k;
        if (k == 1) {
            top_k_one_cuda_launch(one_mode, block_log2, src, first_pass ? nullptr : scratch[buffer], dst,
                                  last_pass ? nullptr : scratch[buffer ^ 1], ncols, ncols_input, ncols_output, nrows,
                                  first_pass, last_pass, stream);
        } else {
            top_k_nary_search_cuda_launch(
                block_log2, src, first_pass ? nullptr : scratch[buffer], dst,
                last_pass ? nullptr : scratch[buffer ^ 1], ncols, ncols_input,
                ncols_output, k, nrows, first_pass, last_pass, stream);
        }
        ncols_input = ncols_output;
        first_pass = false;
        buffer ^= 1;
    }
}

struct top_k_parallel_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_parallel_radix_init(top_k_parallel_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_parallel_radix_histogram(
        const float * __restrict__ src,
        const top_k_parallel_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_parallel_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_parallel_radix_select(
        const int * __restrict__ block_histograms,
        top_k_parallel_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_parallel_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_parallel_radix_reset_counters(top_k_parallel_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_parallel_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_parallel_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_parallel_radix_state * state = &states[row];

    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        }
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_parallel_radix_gather_equal(
        const float * src, int * dst, const top_k_parallel_radix_state * states, int ncols, int k) {
    const int row = blockIdx.x;
    const top_k_parallel_radix_state state = states[row];
    top_k_gather_equal<BLOCK_SIZE>(src + (size_t) row*ncols, dst + (size_t) row*k,
                                  ncols, state.prefix, state.rank, k - state.rank);
}

static void top_k_parallel_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_parallel_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_parallel_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_parallel_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_parallel_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_parallel_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_parallel_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_parallel_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
    top_k_parallel_radix_gather_equal<BLOCK_SIZE>
        <<<nrows, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k);
}

// strixllama: top_k_rows begin
// One block per row with the row's keys in registers: a radix select over them (four 8-bit passes, an LDS histogram per
// wave), then one pass that writes the selection. top_k_parallel_radix_cuda reads the whole matrix
// five times (four histograms, the gather) and scans it again in column order for the ties: at the QSA indexer's
// 512 rows of 16.9K blocks (64K context) that is 1.17 ms a strip; this reads it once.
// The selected set is the same - every element above the k-th largest, then the lowest columns equal to it - but not
// the order within a row, which neither kernel specifies (the QSA callers sort it).
template <int BLOCK, int ITEMS>
static __global__ void __launch_bounds__(BLOCK) top_k_rows_cuda(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int nrows, const int k, const int items) {
    constexpr int NWAVES = BLOCK / 32;
    // a histogram per wave, so same-digit lanes only collide within a wave; two 16-bit counts a word (bins d and d+128)
    // keep it at 16 KB, so two blocks share a WGP and one's loads overlap the other's passes
    __shared__ uint32_t whist[NWAVES][128];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t sel_bucket;
    __shared__ uint32_t sel_above;
    __shared__ uint32_t out_count;
    __shared__ uint32_t wave_eq[NWAVES];

    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const uint64_t lanes_below = (1ull << lane) - 1;

    {   // one row a block (a grid-stride loop would hoist every slot's address out of it and hold them all)
        const int     row     = blockIdx.x;
        const float * row_src = src + (size_t) row * ncols;
        int *         row_dst = dst + (size_t) row * k;

        // key 0 marks a slot past the row (a real key is at least 1: only a negative NaN maps to 0, and it is moved up)
        uint32_t key[ITEMS];
#pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int c = j * BLOCK + tid;
            key[j] = j < items && c < ncols ? max(top_k_float_to_ordered(row_src[c]), 1u) : 0u;
        }

        // the k-th largest key, one 8-bit digit a pass from the top; desired ends as the number of elements equal to
        // it that belong to the selection, eq_total as the number there are
        uint32_t prefix = 0, pmask = 0, desired = (uint32_t) k, eq_total = 0;
#pragma unroll 1
        for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
            for (int b = lane; b < 128; b += 32) {
                whist[wave][b] = 0;
            }
#pragma unroll
            for (int j = 0; j < ITEMS; ++j) {
                if (key[j] != 0 && (key[j] & pmask) == prefix) {
                    const uint32_t d = (key[j] >> shift) & 255u;
                    atomicAdd(&whist[wave][d & 127u], 1u << (16 * (d >> 7)));
                }
            }
            __syncthreads();
            for (int b = tid; b < 256; b += BLOCK) {
                uint32_t sum = 0;
#pragma unroll
                for (int w = 0; w < NWAVES; ++w) {
                    sum += (whist[w][b & 127] >> (16 * (b >> 7))) & 0xffffu;
                }
                hist[b] = sum;
            }
            __syncthreads();
            if (wave == 0) {
                // lane l holds bins 255-8l down to 248-8l; the lane whose range reaches the desired-th largest finds it
                uint32_t cnt[8], sum = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    cnt[i] = hist[255 - 8 * lane - i];
                    sum += cnt[i];
                }
                uint32_t incl = sum;
#pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    const uint32_t o = __shfl_up(incl, off, 32);
                    if (lane >= off) {
                        incl += o;
                    }
                }
                const uint32_t excl = incl - sum;
                if (excl < desired && incl >= desired) {
                    uint32_t above = excl;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (above + cnt[i] >= desired) {
                            sel_bucket = 255 - 8 * lane - i;
                            sel_above  = above;
                            break;
                        }
                        above += cnt[i];
                    }
                }
            }
            __syncthreads();
            const uint32_t b = sel_bucket;
            prefix  |= b << shift;
            pmask   |= 255u << shift;
            desired -= sel_above;
            if (shift == 0) {
                eq_total = hist[b];
            }
            __syncthreads();
        }

        // everything above the threshold, and the equal ones too when all of them belong to the selection: each wave
        // counts its own, the waves take consecutive ranges
        const uint32_t thr         = prefix;
        const bool     take_all_eq = eq_total == desired;
        uint32_t mine = 0;
#pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            mine += (uint32_t) __popcll(__ballot(key[j] > thr || (take_all_eq && key[j] == thr)));
        }
        if (lane == 0) {
            wave_eq[wave] = mine;
        }
        __syncthreads();
        uint32_t base = 0;
        for (int w = 0; w < NWAVES; ++w) {
            base += w < wave ? wave_eq[w] : 0;
        }
        if (tid == BLOCK - 1) {
            out_count = base + mine;
        }
#pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const bool     sel = key[j] > thr || (take_all_eq && key[j] == thr);
            const uint64_t m   = __ballot(sel);
            if (sel) {
                row_dst[base + __popcll(m & lanes_below)] = j * BLOCK + tid;
            }
            base += (uint32_t) __popcll(m);
        }
        if (!take_all_eq) {
            // ties at the threshold: the lowest columns first, as top_k_gather_equal takes them (column = j * block + tid)
            __syncthreads();
            const uint32_t first = out_count;   // k - desired
            uint32_t taken = 0;
            // which of this thread's slots hold the threshold, as bits: the loop below then indexes no registers
            uint64_t eqbits[(ITEMS + 63) / 64] = {};
#pragma unroll
            for (int j = 0; j < ITEMS; ++j) {
                eqbits[j / 64] |= (uint64_t) (key[j] == thr) << (j % 64);
            }
            for (int j = 0; j < items && taken < desired; ++j) {
                const bool     eq = ((j < 64 ? eqbits[0] : eqbits[(ITEMS + 63) / 64 - 1]) >> (j % 64)) & 1;
                const uint64_t m  = __ballot(eq);
                if (lane == 0) {
                    wave_eq[wave] = (uint32_t) __popcll(m);
                }
                __syncthreads();
                uint32_t before = 0, total = 0;
                for (int w = 0; w < NWAVES; ++w) {
                    const uint32_t x = wave_eq[w];
                    before += w < wave ? x : 0;
                    total  += x;
                }
                const uint32_t rank = taken + before + (uint32_t) __popcll(m & lanes_below);
                if (eq && rank < desired) {
                    row_dst[first + rank] = j * BLOCK + tid;
                }
                taken += total;
                __syncthreads();
            }
        }
    }
    GGML_UNUSED(nrows);
}

template <int BLOCK>
static void top_k_rows_launch(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    const dim3 grid((unsigned) nrows), block(BLOCK);
    const int items = (ncols + BLOCK - 1) / BLOCK;
    if (items <= 4) {
        top_k_rows_cuda<BLOCK, 4><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 8) {
        top_k_rows_cuda<BLOCK, 8><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 16) {
        top_k_rows_cuda<BLOCK, 16><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 24) {
        top_k_rows_cuda<BLOCK, 24><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 32) {
        top_k_rows_cuda<BLOCK, 32><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 40) {
        top_k_rows_cuda<BLOCK, 40><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 48) {
        top_k_rows_cuda<BLOCK, 48><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else if (items <= 64) {
        top_k_rows_cuda<BLOCK, 64><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    } else {
        top_k_rows_cuda<BLOCK, 128><<<grid, block, 0, stream>>>(src, dst, ncols, nrows, k, items);
    }
}

static bool top_k_rows_cuda(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    static const bool on = !getenv("STRIX_TOP_K_ROWS") || atoi(getenv("STRIX_TOP_K_ROWS")) != 0;
    static const int  bs = getenv("STRIX_TOP_K_ROWS_BLOCK") ? atoi(getenv("STRIX_TOP_K_ROWS_BLOCK")) : 1024;
    // strixllama: from one row up (it took 32 and more): the QSA decode's TOP_K has a row a query - eight conversations
    // are eight rows of ~20-40K blocks - and top_k_parallel_radix_cuda's dozen launches cost ~340 us a strip there against
    // ~10 us here (a step at 8 x 20K 103 -> 97.6 ms); one conversation, at 110K or 210K, is unchanged. The same set either
    // way (see above). STRIX_TOP_K_ROWS_MIN=32 restores the old bound.
    static const int  min_rows = getenv("STRIX_TOP_K_ROWS_MIN") ? atoi(getenv("STRIX_TOP_K_ROWS_MIN")) : 1;
    if (!on || nrows < min_rows || nrows > 65535 || ncols <= 1024 || k < 1 || k > ncols || ncols > 128 * bs) {
        return false;
    }
    switch (bs) {
        case 256: top_k_rows_launch<256>(src, dst, ncols, nrows, k, stream); break;
        case 512: top_k_rows_launch<512>(src, dst, ncols, nrows, k, stream); break;
        default:  top_k_rows_launch<1024>(src, dst, ncols, nrows, k, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
// strixllama: top_k_rows end

static bool top_k_use_small_kernel(int ncols, int nrows, int k) {
    if (k == 1) {
        return true;
    }
#if HIP_VERSION >= 71500000
    if (ncols <= 1024) {
        return true;
    }
    const uint64_t elements = (uint64_t) ncols * nrows;
    if (k <= 32) {
        return elements <= (1U << 20);
    }
    return nrows == 1 && ncols <= (1U << 17);
#else
    GGML_UNUSED(ncols);
    GGML_UNUSED(nrows);
    return false;
#endif
}
#endif

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
#ifdef CUB_TOP_K_AVAILABLE
    ggml_cuda_pool & pool = ctx.pool();
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_USE_HIP)
    ggml_cuda_pool & pool = ctx.pool();
    if (top_k_use_small_kernel(ncols, nrows, k)) {
        top_k_small_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else if (top_k_rows_cuda(src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream)) {
        // done
    } else if (ncols > 1024) {
        top_k_parallel_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int * tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    ggml_cuda_pool & pool = ctx.pool();
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
