// SPDX-License-Identifier: MIT
//
// strixllama: the routed EXL3 experts of a prompt-sized batch as one pipeline. The data flow and the grouped GEMM come
// from Kyojin's gfx11.5 port of ExLlamaV3 (MIT, vendors/kyojin): the (token, pick) rows are sorted by expert, gate and up
// run as one grouped GEMM with the trellis decoded inside it, one pass applies their output rotations, SwiGLU and the
// down projection's input rotation, the down GEMM writes F32 rows, and the reduce applies the down output rotation and
// sums the token's picks with their routing weights. Ours: the sort (stable in token order, so every run puts a row at
// the same position - the GEMM's result for a row depends on its parity inside the tile), the F32 input, the rotations
// and SwiGLU in F32 (one F16 rounding where the next GEMM reads), and the shared expert's gated tail in the reduce.
#if defined(GGML_USE_HIP)
#define USE_ROCM 1
#include "vendors/kyojin/mpw2.cuh"
#include "mmh-moe.cuh"

#include <algorithm>

namespace {

constexpr int   SORT_BLOCK = 1024;
constexpr float HS         = 0.088388347648f;   // 1 / sqrt(128)
constexpr int   MT         = 128;               // rows per GEMM tile (mpw2, MT 128)

int64_t align256(const int64_t x) { return (x + 255) & ~int64_t(255); }

// H128 of the 128 values lane t holds 4 of (4t .. 4t + 3), scaled to orthonormal
__device__ __forceinline__ void moe_had128(float v[4], const int t) {
    float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v[0] = h0 * HS; v[1] = h1 * HS; v[2] = h2 * HS; v[3] = h3 * HS;
}

__device__ __forceinline__ void moe_load_h4(const half * p, float v[4]) {
    const half4 h = *(const half4 *) p;
    v[0] = __low2float(h.x); v[1] = __high2float(h.x); v[2] = __low2float(h.y); v[3] = __high2float(h.y);
}

__device__ __forceinline__ void moe_store_h4(half * p, const float v[4]) {
    *(half4 *) p = half4(__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3]));
}

// per-expert pointer tables: gate / up / down trellis, then suh / svh of gate, up, down
__global__ void moe_tables(const ggml_cuda_mmh_moe_args a, int64_t * __restrict__ t) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int E = a.E;
    if (e >= E) {
        return;
    }
    t[0 * E + e] = (int64_t) ((const char *) a.w_gate + (size_t) e * a.es_gate);
    t[1 * E + e] = (int64_t) ((const char *) a.w_up   + (size_t) e * a.es_up);
    t[2 * E + e] = (int64_t) ((const char *) a.w_down + (size_t) e * a.es_down);
    t[3 * E + e] = (int64_t) ((const half *) a.rin_g  + e * a.rs_in_g);
    t[4 * E + e] = (int64_t) ((const half *) a.rout_g + e * a.rs_out_g);
    t[5 * E + e] = (int64_t) ((const half *) a.rin_u  + e * a.rs_in_u);
    t[6 * E + e] = (int64_t) ((const half *) a.rout_u + e * a.rs_out_u);
    t[7 * E + e] = (int64_t) ((const half *) a.rin_d  + e * a.rs_in_d);
    t[8 * E + e] = (int64_t) ((const half *) a.rout_d + e * a.rs_out_d);
}

__device__ __forceinline__ int moe_id(const int32_t * ids, const int64_t s, const int n_used, const int64_t ids_ld) {
    return ids[(s / n_used) * ids_ld + s % n_used];
}

__global__ void __launch_bounds__(1024) moe_block_count(const int32_t * __restrict__ ids, const int n_used, const int64_t ids_ld,
        const int A, const int E, int32_t * __restrict__ block_counts) {
    extern __shared__ int s_cnt[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        s_cnt[e] = 0;
    }
    __syncthreads();
    const int s = blockIdx.x * SORT_BLOCK + threadIdx.x;
    if (s < A) {
        atomicAdd(&s_cnt[moe_id(ids, s, n_used, ids_ld)], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        block_counts[(int64_t) blockIdx.x * E + e] = s_cnt[e];
    }
}

// per expert: each block's first position among the expert's rows, and the expert's total
__global__ void moe_block_scan(int32_t * __restrict__ block_counts, const int nb, const int E, int64_t * __restrict__ counts) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e > E) {
        return;
    }
    if (e == E) {
        counts[E] = 0;
        return;
    }
    int run = 0;
    for (int b = 0; b < nb; ++b) {
        const int c = block_counts[(int64_t) b * E + e];
        block_counts[(int64_t) b * E + e] = run;
        run += c;
    }
    counts[e] = run;
}

// sorted position of every (token, pick): the expert's offset + the block's first position + the rank among the block's
// earlier rows of the same expert (stable: token order inside each expert)
__global__ void __launch_bounds__(1024) moe_place(const int32_t * __restrict__ ids, const int n_used, const int64_t ids_ld,
        const int A, const int E,
        const int32_t * __restrict__ block_off, const int64_t * __restrict__ offsets, int64_t * __restrict__ inverse,
        int32_t * __restrict__ pick_e, int32_t * __restrict__ sorted_e) {
    __shared__ int s_e[SORT_BLOCK];
    const int s = blockIdx.x * SORT_BLOCK + threadIdx.x;
    const int e = s < A ? moe_id(ids, s, n_used, ids_ld) : -1;
    s_e[threadIdx.x] = e;
    __syncthreads();
    if (s >= A) {
        return;
    }
    int rank = 0;
    for (int j = 0; j < (int) threadIdx.x; ++j) {
        rank += s_e[j] == e;
    }
    const int64_t pos = offsets[e] + block_off[(int64_t) blockIdx.x * E + e] + rank;
    inverse[s]    = pos;
    pick_e[s]     = e;
    sorted_e[pos] = e;
}

// one wave per (token, 128-block): the row is read once and written, rotated for gate and for up, at each pick's sorted
// position (gate rows 0 .. A - 1, up rows A .. 2A - 1)
__global__ void __launch_bounds__(256) moe_gather(const ggml_cuda_mmh_moe_args a, const int64_t * __restrict__ tab,
        half * __restrict__ out) {
    const int     blocks = a.H / 128;
    const int64_t job    = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) a.T * blocks) {
        return;
    }
    const int     t   = threadIdx.x & 31;
    const int     blk = job % blocks;
    const int64_t r   = job / blocks;
    const int64_t A   = (int64_t) a.T * a.n_used;
    const float4  xv  = *(const float4 *) (a.x + r * a.x_ld + blk * 128 + 4 * t);
    for (int k = 0; k < a.n_used; ++k) {
        const int64_t s   = r * a.n_used + k;
        const int64_t pos = a.inverse[s];
        const int     e   = a.pick_e[s];
        float sg[4], su[4];
        moe_load_h4((const half *) tab[3 * a.E + e] + blk * 128 + 4 * t, sg);
        moe_load_h4((const half *) tab[5 * a.E + e] + blk * 128 + 4 * t, su);
        float g[4] = { xv.x * sg[0], xv.y * sg[1], xv.z * sg[2], xv.w * sg[3] };
        float u[4] = { xv.x * su[0], xv.y * su[1], xv.z * su[2], xv.w * su[3] };
        moe_had128(g, t);
        moe_had128(u, t);
        moe_store_h4(out + pos * a.H + blk * 128 + 4 * t, g);
        moe_store_h4(out + (A + pos) * a.H + blk * 128 + 4 * t, u);
    }
}

// one wave per (sorted row, 128-block): output rotations of gate and up, silu(g) * u, the down input rotation; written
// over the gate row
__global__ void __launch_bounds__(256) moe_act(const ggml_cuda_mmh_moe_args a, const int64_t * __restrict__ tab,
        half * __restrict__ gu) {
    const int     blocks = a.I / 128;
    const int64_t A      = (int64_t) a.T * a.n_used;
    const int64_t job    = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= A * blocks) {
        return;
    }
    const int     t    = threadIdx.x & 31;
    const int     blk  = job % blocks;
    const int64_t slot = job / blocks;
    const int     e    = a.sorted_e[slot];
    half * gp = gu + slot * a.I + blk * 128 + 4 * t;
    half * up = gu + (A + slot) * a.I + blk * 128 + 4 * t;
    float g[4], u[4], vg[4], vu[4], sd[4];
    moe_load_h4(gp, g);
    moe_load_h4(up, u);
    moe_load_h4((const half *) tab[4 * a.E + e] + blk * 128 + 4 * t, vg);
    moe_load_h4((const half *) tab[6 * a.E + e] + blk * 128 + 4 * t, vu);
    moe_load_h4((const half *) tab[7 * a.E + e] + blk * 128 + 4 * t, sd);
    moe_had128(g, t);
    moe_had128(u, t);
    float h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float x = g[j] * vg[j];
        h[j] = x / (1.0f + expf(-x)) * (u[j] * vu[j]) * sd[j];
    }
    moe_had128(h, t);
    moe_store_h4(gp, h);
}

// one wave per (token, 128-block): sum over the picks of w * svh * H128(down row), plus the shared expert's gated tail
__global__ void __launch_bounds__(256) moe_reduce(const ggml_cuda_mmh_moe_args a, const int64_t * __restrict__ tab,
        const float * __restrict__ rows) {
    const int     blocks = a.H / 128;
    const int64_t job    = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= (int64_t) a.T * blocks) {
        return;
    }
    const int     t   = threadIdx.x & 31;
    const int     blk = job % blocks;
    const int64_t r   = job / blocks;
    float sum[4] = { 0.f, 0.f, 0.f, 0.f };
    for (int k = 0; k < a.n_used; ++k) {
        const int64_t s   = r * a.n_used + k;
        const int64_t pos = a.inverse[s];
        const int     e   = a.pick_e[s];
        const float   w   = a.weights[s];
        const float4  fv  = *(const float4 *) (rows + pos * a.H + blk * 128 + 4 * t);
        float v[4] = { fv.x, fv.y, fv.z, fv.w }, sv[4];
        moe_had128(v, t);
        moe_load_h4((const half *) tab[8 * a.E + e] + blk * 128 + 4 * t, sv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum[j] += v[j] * sv[j] * w;
        }
    }
    if (a.sx) {
        const float  g  = a.sg[r];
        const float4 sx = *(const float4 *) (a.sx + r * a.H + blk * 128 + 4 * t);
        sum[0] += sx.x * g; sum[1] += sx.y * g; sum[2] += sx.z * g; sum[3] += sx.w * g;
    }
    *(float4 *) (a.out + r * a.H + blk * 128 + 4 * t) = make_float4(sum[0], sum[1], sum[2], sum[3]);
}

// the served instances of the other engine for 3- and 4-bit experts: MT 128, PF 2, NW 2, LDM 1, mul1, OPT 3 (direct LDS
// fragment reads) plus 256 (tail-wave skip) when N is not a multiple of 256
template <int bits, bool OUT_FP32>
void moe_gemm(const dim3 grid, hipStream_t st, const half * A, const int64_t a_proj, const int64_t * offs, const int32_t * tiles,
        const int32_t * nt, const int64_t * t0, const int64_t * t1, void * C, const int64_t c_proj, const int K, const int N) {
    if (N % 256) {
        mpw2_gemm_kernel<bits, false, OUT_FP32, 2, 1, MT, 2, 2, 0, 259><<<grid, mpw2::THREADS, 0, st>>>(
            A, a_proj, offs, tiles, nt, t0, t1, C, c_proj, K, N, K);
    } else {
        mpw2_gemm_kernel<bits, false, OUT_FP32, 2, 1, MT, 2, 2, 0, 3><<<grid, mpw2::THREADS, 0, st>>>(
            A, a_proj, offs, tiles, nt, t0, t1, C, c_proj, K, N, K);
    }
}

template <bool OUT_FP32>
void moe_gemm_bits(const int bits, const dim3 grid, hipStream_t st, const half * A, const int64_t a_proj, const int64_t * offs,
        const int32_t * tiles, const int32_t * nt, const int64_t * t0, const int64_t * t1, void * C, const int64_t c_proj,
        const int K, const int N) {
    if (bits == 4) {
        moe_gemm<4, OUT_FP32>(grid, st, A, a_proj, offs, tiles, nt, t0, t1, C, c_proj, K, N);
    } else {
        moe_gemm<3, OUT_FP32>(grid, st, A, a_proj, offs, tiles, nt, t0, t1, C, c_proj, K, N);
    }
}

} // namespace

ggml_cuda_mmh_moe_scratch ggml_cuda_mmh_moe_scratch_size(const int T, const int n_used, const int E) {
    ggml_cuda_mmh_moe_scratch s;
    s.A     = (int64_t) T * n_used;
    s.nb    = (s.A + SORT_BLOCK - 1) / SORT_BLOCK;
    s.slots = (s.A + MT - 1) / MT + E;
    s.persist_bytes = align256(8 * s.A) + align256(4 * s.A) + align256(8 * 9 * (int64_t) E);
    s.temp_bytes    = align256(4 * s.A) + align256(4 * s.nb * E) + 2 * align256(8 * (E + 1)) + align256(4 * s.slots) + align256(4);
    return s;
}

void ggml_cuda_mmh_moe_scratch_carve(ggml_cuda_mmh_moe_args & a, void * persist, void * temp) {
    const ggml_cuda_mmh_moe_scratch s = ggml_cuda_mmh_moe_scratch_size(a.T, a.n_used, a.E);
    char * p = (char *) persist;
    auto take = [&p](const int64_t bytes) { char * q = p; p += align256(bytes); return (void *) q; };
    a.inverse      = (int64_t *) take(8 * s.A);
    a.pick_e       = (int32_t *) take(4 * s.A);
    a.tables       = (int64_t *) take(8 * 9 * (int64_t) a.E);
    p = (char *) temp;
    a.sorted_e     = (int32_t *) take(4 * s.A);
    a.block_counts = (int32_t *) take(4 * s.nb * a.E);
    a.counts       = (int64_t *) take(8 * (a.E + 1));
    a.offsets      = (int64_t *) take(8 * (a.E + 1));
    a.tiles        = (int32_t *) take(4 * s.slots);
    a.num_tiles    = (int32_t *) take(4);
}

bool ggml_cuda_mmh_moe_supported(const int H, const int I, const int n_used, const int bits_gu, const int bits_d) {
    const auto bits_ok = [](const int b) { return b == 3 || b == 4; };
    return bits_ok(bits_gu) && bits_ok(bits_d) && H % 128 == 0 && I % 128 == 0 && H % 32 == 0 && I % 32 == 0 &&
        n_used >= 1 && n_used <= 32;
}

void ggml_cuda_mmh_moe_run_main(const ggml_cuda_mmh_moe_args & a, void * stream) {
    hipStream_t st = (hipStream_t) stream;
    const ggml_cuda_mmh_moe_scratch s = ggml_cuda_mmh_moe_scratch_size(a.T, a.n_used, a.E);
    const int E = a.E;
    const int A = (int) s.A;

    moe_tables<<<(E + 255) / 256, 256, 0, st>>>(a, a.tables);
    moe_block_count<<<(int) s.nb, SORT_BLOCK, E * sizeof(int), st>>>(a.ids, a.n_used, a.ids_ld, A, E, a.block_counts);
    moe_block_scan<<<(E + 1 + 255) / 256, 256, 0, st>>>(a.block_counts, (int) s.nb, E, a.counts);
    mpw_metadata_kernel<<<1, 256, 0, st>>>(a.counts, nullptr, a.offsets, nullptr, a.tiles, a.num_tiles, E, A, MT);
    moe_place<<<(int) s.nb, SORT_BLOCK, 0, st>>>(a.ids, a.n_used, a.ids_ld, A, E, a.block_counts, a.offsets, a.inverse, a.pick_e,
                                                  a.sorted_e);

    half * had = (half *) a.buf_had;
    half * gu  = (half *) a.buf_gu;
    const int64_t gjobs = (int64_t) a.T * (a.H / 128);
    moe_gather<<<(int) ((gjobs + 7) / 8), 256, 0, st>>>(a, a.tables, had);

    const int64_t * tg = a.tables, * tu = a.tables + E, * td = a.tables + 2 * E;
    moe_gemm_bits<false>(a.bits_gu, dim3((a.I + 255) / 256, (unsigned) s.slots, 2), st, had, (int64_t) A * a.H, a.offsets,
                         a.tiles, a.num_tiles, tg, tu, gu, (int64_t) A * a.I, a.H, a.I);

    const int64_t ajobs = (int64_t) A * (a.I / 128);
    moe_act<<<(int) ((ajobs + 7) / 8), 256, 0, st>>>(a, a.tables, gu);

    float * down = (float *) a.buf_had;   // the gate / up inputs are dead: the down product takes their bytes
    moe_gemm_bits<true>(a.bits_d, dim3((a.H + 255) / 256, (unsigned) s.slots, 1), st, gu, 0, a.offsets, a.tiles, a.num_tiles,
                        td, td, down, 0, a.I, a.H);
}

void ggml_cuda_mmh_moe_run_reduce(const ggml_cuda_mmh_moe_args & a, void * stream) {
    const int64_t rjobs = (int64_t) a.T * (a.H / 128);
    moe_reduce<<<(int) ((rjobs + 7) / 8), 256, 0, (hipStream_t) stream>>>(a, a.tables, (const float *) a.buf_had);
}

#endif // defined(GGML_USE_HIP)
