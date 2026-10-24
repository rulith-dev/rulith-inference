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
#ifndef MMH_RING_AUX
#define MMH_RING_AUX 6   // strixllama: the rings' weight loads with the SLC and DLC cache-policy bits (weights are read once). 4.05 MTP off,
                         // interleaved: GPU 30.2 -> 28.6 ms a token, 0 (as before) / 4 (Crucible's static cache) / 1: 30.2 / 29.0 /
                         // 31.4 ms. The same bits
#endif
#include "vendors/crucible/exl3_gemv.h"
#include "vendors/crucible/lds_stream_ring.h"
#include "mmh-moe.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cstring>

int ggml_cuda_strix_ab();   // the side of the in-run A/B (ggml-cuda.cu, DEV)

#define MOE_CHECK(x) do { if (!(x)) { fprintf(stderr, "mmh-moe.cu:%d: %s\n", __LINE__, #x); abort(); } } while (0)

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

// the weighted sum of a token's NP picks for one 128-block (moe_reduce): every load first, then the sums in pick order
template <int NP>
__device__ __forceinline__ void moe_reduce_picks(const ggml_cuda_mmh_moe_args & a, const int64_t * __restrict__ tab,
        const float * __restrict__ rows, const int64_t r, const int blk, const int t, float (&sum)[4]) {
    int64_t pos[NP];
    int     e[NP];
    float   w[NP];
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        const int64_t s = r * NP + k;
        pos[k] = a.inverse[s];
        e[k]   = a.pick_e[s];
        w[k]   = a.weights[s];
    }
    float4 fv[NP];
    half4  sh[NP];
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        fv[k] = *(const float4 *) (rows + pos[k] * a.H + blk * 128 + 4 * t);
        sh[k] = *(const half4 *) ((const half *) tab[8 * a.E + e[k]] + blk * 128 + 4 * t);
    }
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        float v[4] = { fv[k].x, fv[k].y, fv[k].z, fv[k].w };
        const float sv[4] = { __low2float(sh[k].x), __high2float(sh[k].x), __low2float(sh[k].y), __high2float(sh[k].y) };
        moe_had128(v, t);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum[j] += v[j] * sv[j] * w[k];
        }
    }
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
    // strixllama: the picks' indices, then their rows and rotations, all in flight before the first sum (a pick at a time
    // was a chain of four dependent loads each); the sums keep their order. A pick count of its own instance: predicated
    // loads would make the compiler wait for every load before the next
    if (a.batch_loads && a.n_used == 10) {
        moe_reduce_picks<10>(a, tab, rows, r, blk, t, sum);
    } else if (a.batch_loads && a.n_used == 8) {
        moe_reduce_picks<8>(a, tab, rows, r, blk, t, sum);
    } else {
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
    }
    if (a.s_w_gate) {
        // the decode pipeline's shared expert: H128 and the output rotation of its down rows (its MUL_MAT_HAD), then
        // the gate (MUL) and the sum (ADD)
        const float  g  = a.sg[r];
        const float4 fv = *(const float4 *) (a.s_raw + r * a.H + blk * 128 + 4 * t);
        float v[4] = { fv.x, fv.y, fv.z, fv.w }, sv[4];
        moe_had128(v, t);
        moe_load_h4((const half *) a.s_rout_d + blk * 128 + 4 * t, sv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float y = v[j] * sv[j];
            sum[j] += y * g;
        }
    } else if (a.sx) {
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
    // (256-row tiles with one 16-column tile a wave, 10-24: gate/up -2%, down +8%, the same bits: not taken)
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

// ---- decode-sized batches ----------------------------------------------------------------------------------------------
// strixllama: the GEMV and the fused launch are Crucible's MoE decode (vendors/crucible, Apache-2.0: MoeGemvCols and
// MoeFusedKernel of engine/layers/moe_rows.h), adapted to this fork's weight layout and arithmetic (vendors/crucible/
// README.md). The picks of the batch are grouped by expert ("jobs"); one launch runs every job's gate and up workgroups,
// then its down workgroups, each of which waits for the gate / up workgroups of its own expert (they finish in dispatch
// order before it, so the wait cannot deadlock), computes the SwiGLU step into LDS and runs the down GEMV - the down work
// fills the gate / up tail instead of starting after a kernel boundary.

constexpr int DEC_PF   = 4;    // tiles in flight per column block (two blocks a wave)
constexpr int DEC_ITEM = 16;   // column blocks a workgroup: wave w takes b0 + w and b0 + w + 8

// the tile loop of one wave over CC column blocks of one expert matrix of k tiles [0, ak) (Crucible's MoeGemvCols). This
// fork's layout: column block b is column tile b % 8 of 128-row group b / 8, whose k tile a sits ((b / 8) * ak + a) * 8 + b % 8
// tiles from the matrix start - a wave steps 8 tiles a k tile, and the 8 waves of a workgroup read their group's tiles
// side by side. xs: R rows of ak * 8 dwords (fp16 pairs in k order); out[r] gets columns 16b + lane / 4 and + 8. Each
// (row, column) keeps one accumulator chain in tile order whatever R is.
template <int K, int R, int CC>
__device__ __forceinline__ void dec_gemv_cols(const uint32_t * __restrict__ w, const int ak, const int (&b)[CC],
        const uint32_t * __restrict__ xs, float * const (&out)[R]) {
    using namespace crucible;
    const int lane = threadIdx.x & 31;
    const LaneWindow<K> win(lane);
    const int q = lane & 3;
    constexpr int T = 8 * K;                      // dwords a tile
    __amdgpu_buffer_rsrc_t rs[CC];
#pragma unroll
    for (int c = 0; c < CC; ++c) {
        rs[c] = StreamRsrc(w + ((size_t) (b[c] / 8) * ak * 8 + b[c] % 8) * T, (uint32_t) (((ak - 1) * 8 + 1) * T * 4));
    }
    const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
    // the ring runs past the last tile: clamp (a re-read of the last tile, not a read beyond the matrix)
    auto load_tile = [&](const int c, const int tile, uint32_t * dst) {
        const int so = min(tile, ak - 1) * 8 * T * 4;
        dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o0, so, 0);
        dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o1, so, 0);
        dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o2, so, 0);
    };
    uint32_t ring[DEC_PF][CC][3];
    // issue order = refill order (slot by slot), so the loop head needs only vmcnt(3 * CC * (PF - 1))
#pragma unroll
    for (int f = 0; f < DEC_PF; ++f) {
#pragma unroll
        for (int c = 0; c < CC; ++c) {
            load_tile(c, f, ring[f][c]);
        }
        __builtin_amdgcn_sched_barrier(0);
    }
    float acc[CC][R][2];
#pragma unroll
    for (int c = 0; c < CC; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[c][r][0] = acc[c][r][1] = 0.f;
        }
    }
    auto step = [&](const int a, const int f) {
        __half2 P[CC][4];
#pragma unroll
        for (int c = 0; c < CC; ++c) {
            DecodeLane<K, kMulTrue16>(ring[f][c][0], ring[f][c][1], ring[f][c][2], win.shift, P[c]);
            load_tile(c, a + f + DEC_PF, ring[f][c]);   // refill the slot after its decode: the same registers
        }
        __builtin_amdgcn_sched_barrier(0);              // the refills stay here, in slot order
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t * xr = xs + (r * ak + a + f) * 8;
            uint32_t xa = xr[q], xb = xr[4 + q];
            const __half2 ha = *reinterpret_cast<__half2 *>(&xa);
            const __half2 hb = *reinterpret_cast<__half2 *>(&xb);
#pragma unroll
            for (int c = 0; c < CC; ++c) {
                acc[c][r][0] = Dot2(P[c][0], ha, acc[c][r][0]);
                acc[c][r][0] = Dot2(P[c][1], hb, acc[c][r][0]);
                acc[c][r][1] = Dot2(P[c][2], ha, acc[c][r][1]);
                acc[c][r][1] = Dot2(P[c][3], hb, acc[c][r][1]);
            }
        }
    };
    if (ak % DEC_PF == 0) {
        // no early exit inside the unrolled round: each ring slot keeps its registers
        for (int a = 0; a < ak; a += DEC_PF) {
#pragma unroll
            for (int f = 0; f < DEC_PF; ++f) {
                step(a, f);
            }
        }
    } else {
        for (int a = 0; a < ak; a += DEC_PF) {
#pragma unroll
            for (int f = 0; f < DEC_PF; ++f) {
                if (a + f >= ak) {
                    break;
                }
                step(a, f);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < CC; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float v = acc[c][r][h];
                v = v + __shfl_xor(v, 1);
                v = v + __shfl_xor(v, 2);
                acc[c][r][h] = v;
            }
        }
    }
    if (q == 0) {
#pragma unroll
        for (int c = 0; c < CC; ++c) {
            const int col = 16 * b[c] + lane / 4;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                out[r][col]     = acc[c][r][0];
                out[r][col + 8] = acc[c][r][1];
            }
        }
    }
}

// the waves of an item over its column blocks [b0, b0 + nbi): two blocks a wave while there are more than 8
template <int K, int R>
__device__ __forceinline__ void dec_item(const uint32_t * w, const int ak, const int b0, const int nbi, const uint32_t * xs,
        float * const (&out)[R]) {
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    if (wave < nbi) {
        if (wave + 8 < nbi) {
            dec_gemv_cols<K, R, 2>(w, ak, { b0 + wave, b0 + wave + 8 }, xs, out);
        } else {
            dec_gemv_cols<K, R, 1>(w, ak, { b0 + wave }, xs, out);
        }
    }
}

// gate (m 0) or up (m 1) item of job j: the rows' rotated inputs to LDS, the GEMV, then the job's counter
template <int K, int R>
__device__ __forceinline__ void dec_gu(const ggml_cuda_mmh_moe_args & a, const int64_t * tab, const int j, const int e,
        const int32_t * sp, const int m, const int it, uint32_t * xs) {
    const int A = a.T * a.n_used, hd = a.H / 2, nb = a.I / 16;
    const uint32_t * xg = (const uint32_t *) a.xh;
    for (int i = threadIdx.x; i < R * hd; i += blockDim.x) {
        const int r = i / hd, o = i - r * hd;
        xs[i] = xg[(size_t) (m * A + sp[r]) * hd + o];
    }
    __syncthreads();
    float * out[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        out[r] = a.part_gu + (size_t) (m * A + sp[r]) * a.I;
    }
    const int b0 = it * DEC_ITEM;
    dec_item<K, R>((const uint32_t *) tab[m * a.E + e], a.H / 16, b0, min(DEC_ITEM, nb - b0), xs, out);
    __syncthreads();   // every wave's products precede the release
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(a.done + j, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// down item of job j: wait for the job's gate / up workgroups, the SwiGLU step of its rows into LDS (moe_act's F32
// arithmetic: output rotations of gate and up, silu(g) * u, the down input rotation, one fp16 rounding), the GEMV
template <int K, int R>
__device__ __forceinline__ void dec_down(const ggml_cuda_mmh_moe_args & a, const int64_t * tab, const int j, const int e,
        const int32_t * sp, const int it, const unsigned n_gu, uint32_t * xs) {
    const int A = a.T * a.n_used, blocks = a.I / 128, nb = a.H / 16, E = a.E;
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(a.done + j, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n_gu) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    half * xh = (half *) xs;
    for (int t = wave; t < R * blocks; t += blockDim.x / 32) {
        const int r = t / blocks, blk = t % blocks;
        const int c = blk * 128 + 4 * lane;
        const float4 g4 = *(const float4 *) (a.part_gu + (size_t) sp[r] * a.I + c);
        const float4 u4 = *(const float4 *) (a.part_gu + (size_t) (A + sp[r]) * a.I + c);
        float g[4] = { g4.x, g4.y, g4.z, g4.w }, u[4] = { u4.x, u4.y, u4.z, u4.w }, vg[4], vu[4], sd[4];
        moe_load_h4((const half *) tab[4 * E + e] + c, vg);
        moe_load_h4((const half *) tab[6 * E + e] + c, vu);
        moe_load_h4((const half *) tab[7 * E + e] + c, sd);
        moe_had128(g, lane);
        moe_had128(u, lane);
        float h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = g[i] * vg[i];
            h[i] = x / (1.0f + expf(-x)) * (u[i] * vu[i]) * sd[i];
        }
        moe_had128(h, lane);
        moe_store_h4(xh + r * a.I + c, h);
    }
    __syncthreads();
    float * out[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        out[r] = (float *) a.buf_had + (size_t) sp[r] * a.H;
    }
    const int b0 = it * DEC_ITEM;
    dec_item<K, R>((const uint32_t *) tab[2 * E + e], a.I / 16, b0, min(DEC_ITEM, nb - b0), xs, out);
}

template <int K, int R>
__device__ __forceinline__ void dec_block(const ggml_cuda_mmh_moe_args & a, const int64_t * tab, const int j, const int e,
        const int32_t * sp, const bool gu, const int m, const int it, const unsigned n_gu, uint32_t * xs) {
    if (gu) {
        dec_gu<K, R>(a, tab, j, e, sp, m, it, xs);
    } else {
        dec_down<K, R>(a, tab, j, e, sp, it, n_gu, xs);
    }
}

// workgroups [0, A * 2 * ig): gate / up items of jobs 0 .. A - 1 (ig a matrix), then A * id down items; jobs past the
// batch's count exit. Dynamic LDS: T rows of H halves
template <int K>
__global__ void __launch_bounds__(256) moe_dec_fused(const ggml_cuda_mmh_moe_args a, const int64_t * __restrict__ tab) {
    extern __shared__ uint32_t xs[];
    const int A  = a.T * a.n_used;
    const int ig = (a.I / 16 + DEC_ITEM - 1) / DEC_ITEM, id = (a.H / 16 + DEC_ITEM - 1) / DEC_ITEM;
    const int ngu = A * 2 * ig;
    const bool gu = (int) blockIdx.x < ngu;
    int j, m = 0, it;
    if (gu) {
        j = blockIdx.x / (2 * ig);
        const int rem = blockIdx.x % (2 * ig);
        m  = rem / ig;
        it = rem % ig;
    } else {
        j  = (blockIdx.x - ngu) / id;
        it = (blockIdx.x - ngu) % id;
    }
    if (j >= __builtin_amdgcn_readfirstlane(*a.n_jobs)) {
        return;
    }
    const int e    = __builtin_amdgcn_readfirstlane(a.job_e[j]);
    const int rows = __builtin_amdgcn_readfirstlane(a.job_n[j]);
    const int32_t * sp = a.job_s + (size_t) j * a.T;
    const unsigned n_gu = 2 * ig;
    switch (rows) {
        case 1:  dec_block<K, 1>(a, tab, j, e, sp, gu, m, it, n_gu, xs); break;
        case 2:  dec_block<K, 2>(a, tab, j, e, sp, gu, m, it, n_gu, xs); break;
        case 3:  dec_block<K, 3>(a, tab, j, e, sp, gu, m, it, n_gu, xs); break;
        default: dec_block<K, 4>(a, tab, j, e, sp, gu, m, it, n_gu, xs); break;
    }
}
static_assert(GGML_CUDA_MMH_MOE_DEC_MAX_T == 4, "moe_dec_fused has row bodies for 1 .. 4 rows");

// the last workgroup of moe_dec_prep: the per-expert pointer tables, then the picks grouped by expert in pick order
// (jobs in order of their first pick, each job's picks in token order), every pick's own row (inverse = identity),
// zeroed job counters and the ring's item counter
__device__ __forceinline__ void moe_dec_plan(const ggml_cuda_mmh_moe_args & a, int64_t * __restrict__ t) {
    constexpr int MAXA = GGML_CUDA_MMH_MOE_DEC_MAX_T * 16;
    __shared__ int s_e[MAXA], s_n[MAXA], s_s[MAXA * GGML_CUDA_MMH_MOE_DEC_MAX_T], s_jobs;
    const int E = a.E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
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
    const int A = a.T * a.n_used;
    for (int s = threadIdx.x; s < A; s += blockDim.x) {
        a.inverse[s] = s;
        a.pick_e[s]  = moe_id(a.ids, s, a.n_used, a.ids_ld);
        a.done[s]    = 0;
    }
    if (threadIdx.x == 0) {
        if (a.s_w_gate) {
            a.done[A] = 0;   // a shared expert's job: n_jobs + 1 <= A + 1 counters
        }
        *a.next = 0;
    }
    if (threadIdx.x == 0) {
        int n = 0;
        for (int s = 0; s < A; ++s) {
            const int e = moe_id(a.ids, s, a.n_used, a.ids_ld);
            int j = 0;
            while (j < n && s_e[j] != e) {
                ++j;
            }
            if (j == n) {
                s_e[n] = e;
                s_n[n] = 0;
                ++n;
            }
            s_s[j * a.T + s_n[j]++] = s;
        }
        s_jobs = n;
    }
    __syncthreads();
    const int n = s_jobs;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        a.job_e[j] = s_e[j];
        a.job_n[j] = s_n[j];
        for (int r = 0; r < s_n[j]; ++r) {
            a.job_s[j * a.T + r] = s_s[j * a.T + r];
        }
    }
    if (threadIdx.x == 0) {
        *a.n_jobs = n;
    }
}

// the folded shared expert's gate of token t: sigmoid of its weight row times x, with the arithmetic of the node it
// replaces (ggml's mul_mat_vec_f at 256 threads for a row of 2560: thread t takes the element pairs t, t + 256, ... and
// adds w*x for each element of a pair in turn, a wave's sum by an xor butterfly - every lane ends with the same bits -,
// the eight waves' sums zero-padded to 32 and butterflied once more; the unary op's sigmoid). The same bits as the
// gate's own nodes: an in-kernel dot with another order had moved one sensitive chunk of the decode KLD (10-24)
__device__ __forceinline__ void moe_dec_gate(const ggml_cuda_mmh_moe_args & a, const int t) {
    __shared__ float buf[32];
    if (threadIdx.x < 32) {
        buf[threadIdx.x] = 0.0f;
    }
    const float2 * y2 = (const float2 *) (a.x + (int64_t) t * a.x_ld);
    float acc = 0.0f;
    for (int c2 = threadIdx.x; c2 < a.H / 2; c2 += 256) {
        const float2 y = y2[c2];
        float w0, w1;
        if (a.s_gin_bf16) {
            const uint32_t b = ((const uint32_t *) a.s_gin_w)[c2];   // two BF16: the high halves of two F32
            w0 = __uint_as_float(b << 16);
            w1 = __uint_as_float(b & 0xffff0000u);
        } else {
            const float2 w = ((const float2 *) a.s_gin_w)[c2];
            w0 = w.x;
            w1 = w.y;
        }
        acc += w0 * y.x;
        acc += w1 * y.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor(acc, o, 32);
    }
    __syncthreads();
    buf[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = buf[threadIdx.x];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor(v, o, 32);
        }
        if (threadIdx.x == 0) {
            a.s_gate[t] = 1.0f / (1.0f + expf(-v));
        }
    }
}

// one launch before the ring: workgroups [0, G) the gather of moe_gather for a decode batch (pick s keeps row s, its
// expert's rotations straight from the arguments), then a folded shared expert's gate a token (when s_gin_w), then the
// plan
__global__ void __launch_bounds__(256) moe_dec_prep(const ggml_cuda_mmh_moe_args a, int64_t * __restrict__ tab) {
    const int     blocks = a.H / 128;
    const int64_t gjobs  = (int64_t) a.T * blocks;
    const int     G      = (int) ((gjobs + 7) / 8);
    if ((int) blockIdx.x >= G) {
        const int t = (int) blockIdx.x - G;
        if (a.s_gin_w && t < a.T) {
            moe_dec_gate(a, t);
        } else {
            moe_dec_plan(a, tab);
        }
        return;
    }
    const int64_t job = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job >= gjobs) {
        return;
    }
    half * out = (half *) a.xh;
    const int     t   = threadIdx.x & 31;
    const int     blk = job % blocks;
    const int64_t r   = job / blocks;
    const int64_t A   = (int64_t) a.T * a.n_used;
    const float4  xv  = *(const float4 *) (a.x + r * a.x_ld + blk * 128 + 4 * t);
    for (int k = 0; k < a.n_used; ++k) {
        const int64_t s = r * a.n_used + k;
        const int     e = moe_id(a.ids, s, a.n_used, a.ids_ld);
        float sg[4], su[4];
        moe_load_h4((const half *) a.rin_g + e * a.rs_in_g + blk * 128 + 4 * t, sg);
        moe_load_h4((const half *) a.rin_u + e * a.rs_in_u + blk * 128 + 4 * t, su);
        float g[4] = { xv.x * sg[0], xv.y * sg[1], xv.z * sg[2], xv.w * sg[3] };
        float u[4] = { xv.x * su[0], xv.y * su[1], xv.z * su[2], xv.w * su[3] };
        moe_had128(g, t);
        moe_had128(u, t);
        moe_store_h4(out + s * a.H + blk * 128 + 4 * t, g);
        moe_store_h4(out + (A + s) * a.H + blk * 128 + 4 * t, u);
    }
    if (a.s_w_gate) {
        // the shared expert: rows 2A + r (gate) and 2A + T + r (up)
        float sg[4], su[4];
        moe_load_h4((const half *) a.s_rin_g + blk * 128 + 4 * t, sg);
        moe_load_h4((const half *) a.s_rin_u + blk * 128 + 4 * t, su);
        float g[4] = { xv.x * sg[0], xv.y * sg[1], xv.z * sg[2], xv.w * sg[3] };
        float u[4] = { xv.x * su[0], xv.y * su[1], xv.z * su[2], xv.w * su[3] };
        moe_had128(g, t);
        moe_had128(u, t);
        moe_store_h4(out + (2 * A + r) * a.H + blk * 128 + 4 * t, g);
        moe_store_h4(out + (2 * A + a.T + r) * a.H + blk * 128 + 4 * t, u);
    }
}

// ---- the decode-sized batch as a stream ring ----------------------------------------------------------------------------
// strixllama: moe_dec_fused's jobs and items run by Crucible's wave-specialised LDS stream ring instead (vendors/crucible,
// Apache-2.0: LinRingRowsKernel of engine/exl3/ring_rows.h, as mul-mat-had.cu's dense rings): two loader waves stream a
// 128-row group's chunks (4 k tiles x 8 column tiles, contiguous in this fork's layout) with 128-bit buffer loads into a
// 3-slot LDS ring and, for gate / up items, the job's rows of the rotated input behind them; eight compute waves decode
// their column tile and take the products. One k span per item (no split), the accumulation order of dec_gemv_cols, so
// the gate / up products, the SwiGLU rows and the down product are the bits moe_dec_fused makes. Items: every job's
// gate and up groups first, then every job's down groups; a down item waits for its job's gate / up items (every
// workgroup finishes all its gate / up items before its first down item, so the wait cannot deadlock), computes the
// SwiGLU rows into LDS (dec_down's arithmetic) and reads them as its input.
namespace moe_ring {
constexpr int SLOTS = 3, LOADERS = 2, CONSUMERS = 8, CHUNK = 4;
constexpr int ROW_WORDS  = CHUNK * 8;                       // one row's input of a chunk: 4 tiles of 16 halves
constexpr int MAXR = GGML_CUDA_MMH_MOE_DEC_MAX_T;
constexpr int THREADS = 32 * (LOADERS + CONSUMERS);
// a slot of a ring whose widest matrix has KM bits: the weights of a chunk, then MAXR rows of its input
template <int KM> struct slot_of {
    static constexpr int WEIGHTS = CONSUMERS * CHUNK * 8 * KM;
    static constexpr int WORDS   = WEIGHTS + MAXR * ROW_WORDS;
};
typedef crucible::LdsStreamRing<SLOTS, LOADERS, CONSUMERS> control_t;

__device__ __forceinline__ void consumer_barrier(uint32_t * arrived, const uint32_t round, const int lane) {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    if (lane == 0) {
        __hip_atomic_fetch_add(arrived, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    while (__hip_atomic_load(arrived, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < CONSUMERS * round) {
        __builtin_amdgcn_s_sleep(1);
    }
    asm volatile("" ::: "memory");
}

// the item it of a batch of n_jobs jobs: gate / up groups of every job, then down groups. last0: job 0's down groups
// come last (a folded shared expert: its wider gate / up items, taken first, finish after the routed jobs')
struct item_t {
    int j, m, g;      // job, matrix (0 gate, 1 up, 2 down), 128-row group
    bool down;
};
__device__ __forceinline__ item_t item_of(const int it, const int ig, const int id, const int n_jobs, const bool last0 = false) {
    item_t x;
    const int n_gu = n_jobs * 2 * ig;
    if (it < n_gu) {
        x.j = it / (2 * ig);
        const int rem = it % (2 * ig);
        x.m = rem / ig;
        x.g = rem % ig;
        x.down = false;
    } else {
        const int d = it - n_gu;
        x.j = d / id;
        if (last0) {
            x.j = x.j + 1 == n_jobs ? 0 : x.j + 1;
        }
        x.m = 2;
        x.g = d % id;
        x.down = true;
    }
    return x;
}

// the loader waves' part of one item: the 128-row group's chunks (KK bits) into the ring, and for gate / up items the
// job's rows of the rotated input (lane xv stages row xv / 8, words (xv % 8) * 4 of each chunk's 32) behind them. D
// chunks in flight a lane; the chunk offset goes in the vector offset, so that loads past the item's last chunk are
// out of the buffer's range (zeros, no memory traffic) and the loop needs no tail
template <int KK, int KM, int D>
__device__ __forceinline__ void load_item(control_t & control, uint32_t * payload, const uint32_t * base, const int ak,
        const __amdgpu_buffer_rsrc_t xrs, const bool stage_x, const uint32_t xrow, const uint32_t xh_bytes, const int loader,
        const int lane, const uint32_t seq0) {
    using namespace crucible;
    constexpr int T   = 8 * KK;
    constexpr int VPL = KK;                                   // 128-bit vectors a loader lane moves a chunk: 8T / 64
    constexpr int SW  = slot_of<KM>::WORDS;
    const int xv = loader * 32 + lane, xo = (xv & 7) * 4;
    const int count = ak / CHUNK;
    const __amdgpu_buffer_rsrc_t rs = StreamRsrc(base, (uint32_t) ak * 8 * T * 4);
    auto ld = [&](const int c, const int v) -> uint4 {
        const int vi = v * 64 + loader * 32 + lane;
        const auto raw = __builtin_amdgcn_raw_buffer_load_b128(rs, vi * 16 + c * 32 * T * 4, 0, MMH_RING_AUX);
        return *reinterpret_cast<const uint4 *>(&raw);
    };
    auto ldx = [&](const int c) -> uint4 {
        const int vo = stage_x && c < count ? (int) ((xrow + (c * CHUNK) * 8 + xo) * 4) : (int) xh_bytes;
        const auto raw = __builtin_amdgcn_raw_buffer_load_b128(xrs, vo, 0, 0);
        return *reinterpret_cast<const uint4 *>(&raw);
    };
    auto put = [&](const int c, const uint4 (&r)[VPL], const uint4 rx) {
        if (c < count) {
            const uint32_t gs = seq0 + c;
            control.WaitWritable(gs);
            uint32_t * slot = payload + (gs % SLOTS) * SW;
            uint4 * d = reinterpret_cast<uint4 *>(slot) + loader * 32 + lane;
#pragma unroll
            for (int v = 0; v < VPL; ++v) {
                d[64 * v] = r[v];
            }
            if (stage_x) {
                reinterpret_cast<uint4 *>(slot + slot_of<KM>::WEIGHTS)[xv] = rx;
            }
            control.Publish(gs, lane);
        }
    };
    uint4 r[D][VPL], rx[D];
#pragma unroll
    for (int f = 0; f + 1 < D; ++f) {
#pragma unroll
        for (int v = 0; v < VPL; ++v) {
            r[f][v] = ld(f, v);
        }
        rx[f] = ldx(f);
    }
    for (int c = 0; c < count; c += D) {
#pragma unroll
        for (int f = 0; f < D; ++f) {
            // the chunk D - 1 ahead into the buffer the last put freed, then this chunk into the ring
            const int nb = (f + D - 1) % D;
#pragma unroll
            for (int v = 0; v < VPL; ++v) {
                r[nb][v] = ld(c + f + D - 1, v);
            }
            rx[nb] = ldx(c + f + D - 1);
            put(c + f, r[f], rx[f]);
        }
    }
}

// the compute waves' products of one item: R rows, x from the slot (gate / up) or from the SwiGLU rows in LDS (down)
template <int K, int R, int KM>
__device__ __forceinline__ void consume(const ggml_cuda_mmh_moe_args & a, control_t & control, const uint32_t * payload,
        const item_t & x, const int count, const uint32_t seq0, const uint32_t * xact, const int cw, const int lane,
        float * const * out) {
    using namespace crucible;
    constexpr int T  = 8 * K;
    constexpr int SW = slot_of<KM>::WORDS;
    const int q = lane & 3;
    const LaneWindow<K> win(lane);
    const int act_row = a.I / 2;                         // words of one SwiGLU row
    float acc0[R], acc1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc0[r] = acc1[r] = 0.f;
    }
    for (int c = 0; c < count; ++c) {
        const uint32_t gs = seq0 + c;
        control.WaitReady(gs);
        const uint32_t * slot = payload + (gs % SLOTS) * SW;
#pragma unroll
        for (int f = 0; f < CHUNK; ++f) {
            const uint32_t * tile = slot + (f * 8 + cw) * T;
            __half2 P[4];
            DecodeLane<K, kMulTrue16>(tile[win.d0], tile[(win.d0 + 1) % T], tile[(win.d0 + 2) % T], win.shift, P);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t * xw = x.down ? xact + r * act_row + (c * CHUNK + f) * 8 : slot + slot_of<KM>::WEIGHTS + r * ROW_WORDS + f * 8;
                const uint32_t xa = xw[q], xb = xw[4 + q];
                const __half2 ha = *reinterpret_cast<const __half2 *>(&xa), hb = *reinterpret_cast<const __half2 *>(&xb);
                acc0[r] = Dot2(P[0], ha, acc0[r]);
                acc0[r] = Dot2(P[1], hb, acc0[r]);
                acc1[r] = Dot2(P[2], ha, acc1[r]);
                acc1[r] = Dot2(P[3], hb, acc1[r]);
            }
        }
        control.Release(gs, lane);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float v = h ? acc1[r] : acc0[r];
            v = v + __shfl_xor(v, 1);
            v = v + __shfl_xor(v, 2);
            (h ? acc1[r] : acc0[r]) = v;
        }
    }
    if (q == 0) {
        const int col = 16 * (x.g * 8 + cw) + lane / 4;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            out[r][col]     = acc0[r];
            out[r][col + 8] = acc1[r];
        }
    }
}

template <int K, int KM>
__device__ __forceinline__ void consume_rows(const int rows, const ggml_cuda_mmh_moe_args & a, control_t & control,
        const uint32_t * payload, const item_t & x, const int count, const uint32_t seq0, const uint32_t * xact, const int cw,
        const int lane, float * const * out) {
    switch (rows) {
        case 1:  consume<K, 1, KM>(a, control, payload, x, count, seq0, xact, cw, lane, out); break;
        case 2:  consume<K, 2, KM>(a, control, payload, x, count, seq0, xact, cw, lane, out); break;
        case 3:  consume<K, 3, KM>(a, control, payload, x, count, seq0, xact, cw, lane, out); break;
        default: consume<K, 4, KM>(a, control, payload, x, count, seq0, xact, cw, lane, out); break;
    }
}

// dec_down's SwiGLU step of the job's rows into LDS, one wave a (row, 128-block): gate rows gi[r] and up rows ui[r] of
// part_gu, the output rotations vg / vu of gate and up and the down input rotation sd
__device__ __forceinline__ void act_rows(const ggml_cuda_mmh_moe_args & a, const half * vgp, const half * vup, const half * sdp,
        const int (&gi)[MAXR], const int (&ui)[MAXR], const int rows, uint32_t * xact, const int cw, const int lane) {
    const int blocks = a.I / 128;
    half * xh = (half *) xact;
    for (int t = cw; t < rows * blocks; t += CONSUMERS) {
        const int r = t / blocks, blk = t % blocks;
        const int c = blk * 128 + 4 * lane;
        const float4 g4 = *(const float4 *) (a.part_gu + (size_t) gi[r] * a.I + c);
        const float4 u4 = *(const float4 *) (a.part_gu + (size_t) ui[r] * a.I + c);
        float g[4] = { g4.x, g4.y, g4.z, g4.w }, u[4] = { u4.x, u4.y, u4.z, u4.w }, vg[4], vu[4], sd[4];
        moe_load_h4(vgp + c, vg);
        moe_load_h4(vup + c, vu);
        moe_load_h4(sdp + c, sd);
        moe_had128(g, lane);
        moe_had128(u, lane);
        float h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = g[i] * vg[i];
            h[i] = x / (1.0f + expf(-x)) * (u[i] * vu[i]) * sd[i];
        }
        moe_had128(h, lane);
        moe_store_h4(xh + r * a.I + c, h);
    }
}

// K: the routed experts' width; KS: the shared expert's (job 0, every token), 0 = none folded in. Items are taken
// from a counter as workgroups come free (heaviest first: the shared expert's gate / up groups, every routed job's,
// then the routed jobs' down groups and the shared expert's last), so the launch ends within about one down item of its byte share; a
// static round-robin left the busiest workgroups 1.4x the mean bytes. Loader wave 0 takes the next item's number
// while its current one streams and hands it to the other waves through a 4-entry queue in LDS (an entry is reused 4
// items later, when every wave has read it: the ring holds 3 chunks, an item at least one)
template <int K, int KS, int D>
__global__ void __launch_bounds__(THREADS) moe_dec_ring(const ggml_cuda_mmh_moe_args a, const int64_t * __restrict__ tab) {
    using namespace crucible;
    constexpr int KM  = KS > K ? KS : K;
    constexpr int KSx = KS ? KS : K;                          // a valid width for the dead shared branch of KS = 0
    constexpr int J0  = KS ? 1 : 0;                           // the first routed job
    __shared__ control_t control;
    __shared__ uint32_t payload[SLOTS * slot_of<KM>::WORDS];
    extern __shared__ uint32_t xact_dyn[];                   // a down item's SwiGLU rows: MAXR rows of I halves
    __shared__ uint32_t arrived, q_seq;
    __shared__ int q_item[4];
    const int lane = threadIdx.x & 31;
    const int A  = a.T * a.n_used, E = a.E;
    const int ig = a.I / 128, id = a.H / 128;
    const int n_jobs  = __builtin_amdgcn_readfirstlane(*a.n_jobs);
    const int n_all   = n_jobs + J0;
    const int n_items = n_all * (2 * ig + id);
    const int ak_gu = a.H / 16, ak_d = a.I / 16;             // k tiles of a gate / up matrix, of the down matrix
    control.Initialize(threadIdx.x);
    if (threadIdx.x == 0) {
        arrived = 0;
        q_seq   = 0;
    }
    __syncthreads();
    // the k-th item of this workgroup, as loader wave 0 queued it
    auto queued = [&](const uint32_t k) -> int {
        while (__hip_atomic_load(&q_seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) <= k) {
            __builtin_amdgcn_s_sleep(1);
        }
        asm volatile("" ::: "memory");
        return __builtin_amdgcn_readfirstlane(q_item[k & 3]);
    };
    if (threadIdx.x < 32 * LOADERS) {
        const int loader = threadIdx.x >> 5;
        const int xr = (loader * 32 + lane) >> 3;
        const uint32_t xh_bytes = (uint32_t) ((2 * A + (KS ? 2 * a.T : 0)) * a.H * 2);
        const __amdgpu_buffer_rsrc_t xrs = StreamRsrc((const uint32_t *) a.xh, xh_bytes);
        uint32_t seq0 = 0;
        int nxt = 0;
        if (loader == 0 && lane == 0) {
            nxt = (int) __hip_atomic_fetch_add(a.next, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        for (uint32_t k = 0; ; ++k) {
            int it;
            if (loader == 0) {
                it = __builtin_amdgcn_readfirstlane(nxt);
                if (lane == 0) {
                    q_item[k & 3] = it;
                }
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                if (lane == 0) {
                    __hip_atomic_store(&q_seq, k + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                    if (it < n_items) {
                        nxt = (int) __hip_atomic_fetch_add(a.next, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    }
                }
            } else {
                it = queued(k);
            }
            if (it >= n_items) {
                break;
            }
            const item_t x = item_of(it, ig, id, n_all, KS != 0);
            const int ak = x.down ? ak_d : ak_gu;
            if (KS && x.j == 0) {
                const void * w = x.m == 0 ? a.s_w_gate : x.m == 1 ? a.s_w_up : a.s_w_down;
                const bool stage_x = !x.down && xr < a.T;
                const uint32_t xrow = stage_x ? (uint32_t) ((2 * A + x.m * a.T + xr) * (a.H / 2)) : 0u;
                load_item<KSx, KM, D>(control, payload, (const uint32_t *) w + (size_t) x.g * ak * 8 * (8 * KSx), ak, xrs, stage_x,
                                   xrow, xh_bytes, loader, lane, seq0);
            } else {
                const int jr   = x.j - J0;
                const int e    = __builtin_amdgcn_readfirstlane(a.job_e[jr]);
                const int rows = __builtin_amdgcn_readfirstlane(a.job_n[jr]);
                const int32_t * sp = a.job_s + (size_t) jr * a.T;
                const bool stage_x = !x.down && xr < rows;
                const uint32_t xrow = stage_x ? (uint32_t) ((x.m * A + sp[xr]) * (a.H / 2)) : 0u;
                load_item<K, KM, D>(control, payload, (const uint32_t *) tab[x.m * E + e] + (size_t) x.g * ak * 8 * (8 * K), ak, xrs,
                                 stage_x, xrow, xh_bytes, loader, lane, seq0);
            }
            seq0 += ak / CHUNK;
        }
        return;
    }
    const int cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - LOADERS);
    uint32_t round = 0, seq0 = 0;
    for (uint32_t k = 0; ; ++k) {
        const int it = queued(k);
        if (it >= n_items) {
            break;
        }
        const item_t x = item_of(it, ig, id, n_all, KS != 0);
        const bool sh  = KS && x.j == 0;
        const int jr   = x.j - J0;
        const int rows = sh ? a.T : __builtin_amdgcn_readfirstlane(a.job_n[jr]);
        const int e    = sh ? 0 : __builtin_amdgcn_readfirstlane(a.job_e[jr]);
        const int count = (x.down ? ak_d : ak_gu) / CHUNK;
        int gi[MAXR], ui[MAXR];   // the rows' gate and up rows of xh / part_gu
#pragma unroll
        for (int r = 0; r < MAXR; ++r) {
            const int rr = r < rows ? r : 0;
            if (sh) {
                gi[r] = 2 * A + rr;
                ui[r] = 2 * A + a.T + rr;
            } else {
                const int s = a.job_s[(size_t) jr * a.T + rr];
                gi[r] = s;
                ui[r] = A + s;
            }
        }
        if (x.down) {
            // the job's gate / up items first (counter x.j), then its SwiGLU rows (every compute wave reads them)
            if (cw == 0 && lane == 0) {
                while (__hip_atomic_load(a.done + x.j, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned) (2 * ig)) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            consumer_barrier(&arrived, ++round, lane);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if (sh) {
                act_rows(a, (const half *) a.s_rout_g, (const half *) a.s_rout_u, (const half *) a.s_rin_d, gi, ui, rows, xact_dyn,
                         cw, lane);
            } else {
                act_rows(a, (const half *) tab[4 * E + e], (const half *) tab[6 * E + e], (const half *) tab[7 * E + e], gi, ui,
                         rows, xact_dyn, cw, lane);
            }
            consumer_barrier(&arrived, ++round, lane);
        }
        float * out[MAXR];
#pragma unroll
        for (int r = 0; r < MAXR; ++r) {
            const int rr = r < rows ? r : 0;
            if (x.down) {
                out[r] = sh ? a.s_raw + (size_t) rr * a.H : (float *) a.buf_had + (size_t) gi[r] * a.H;
            } else {
                out[r] = a.part_gu + (size_t) (x.m == 0 ? gi[r] : ui[r]) * a.I;
            }
        }
        if (sh) {
            consume_rows<KSx, KM>(rows, a, control, payload, x, count, seq0, xact_dyn, cw, lane, out);
        } else {
            consume_rows<K, KM>(rows, a, control, payload, x, count, seq0, xact_dyn, cw, lane, out);
        }
        seq0 += count;
        if (!x.down) {
            // every wave's products precede the job counter's release
            __threadfence();
            consumer_barrier(&arrived, ++round, lane);
            if (cw == 0 && lane == 0) {
                __hip_atomic_fetch_add(a.done + x.j, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        } else {
            // the next item's SwiGLU rows may not overwrite xact before every wave is done reading it
            consumer_barrier(&arrived, ++round, lane);
        }
    }
}
} // namespace moe_ring

struct dec_layout {
    size_t inverse, pick_e, tables, job_e, job_n, job_s, n_jobs, done, xh, part, s_raw, next, s_gate, total;
};

dec_layout dec_layout_of(const int E, const int H, const int I) {
    const int64_t A = (int64_t) GGML_CUDA_MMH_MOE_DEC_MAX_T * 16;
    dec_layout l;
    size_t p = 0;
    auto take = [&p](const int64_t bytes) { const size_t q = p; p += align256(bytes); return q; };
    l.inverse = take(8 * A);
    l.pick_e  = take(4 * A);
    l.tables  = take(8 * 9 * (int64_t) E);
    l.job_e   = take(4 * A);
    l.job_n   = take(4 * A);
    l.job_s   = take(4 * A * GGML_CUDA_MMH_MOE_DEC_MAX_T);
    l.n_jobs  = take(4);
    l.done    = take(4 * A);
    l.xh      = take(2 * A * H * 2);
    l.part    = take(2 * A * I * 4);
    l.s_raw   = take((int64_t) GGML_CUDA_MMH_MOE_DEC_MAX_T * H * 4);
    l.next    = take(4);
    l.s_gate  = take(4 * GGML_CUDA_MMH_MOE_DEC_MAX_T);
    l.total   = p;
    return l;
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

    // DEV: STRIX_MMH_MOE_PROF=1 - stage times of the prompt pipeline (events, synchronous; not under graph capture),
    // summed over every call and printed every 48 calls
    static const bool prof = getenv("STRIX_MMH_MOE_PROF") && atoi(getenv("STRIX_MMH_MOE_PROF")) != 0;
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if (prof) {
        (void) hipStreamIsCapturing(st, &cap);
    }
    const bool timed = prof && cap == hipStreamCaptureStatusNone;
    hipEvent_t ev[6] = {};
    auto mark = [&](const int k) {
        if (timed) {
            if (!ev[k]) { (void) hipEventCreate(&ev[k]); }
            (void) hipEventRecord(ev[k], st);
        }
    };
    mark(0);
    moe_tables<<<(E + 255) / 256, 256, 0, st>>>(a, a.tables);
    moe_block_count<<<(int) s.nb, SORT_BLOCK, E * sizeof(int), st>>>(a.ids, a.n_used, a.ids_ld, A, E, a.block_counts);
    moe_block_scan<<<(E + 1 + 255) / 256, 256, 0, st>>>(a.block_counts, (int) s.nb, E, a.counts);
    mpw_metadata_kernel<<<1, 256, 0, st>>>(a.counts, nullptr, a.offsets, nullptr, a.tiles, a.num_tiles, E, A, MT);
    moe_place<<<(int) s.nb, SORT_BLOCK, 0, st>>>(a.ids, a.n_used, a.ids_ld, A, E, a.block_counts, a.offsets, a.inverse, a.pick_e,
                                                  a.sorted_e);

    half * had = (half *) a.buf_had;
    half * gu  = (half *) a.buf_gu;
    const int64_t gjobs = (int64_t) a.T * (a.H / 128);
    mark(1);
    moe_gather<<<(int) ((gjobs + 7) / 8), 256, 0, st>>>(a, a.tables, had);
    mark(2);

    const int64_t * tg = a.tables, * tu = a.tables + E, * td = a.tables + 2 * E;
    moe_gemm_bits<false>(a.bits_gu, dim3((a.I + 255) / 256, (unsigned) s.slots, 2), st, had, (int64_t) A * a.H, a.offsets,
                         a.tiles, a.num_tiles, tg, tu, gu, (int64_t) A * a.I, a.H, a.I);
    mark(3);

    const int64_t ajobs = (int64_t) A * (a.I / 128);
    moe_act<<<(int) ((ajobs + 7) / 8), 256, 0, st>>>(a, a.tables, gu);
    mark(4);

    float * down = (float *) a.buf_had;   // the gate / up inputs are dead: the down product takes their bytes
    moe_gemm_bits<true>(a.bits_d, dim3((a.H + 255) / 256, (unsigned) s.slots, 1), st, gu, 0, a.offsets, a.tiles, a.num_tiles,
                        td, td, down, 0, a.I, a.H);
    mark(5);
    if (timed) {
        static double acc[5] = {};
        static int calls = 0;
        (void) hipEventSynchronize(ev[5]);
        for (int k = 0; k < 5; ++k) {
            float ms = 0.f;
            (void) hipEventElapsedTime(&ms, ev[k], ev[k + 1]);
            acc[k] += ms;
        }
        if (++calls % 48 == 0) {
            fprintf(stderr, "MMH_MOE_PROF T=%d, ms a call over %d calls: sort %.2f gather %.2f gate/up %.2f act %.2f down %.2f\n", a.T,
                    calls, acc[0] / calls, acc[1] / calls, acc[2] / calls, acc[3] / calls, acc[4] / calls);
        }
    }
}

void ggml_cuda_mmh_moe_run_reduce(const ggml_cuda_mmh_moe_args & a0, void * stream) {
    // DEV: STRIX_MMH_MOE_REDUCE_BATCH=1 the batched loads, 0 a pick at a time, -1 the STRIX_AB side
    static const int env = getenv("STRIX_MMH_MOE_REDUCE_BATCH") ? atoi(getenv("STRIX_MMH_MOE_REDUCE_BATCH")) : 1;
    ggml_cuda_mmh_moe_args a = a0;
    a.batch_loads = env < 0 ? ggml_cuda_strix_ab() != 0 : env != 0;
    const int64_t rjobs = (int64_t) a.T * (a.H / 128);
    moe_reduce<<<(int) ((rjobs + 7) / 8), 256, 0, (hipStream_t) stream>>>(a, a.tables, (const float *) a.buf_had);
}

size_t ggml_cuda_mmh_moe_dec_scratch_bytes(const int E, const int H, const int I) {
    return dec_layout_of(E, H, I).total;
}

void ggml_cuda_mmh_moe_dec_carve(ggml_cuda_mmh_moe_args & a, void * scratch) {
    const dec_layout l = dec_layout_of(a.E, a.H, a.I);
    char * p = (char *) scratch;
    a.inverse = (int64_t *)  (p + l.inverse);
    a.pick_e  = (int32_t *)  (p + l.pick_e);
    a.tables  = (int64_t *)  (p + l.tables);
    a.job_e   = (int32_t *)  (p + l.job_e);
    a.job_n   = (int32_t *)  (p + l.job_n);
    a.job_s   = (int32_t *)  (p + l.job_s);
    a.n_jobs  = (int32_t *)  (p + l.n_jobs);
    a.done    = (uint32_t *) (p + l.done);
    a.xh      = (void *)     (p + l.xh);
    a.part_gu = (float *)    (p + l.part);
    a.s_raw   = (float *)    (p + l.s_raw);
    a.next    = (uint32_t *) (p + l.next);
    a.s_gate  = (float *)    (p + l.s_gate);
}

static bool ggml_cuda_mmh_moe_ring_on() {
    static const bool ring = !getenv("STRIX_MMH_MOE_RING") || atoi(getenv("STRIX_MMH_MOE_RING")) != 0;
    return ring;
}

bool ggml_cuda_mmh_moe_dec_shexp_ok(const int T, const int n_used, const int I, const int bits_gu, const int I_s, const int bits_s) {
    // the ring's instances (4.05: 4-bit experts, a 6-bit shared expert; 3.05: 3 and 5), its rows after the picks' in
    // the fixed scratch (T * n_used + T of 64), and an expert's shape
    const bool pair = (bits_gu == 4 && bits_s == 6) || (bits_gu == 3 && bits_s == 5);
    return ggml_cuda_mmh_moe_ring_on() && pair && I_s == I && T * n_used + T <= GGML_CUDA_MMH_MOE_DEC_MAX_T * 16;
}

void ggml_cuda_mmh_moe_run_dec(const ggml_cuda_mmh_moe_args & a, void * stream) {
    hipStream_t st = (hipStream_t) stream;
    const int A = a.T * a.n_used;
    const int64_t gjobs = (int64_t) a.T * (a.H / 128);
    moe_dec_prep<<<(int) ((gjobs + 7) / 8) + (a.s_gin_w ? a.T : 0) + 1, 256, 0, st>>>(a, a.tables);
    // the jobs by the stream ring (moe_ring::moe_dec_ring; STRIX_MMH_MOE_RING=0: the fused GEMV launch, the same bits):
    // 4.05 MTP verification 16.5 -> 15.1 ms of 40 a step, one-token decode +2%. A folded shared expert is one more job.
    // Items from a counter: one token's experts 8.07 -> 7.67 ms a token (the round-robin left the busiest workgroups
    // 1.4x the mean bytes); 64 workgroups for one token (-0.29 ms against 80), 80 for 2-4 rows (-0.7% against 64)
    if (ggml_cuda_mmh_moe_ring_on()) {
        const int n_items_max = (A + (a.s_w_gate ? 1 : 0)) * (2 * (a.I / 128) + a.H / 128);
        // DEV: STRIX_MMH_MOE_GRID=a[,b] workgroups (b: side B of the in-run A/B, STRIX_AB)
        static int grid_ab[2] = { 0, 0 };   // 0: 64 for one token, 80 for 2-4 (in-run A/Bs, 4.05)
        static const bool grid_env = [] {
            if (const char * g = getenv("STRIX_MMH_MOE_GRID")) {
                grid_ab[0] = grid_ab[1] = atoi(g);
                if (const char * c = strchr(g, ',')) { grid_ab[1] = atoi(c + 1); }
            }
            return true;
        }();
        (void) grid_env;
        const int      g_env = grid_ab[ggml_cuda_strix_ab() ? 1 : 0];
        const unsigned grid  = (unsigned) std::min(g_env > 0 ? g_env : a.T == 1 ? 64 : 80, n_items_max);
        const size_t   lds  = (size_t) moe_ring::MAXR * a.I * 2;     // the SwiGLU rows of a down item
        const int      ks   = a.s_w_gate ? a.bits_s : 0;
        // two chunks in flight a loader lane (three measured the same at one token, 4.05)
#define MOE_RING_LAUNCH(K_, KS_) moe_ring::moe_dec_ring<K_, KS_, 2><<<grid, moe_ring::THREADS, lds, st>>>(a, a.tables)
        if (a.bits_gu == 4 && ks == 6) {
            MOE_RING_LAUNCH(4, 6);
        } else if (a.bits_gu == 3 && ks == 5) {
            MOE_RING_LAUNCH(3, 5);
        } else if (a.bits_gu == 4) {
            MOE_CHECK(ks == 0);
            MOE_RING_LAUNCH(4, 0);
        } else {
            MOE_CHECK(ks == 0);
            MOE_RING_LAUNCH(3, 0);
        }
#undef MOE_RING_LAUNCH
        return;
    }
    MOE_CHECK(!a.s_w_gate && "the fused GEMV launch takes no shared expert");
    const int ig = (a.I / 16 + DEC_ITEM - 1) / DEC_ITEM, id = (a.H / 16 + DEC_ITEM - 1) / DEC_ITEM;
    const unsigned grid = (unsigned) (A * 2 * ig + A * id);
    const size_t   lds  = (size_t) a.T * a.H * 2;
    if (a.bits_gu == 4) {
        moe_dec_fused<4><<<grid, 256, lds, st>>>(a, a.tables);
    } else {
        moe_dec_fused<3><<<grid, 256, lds, st>>>(a, a.tables);
    }
}

#endif // defined(GGML_USE_HIP)
