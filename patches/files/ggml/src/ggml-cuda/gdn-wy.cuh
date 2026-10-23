// SPDX-License-Identifier: Apache-2.0
//
// Gated delta net prefill in the chunked (WY) form with 16-row chunks on gfx11 WMMA.
//
// Adapted from hipfire (https://github.com/warpfront/hipfire, v0.4.2.1-13-gbfb90b9,
// kernels/src/gated_delta_chunk_wmma.gfx1151.hip), Copyright the hipfire authors, licensed under the Apache License 2.0
// (licenses/LICENSE-hipfire-APACHE, notice in licenses/NOTICE-hipfire), by way of exllamav3's KYO_RAW build in Kyojin
// (MIT; its change: T = (I - L)^-1 by fp32 forward substitution instead of the f16 product form).
//
// strixllama changes: reads the op's F32 q / k / v / g / beta through their strides and maps value head h to key head
// h % H_k as the op does; takes the initial state from the op's transposed rows (row j = column j of S) or a cache
// row, and writes the states the same way; writes the attention output in F32 (no gated-norm epilogue, no BF16
// rounding); with snapshots (keep_rs) the last K - 1 tokens run one at a time after the chunks so each leaves its
// state. All products are F16 WMMA with F32 accumulation (q, k, the decayed operands, T, the state as an operand and
// Vnew round to F16), so this path is not bitwise equal to the per-token F32 kernels: STRIX_GDN_WY=0 turns it off.
#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)
#define GDN_WY_AVAILABLE 1

typedef _Float16 __attribute__((ext_vector_type(16))) gdn_wy_h16;
typedef float    __attribute__((ext_vector_type(8)))  gdn_wy_f8;
typedef unsigned int __attribute__((ext_vector_type(8))) gdn_wy_u8;
typedef _Float16 __attribute__((ext_vector_type(2)))  gdn_wy_h2;

// __syncthreads also waits for every outstanding global access, which would drain the prefetch
#define GDN_WY_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
// wave-private LDS handoff: this wave's stores before its loads
#define GDN_WY_WAVE_SYNC() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); \
                                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront"); } while (0)
#define GDN_WY_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)

#if defined(RDNA3)
static __device__ __forceinline__ unsigned int gdn_wy_pack(float a, float b) {
    return __builtin_bit_cast(unsigned int, gdn_wy_h2{(_Float16) a, (_Float16) b});
}

// C-layout tile (lane: column l16, rows 2r + hi) -> the B operand of that matrix (lane: column l16, all 16 rows), by
// swapping half-waves
static __device__ __forceinline__ gdn_wy_h16 gdn_wy_c_to_b(const gdn_wy_f8 c, int hi) {
    gdn_wy_u8 out;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const unsigned int own = gdn_wy_pack(c[2 * p], c[2 * p + 1]);
        const unsigned int oth = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
        const unsigned int a = hi ? oth : own, b = hi ? own : oth;
        out[2 * p]     = __builtin_amdgcn_perm(b, a, 0x05040100u);
        out[2 * p + 1] = __builtin_amdgcn_perm(b, a, 0x07060302u);
    }
    return __builtin_bit_cast(gdn_wy_h16, out);
}

// The state in registers: wave w holds columns 16 w + l16, tile t rows 16 t + 2 r + hi (r = 0..7). In memory (the op's
// layout) column j is the contiguous row j: element S[i][j] at j * 128 + i. A lane moves rows 16 t + 8 hi .. + 7 of its
// column as two float4s; the other parity comes from the partner half-wave.
static __device__ __forceinline__ void gdn_wy_load_state(gdn_wy_f8 S[8], const float * src, int col, int hi) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float4 a = *(const float4 *) (src + (size_t) col * 128 + 16 * t + 8 * hi);
        const float4 b = *(const float4 *) (src + (size_t) col * 128 + 16 * t + 8 * hi + 4);
        const float m[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };   // rows 16 t + 8 hi + m
        // own parity rows 2 r + hi for r = 4 hi .. 4 hi + 3 are m = 2 (r - 4 hi) + hi; the partner wants the other parity
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float mine = m[2 * q + hi];          // row 16 t + 8 hi + 2 q + hi = 2 (4 hi + q) + hi
            const float give = m[2 * q + 1 - hi];      // row 2 (4 hi + q) + (1 - hi): the partner's parity, its r = 4 hi + q
            const float got  = __shfl_xor(give, 16, 32);   // the partner's rows 2 (4 (1 - hi) + q) + hi
            S[t][4 * hi + q]       = mine;
            S[t][4 * (1 - hi) + q] = got;
        }
    }
}

static __device__ __forceinline__ void gdn_wy_store_state(const gdn_wy_f8 S[8], float * dst, int col, int hi) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        float m[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float give = S[t][4 * (1 - hi) + q];     // rows 2 (4 (1 - hi) + q) + hi: the partner writes them
            const float got  = __shfl_xor(give, 16, 32);   // the partner's rows 2 (4 hi + q) + (1 - hi)
            m[2 * q + hi]     = S[t][4 * hi + q];
            m[2 * q + 1 - hi] = got;
        }
        *(float4 *) (dst + (size_t) col * 128 + 16 * t + 8 * hi)     = make_float4(m[0], m[1], m[2], m[3]);
        *(float4 *) (dst + (size_t) col * 128 + 16 * t + 8 * hi + 4) = make_float4(m[4], m[5], m[6], m[7]);
    }
}
#endif // defined(RDNA3)

// grid [value heads], block 256 (8 waves x 16 value columns); one sequence (n_seqs == 1)
template <bool keep_rs>
__launch_bounds__(256)
static __global__ void gated_delta_net_wy_cuda(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
        const int H, const int n_tokens, const int64_t sq1, const int64_t sq2, const int64_t sv1, const int64_t sv2,
        const int64_t sb1, const int64_t sb2, const int neqk1, const float scale, const int64_t state_slot_stride,
        const int K, const int32_t * __restrict__ state_rows) {
#if defined(RDNA3)
    const int vh = blockIdx.x;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int kh = vh % neqk1;
    const int dv0 = wave * 16, col = dv0 + l16;
    const int64_t SS = 128 * 128;

    __shared__ __align__(16) _Float16 k_s[16][136];    // k rows
    __shared__ __align__(16) _Float16 q_s[16][136];    // scaled q rows
    __shared__ __align__(16) _Float16 kb_s[16][136];   // k beta e^G rows
    __shared__ __align__(16) _Float16 kdT[128][16];    // (k e^{GL - G})^T
    __shared__ float vb_s[16][132];                    // v beta rows (F32)
    __shared__ __align__(16) _Float16 t_s[16][16];
    __shared__ float lf_s[16][17];                     // L (F32) for the substitution
    __shared__ __align__(16) _Float16 p_s[16][16];
    __shared__ float g_s[16], bt_s[16];

    const float * s_in = curr_state + (state_rows ? (int64_t) state_rows[0] : 0) * H * SS + (int64_t) vh * SS;
    float * s_out = state + (int64_t) vh * SS;
    float * out   = dst + (int64_t) vh * 128;   // token t at t * 128 * H

    gdn_wy_f8 S[8];
    gdn_wy_load_state(S, s_in, col, hi);

    // the chunks cover the tokens before the last K - 1, which (keep_rs) then run one at a time for their snapshots
    const int n_chunked = keep_rs ? max(0, n_tokens - (K - 1)) : n_tokens;

    const int si = tid & 15, sd = (tid >> 4) * 8;
    float4 pk0, pk1, pq0, pq1, pv0, pv1;
    float pg = 0.f, pb = 0.f;
    auto fetch = [&](int c0) {
        const int nn = min(16, n_chunked - c0);
        const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
        pk0 = pk1 = pq0 = pq1 = pv0 = pv1 = z;
        if (si < nn) {
            const int64_t qo = (int64_t) (c0 + si) * sq2 + (int64_t) kh * sq1 + sd;
            const int64_t vo = (int64_t) (c0 + si) * sv2 + (int64_t) vh * sv1 + sd;
            pk0 = *(const float4 *) (k + qo); pk1 = *(const float4 *) (k + qo + 4);
            pq0 = *(const float4 *) (q + qo); pq1 = *(const float4 *) (q + qo + 4);
            pv0 = *(const float4 *) (v + vo); pv1 = *(const float4 *) (v + vo + 4);
        }
        if (tid < 16) {
            pg = tid < nn ? g[(int64_t) (c0 + tid) * sb2 + (int64_t) vh * sb1] : 0.f;
            pb = tid < nn ? beta[(int64_t) (c0 + tid) * sb2 + (int64_t) vh * sb1] : 0.f;
        }
    };
    if (n_chunked > 0) {
        fetch(0);
    }
    for (int t0 = 0; t0 < n_chunked; t0 += 16) {
        const float kf[8] = { pk0.x, pk0.y, pk0.z, pk0.w, pk1.x, pk1.y, pk1.z, pk1.w };
        const float qf[8] = { pq0.x, pq0.y, pq0.z, pq0.w, pq1.x, pq1.y, pq1.z, pq1.w };
        const float vf[8] = { pv0.x, pv0.y, pv0.z, pv0.w, pv1.x, pv1.y, pv1.z, pv1.w };
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            *(unsigned int *) &k_s[si][sd + 2 * h] = gdn_wy_pack(kf[2 * h], kf[2 * h + 1]);
            *(unsigned int *) &q_s[si][sd + 2 * h] = gdn_wy_pack(qf[2 * h] * scale, qf[2 * h + 1] * scale);
        }
        if (tid < 16) {
            float G = pg;   // chunk-local cumulative log decay
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) {
                const float up = __shfl_up(G, o, 16);
                if (tid >= o) {
                    G += up;
                }
            }
            g_s[tid]  = G;
            bt_s[tid] = pb;
        }
        GDN_WY_BARRIER();
        {
            const float b = bt_s[si], eg = expf(g_s[si]), ed = expf(g_s[15] - g_s[si]);
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                vb_s[si][sd + h]  = vf[h] * b;
                kdT[sd + h][si] = (_Float16) (kf[h] * ed);
            }
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                *(unsigned int *) &kb_s[si][sd + 2 * h] = gdn_wy_pack(kf[2 * h] * b * eg, kf[2 * h + 1] * b * eg);
            }
        }
        if (t0 + 16 < n_chunked) {
            fetch(t0 + 16);
        }
        GDN_WY_BARRIER();
        if (wave == 0) {
            gdn_wy_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                c = GDN_WY_WMMA(*(const gdn_wy_h16 *) &k_s[l16][16 * s], *(const gdn_wy_h16 *) &k_s[l16][16 * s], c);
            }
            // L strictly lower; T = (I - L)^-1 by forward substitution in F32, rounded to F16 once
            gdn_wy_f8 P;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                P[r] = j < i ? -bt_s[i] * c[r] * expf(g_s[i] - g_s[j]) : 0.f;
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                lf_s[2 * r + hi][l16] = P[r];
            }
            GDN_WY_WAVE_SYNC();
            if (hi == 0) {   // lane l16 = column c: T[i][c] = (i == c) + sum_{j < i} L[i][j] T[j][c]
                float tc[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float a = i == l16 ? 1.f : 0.f;
#pragma unroll
                    for (int j = 0; j < i; ++j) {
                        a = fmaf(lf_s[i][j], tc[j], a);
                    }
                    asm volatile("" : "+v"(a));   // a materialized in F32: no fma -> f16 fusion at the store
                    tc[i] = a;
                    t_s[i][l16] = (_Float16) a;
                }
            }
        } else if (wave == 1) {
            gdn_wy_f8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                c = GDN_WY_WMMA(*(const gdn_wy_h16 *) &q_s[l16][16 * s], *(const gdn_wy_h16 *) &k_s[l16][16 * s], c);
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int i = 2 * r + hi, j = l16;
                p_s[i][j] = (_Float16) (j <= i ? c[r] * expf(g_s[i] - g_s[j]) : 0.f);
            }
        }
        // K' = (k beta e^G) S and O_inter = Q S on this wave's slice (S as B from registers)
        gdn_wy_f8 Ks = {0, 0, 0, 0, 0, 0, 0, 0}, Oi = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const gdn_wy_h16 sb = gdn_wy_c_to_b(S[t], hi);
            Ks = GDN_WY_WMMA(*(const gdn_wy_h16 *) &kb_s[l16][16 * t], sb, Ks);
            Oi = GDN_WY_WMMA(*(const gdn_wy_h16 *) &q_s[l16][16 * t], sb, Oi);
        }
        gdn_wy_f8 Dm;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            Dm[r] = vb_s[2 * r + hi][dv0 + l16] - Ks[r];
        }
        GDN_WY_BARRIER();   // T, P ready
        gdn_wy_f8 Vn = {0, 0, 0, 0, 0, 0, 0, 0};
        Vn = GDN_WY_WMMA(*(const gdn_wy_h16 *) &t_s[l16][0], gdn_wy_c_to_b(Dm, hi), Vn);
        const gdn_wy_h16 vb = gdn_wy_c_to_b(Vn, hi);
        gdn_wy_f8 O = {0, 0, 0, 0, 0, 0, 0, 0};
        O = GDN_WY_WMMA(*(const gdn_wy_h16 *) &p_s[l16][0], vb, O);
        const int nn = min(16, n_chunked - t0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int i = 2 * r + hi;
            if (i < nn) {
                out[(int64_t) (t0 + i) * 128 * H + col] = O[r] + expf(g_s[i]) * Oi[r];
            }
        }
        const float eGL = expf(g_s[15]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                S[t][r] *= eGL;
            }
            S[t] = GDN_WY_WMMA(*(const gdn_wy_h16 *) &kdT[16 * t + l16][0], vb, S[t]);
        }
        GDN_WY_BARRIER();   // the chunk's inputs consumed
    }

    if constexpr (keep_rs) {
        // snapshot slot s = the state after token n_tokens - 1 - s
        if (n_chunked > 0 && n_tokens - n_chunked < K) {
            gdn_wy_store_state(S, s_out + (int64_t) (n_tokens - n_chunked) * state_slot_stride, col, hi);
        }
        __shared__ float kt_s[128], qt_s[128];
        for (int t = n_chunked; t < n_tokens; ++t) {
            __syncthreads();
            if (tid < 128) {
                kt_s[tid] = k[(int64_t) t * sq2 + (int64_t) kh * sq1 + tid];
                qt_s[tid] = q[(int64_t) t * sq2 + (int64_t) kh * sq1 + tid];
            }
            __syncthreads();
            const float a = expf(g[(int64_t) t * sb2 + (int64_t) vh * sb1]);
            const float b = beta[(int64_t) t * sb2 + (int64_t) vh * sb1];
            float kv = 0.f;
#pragma unroll
            for (int tt = 0; tt < 8; ++tt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    S[tt][r] *= a;
                    kv = fmaf(S[tt][r], kt_s[16 * tt + 2 * r + hi], kv);
                }
            }
            kv += __shfl_xor(kv, 16, 32);
            const float d = (v[(int64_t) t * sv2 + (int64_t) vh * sv1 + col] - kv) * b;
            float o = 0.f;
#pragma unroll
            for (int tt = 0; tt < 8; ++tt) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    S[tt][r] = fmaf(kt_s[16 * tt + 2 * r + hi], d, S[tt][r]);
                    o = fmaf(S[tt][r], qt_s[16 * tt + 2 * r + hi], o);
                }
            }
            o += __shfl_xor(o, 16, 32);
            if (hi == 0) {
                out[(int64_t) t * 128 * H + col] = o * scale;
            }
            gdn_wy_store_state(S, s_out + (int64_t) (n_tokens - 1 - t) * state_slot_stride, col, hi);
        }
    } else {
        gdn_wy_store_state(S, s_out, col, hi);
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, curr_state, dst, state, H, n_tokens, sq1, sq2, sv1, sv2, sb1, sb2, neqk1, scale,
                     state_slot_stride, K, state_rows);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}
#endif // defined(GGML_USE_HIP)
