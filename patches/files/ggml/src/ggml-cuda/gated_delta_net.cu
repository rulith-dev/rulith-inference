#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "common.cuh"

#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
template <int mask>
static __device__ __forceinline__ float gdn_dpp_row_xmask(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
}
static __device__ __forceinline__ float gdn_permlanex16_swap(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
}
static __device__ __forceinline__ float gdn_warp_reduce_sum32(float x) {
    x += gdn_permlanex16_swap(x);
    x += gdn_dpp_row_xmask<8>(x);
    x += gdn_dpp_row_xmask<4>(x);
    x += gdn_dpp_row_xmask<2>(x);
    x += gdn_dpp_row_xmask<1>(x);
    return x;
}
#define GDN_DPP_REDUCE 1
#endif

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(const float x) {
#if defined(GDN_DPP_REDUCE)
    if constexpr (width == 32) {
        return gdn_warp_reduce_sum32(x);
    }
#endif
    return warp_reduce_sum<width>(x);
}

// strixllama: COLS columns a warp. With one, a warp had one 512-byte column of state in flight, and at decode (a token a
// sequence) that is all a warp waits for: 8 conversations' 24 MB were read at ~117 GB/s. With two, both columns' loads
// are in flight together (the read ~0.6 of the time). Each column is summed by the same lanes in the same order.
template <int S_v, bool KDA, bool keep_rs_t, int COLS = 1>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_rows) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns COLS columns, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * COLS;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // strixllama: with state_rows, curr_state is the whole state tensor and the sequence's row is looked up
    const int64_t state_in_offset      = (state_rows ? (int64_t) state_rows[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col0 * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[COLS][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i   = r * warp_size + lane;
            s_shard[c][r] = curr_state[c * S_v + i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

#pragma unroll
            for (int c = 0; c < COLS; c++) {
                const int col = col0 + c;

                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    // strixllama: the fmaf the compiler made of g * s + k * delta, spelled out: the deferred-rollback
                    // kernels' replay must give these bits
                    s_shard[c][r]  = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            }
        } else {
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                const int col = col0 + c;

                // kv[col] = sum_i g[i] * S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kv_shard += expf(g_t[i]) * s_shard[c][r] * k_reg[r];
                }

                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - kv[col]) * beta
                float delta_col = (v_t[col] - kv_col) * beta_val;

                // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[c][r]  = expf(g_t[i]) * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < COLS; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[(col0 + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < COLS; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                state[(col0 + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NUM_WARPS, 1)
gated_delta_net_tiled_cuda(const float * q,
                           const float * k,
                           const float * v,
                           const float * g,
                           const float * beta,
                           const float * curr_state,
                           float *       dst,
                           float *       state,
                           int64_t       H,
                           int64_t       n_tokens,
                           int64_t       sq1,
                           int64_t       sq2,
                           int64_t       sq3,
                           int64_t       sv1,
                           int64_t       sv2,
                           int64_t       sv3,
                           int64_t       sb1,
                           int64_t       sb2,
                           int64_t       sb3,
                           const uint3   neqk1_magic,
                           const uint3   rq3_magic,
                           float         scale,
                           int64_t       state_slot_stride,
                           int           K,
                           const int32_t * state_rows) {
    constexpr int warp_size     = 32;
    constexpr int rows_per_lane = S_v / warp_size;
    constexpr int block_cols    = NUM_WARPS * COLS;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of the warp size");
    static_assert(S_v % block_cols == 0, "block columns must divide S_v");

    __shared__ float q_shared[TOKEN_TILE][S_v];
    __shared__ float k_shared[TOKEN_TILE][S_v];
    __shared__ float v_shared[TOKEN_TILE][block_cols];
    __shared__ float g_shared[TOKEN_TILE];
    __shared__ float beta_shared[TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = blockIdx.z * block_cols;          // first column of the block
    const int      colw     = threadIdx.y * COLS;               // first column of this warp inside the block
    const int      thread   = threadIdx.y * warp_size + lane;
    constexpr int  nthreads = NUM_WARPS * warp_size;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = (state_rows ? (int64_t) state_rows[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + (col0 + colw) * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0 + colw;

    float s_shard[COLS][rows_per_lane];

#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile_size = min((int64_t) TOKEN_TILE, n_tokens - t0);

        for (int idx = thread; idx < tile_size * S_v; idx += nthreads) {
            const int tt = idx / S_v;
            const int i  = idx % S_v;
            const int t  = t0 + tt;
            q_shared[tt][i] = q[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
            k_shared[tt][i] = k[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
        }
        for (int idx = thread; idx < tile_size * block_cols; idx += nthreads) {
            const int tt = idx / block_cols;
            const int c  = idx % block_cols;
            const int t  = t0 + tt;
            v_shared[tt][c] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col0 + c];
        }
        if (thread < tile_size) {
            const int64_t gb_offset = sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1;
            g_shared[thread]    = g[gb_offset];
            beta_shared[thread] = beta[gb_offset];
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_shared[tt][i];
                q_reg[r] = q_shared[tt][i];
            }

            const float g_val    = expf(g_shared[tt]);
            const float beta_val = beta_shared[tt];

            float attn_col[COLS];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard = fmaf(s_shard[c][r], k_reg[r], kv_shard);
                }
                const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = fmaf(-g_val, kv_col, v_shared[tt][colw + c]) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                    attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
                }
                attn_col[c] = gdn_warp_reduce_sum<warp_size>(attn_partial);
            }

            if (lane < COLS) {
                float a = attn_col[0];
#pragma unroll
                for (int c = 1; c < COLS; c++) {
                    a = lane == c ? attn_col[c] : a;
                }
                attn_data[(int64_t) t * S_v * H + lane] = a * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snapshot = state + target_slot * state_slot_stride + (col0 + colw) * S_v;
#pragma unroll
                    for (int c = 0; c < COLS; c++) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            snapshot[c * S_v + r * warp_size + lane] = s_shard[c][r];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < COLS; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + colw + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

// strixllama: the prefill recurrence with 16 state rows and C_LANE columns a lane (8 lanes share a column group), so a
// dot product over the 128 rows is 16 FMAs and a 3-step DPP reduction instead of 4 FMAs and 5 steps. The output
// comes from the state before the update, o = scale * (g * S^T q + delta * (k . q)) - the same value as scale * S'^T q
// - so both reductions of a token (S^T k and S^T q, every column at once) run together and nothing waits on the update.
// The tiled kernel above spends ~3.3 ms a layer at 2K tokens on latency; this one does the same arithmetic in ~1/3.
template <int mask>
static __device__ __forceinline__ float gdn_xor8_step(const float x) {
#if defined(GDN_DPP_REDUCE)
    return gdn_dpp_row_xmask<mask>(x);
#else
    return __shfl_xor(x, mask, 32);
#endif
}

static __device__ __forceinline__ float gdn_sum8(float x) {   // over the 8 lanes that differ in bits 0-2
    x += gdn_xor8_step<1>(x);
    x += gdn_xor8_step<2>(x);
    x += gdn_xor8_step<4>(x);
    return x;
}

template <int C_LANE, int NWAVES, int TOKEN_TILE, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NWAVES)
gated_delta_net_r16_cuda(const float * q,
                         const float * k,
                         const float * v,
                         const float * g,
                         const float * beta,
                         const float * curr_state,
                         float *       dst,
                         float *       state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       state_slot_stride,
                         int           K,
                         const int32_t * state_rows) {
    constexpr int S_v    = 128;
    constexpr int RL     = 16;                  // state rows a lane
    constexpr int WCOLS  = 4 * C_LANE;          // a wave: 4 column groups of C_LANE, each over 8 lanes x 16 rows
    constexpr int BCOLS  = NWAVES * WCOLS;
    static_assert(S_v % BCOLS == 0, "block columns must divide S_v");

    // one tile: a second one (loads in flight while this one is consumed) doubles the LDS a workgroup takes, and then
    // fewer workgroups share a CU - with this kernel's per-token latency the co-resident count is what sets the time
    __shared__ __align__(16) float q_shared[1][TOKEN_TILE][S_v];
    __shared__ __align__(16) float k_shared[1][TOKEN_TILE][S_v];
    __shared__ __align__(16) float v_shared[1][TOKEN_TILE][BCOLS];
    __shared__ float g_shared[1][TOKEN_TILE];
    __shared__ float beta_shared[1][TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x & 31;
    const int      wave     = threadIdx.x >> 5;
    const int      rg       = lane & 7;          // rows 16 rg .. 16 rg + 15
    const int      cg       = lane >> 3;
    const int      col0     = blockIdx.z * BCOLS;
    const int      colb     = wave * WCOLS + cg * C_LANE;   // this lane's first column inside the block
    const int      thread   = threadIdx.x;
    constexpr int  nthreads = NWAVES * 32;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = (state_rows ? (int64_t) state_rows[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0 + colb;

    // s[c][r] = S[16 rg + r][column c] = M[column c][16 rg + r]: the stored (transposed) rows are contiguous. The
    // registers hold the state divided by gm, the decay since it was last folded in: S = gm * s
    float s[C_LANE][RL];
    float gm = 1.0f;
#pragma unroll
    for (int c = 0; c < C_LANE; c++) {
        const float4 * src = (const float4 *) (curr_state + (int64_t) (col0 + colb + c) * S_v + RL * rg);
#pragma unroll
        for (int r4 = 0; r4 < RL / 4; r4++) {
            const float4 x = src[r4];
            s[c][4*r4 + 0] = x.x; s[c][4*r4 + 1] = x.y; s[c][4*r4 + 2] = x.z; s[c][4*r4 + 3] = x.w;
        }
    }

    // the tile loader: float4s of q and k (whole rows) and of v (this block's columns), g and beta per token
    constexpr int QK4 = TOKEN_TILE * S_v / 4;
    constexpr int V4  = TOKEN_TILE * BCOLS / 4;
    constexpr int NQ  = (QK4 + nthreads - 1) / nthreads;
    constexpr int NV  = (V4 + nthreads - 1) / nthreads;
    // strixllama: the tile goes from global memory straight to LDS. Staged through register arrays (rq, rk, rv) it
    // kept them in scratch - 10 scratch_store_b128 and 10 scratch_load_b128 a tile, a private segment of 112-272
    // bytes a lane - and a dispatch that needs scratch shares the queue's scratch ring with every other one
    const int buf = 0;
    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile_size = min((int64_t) TOKEN_TILE, n_tokens - t0);
#pragma unroll
        for (int j = 0; j < NQ; j++) {
            const int idx = thread + j * nthreads, tt = idx / (S_v / 4), i4 = idx % (S_v / 4);
            if (idx < QK4 && tt < tile_size) {
                const int64_t off = iq3 * sq3 + (int64_t) (t0 + tt) * sq2 + iq1 * sq1 + 4 * i4;
                const float4 xq = *(const float4 *) (q + off);
                const float4 xk = *(const float4 *) (k + off);
                *(float4 *) &q_shared[buf][tt][4 * i4] = xq;
                *(float4 *) &k_shared[buf][tt][4 * i4] = xk;
            }
        }
#pragma unroll
        for (int j = 0; j < NV; j++) {
            const int idx = thread + j * nthreads, tt = idx / (BCOLS / 4), c4 = idx % (BCOLS / 4);
            if (idx < V4 && tt < tile_size) {
                *(float4 *) &v_shared[buf][tt][4 * c4] =
                    *(const float4 *) (v + sequence * sv3 + (int64_t) (t0 + tt) * sv2 + h_idx * sv1 + col0 + 4 * c4);
            }
        }
        if (thread < tile_size) {
            const int64_t gb_offset = sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1;
            g_shared[buf][thread]    = g[gb_offset];
            beta_shared[buf][thread] = beta[gb_offset];
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[RL], q_reg[RL];
#pragma unroll
            for (int r4 = 0; r4 < RL / 4; r4++) {
                const float4 kx = *(const float4 *) &k_shared[buf][tt][RL * rg + 4 * r4];
                const float4 qx = *(const float4 *) &q_shared[buf][tt][RL * rg + 4 * r4];
                k_reg[4*r4 + 0] = kx.x; k_reg[4*r4 + 1] = kx.y; k_reg[4*r4 + 2] = kx.z; k_reg[4*r4 + 3] = kx.w;
                q_reg[4*r4 + 0] = qx.x; q_reg[4*r4 + 1] = qx.y; q_reg[4*r4 + 2] = qx.z; q_reg[4*r4 + 3] = qx.w;
            }

            // S^T k and S^T q for every column, and k . q, over this lane's rows; then over the 8 row groups
            float kv[C_LANE], qs[C_LANE], kq = 0.0f;
#pragma unroll
            for (int c = 0; c < C_LANE; c++) {
                kv[c] = 0.0f;
                qs[c] = 0.0f;
#pragma unroll
                for (int r = 0; r < RL; r++) {
                    kv[c] = fmaf(s[c][r], k_reg[r], kv[c]);
                    qs[c] = fmaf(s[c][r], q_reg[r], qs[c]);
                }
            }
#pragma unroll
            for (int r = 0; r < RL; r++) {
                kq = fmaf(k_reg[r], q_reg[r], kq);
            }
#pragma unroll
            for (int c = 0; c < C_LANE; c++) {
                kv[c] = gdn_sum8(kv[c]);
                qs[c] = gdn_sum8(qs[c]);
            }
            kq = gdn_sum8(kq);

            const float g_val    = expf(g_shared[buf][tt]);
            const float beta_val = beta_shared[buf][tt];
            // kv and qs were taken from the scaled state: S^T k = gm * kv, so g * S^T k = gt * kv
            const float gt   = gm * g_val;
            const bool  fold = gt < 0x1p-20f;   // uniform: the decay is the head's

            float o[C_LANE], dl[C_LANE];
#pragma unroll
            for (int c = 0; c < C_LANE; c++) {
                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta = fmaf(-gt, kv[c], v_shared[buf][tt][colb + c]) * beta_val;
                o[c]  = fmaf(gt, qs[c], delta * kq) * scale;
                dl[c] = delta;
            }
            if (!fold) {
                // S' = g S + k delta^T = gt * (Sh + k (delta / gt)^T): one FMA an element
                const float inv = 1.0f / gt;
#pragma unroll
                for (int c = 0; c < C_LANE; c++) {
                    const float dc = dl[c] * inv;
#pragma unroll
                    for (int r = 0; r < RL; r++) {
                        s[c][r] = fmaf(k_reg[r], dc, s[c][r]);
                    }
                }
                gm = gt;
            } else {
                // the scale is about to leave the float range the updates keep: fold it into the state
#pragma unroll
                for (int c = 0; c < C_LANE; c++) {
#pragma unroll
                    for (int r = 0; r < RL; r++) {
                        s[c][r] = fmaf(gt, s[c][r], k_reg[r] * dl[c]);
                    }
                }
                gm = 1.0f;
            }

            if (rg == 0) {
                float * out = attn_data + (int64_t) t * S_v * H;
#pragma unroll
                for (int c = 0; c < C_LANE; c++) {
                    out[c] = o[c];
                }
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snapshot = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int c = 0; c < C_LANE; c++) {
                        float4 * d4 = (float4 *) (snapshot + (int64_t) (col0 + colb + c) * S_v + RL * rg);
#pragma unroll
                        for (int r4 = 0; r4 < RL / 4; r4++) {
                            d4[r4] = make_float4(gm * s[c][4*r4 + 0], gm * s[c][4*r4 + 1], gm * s[c][4*r4 + 2], gm * s[c][4*r4 + 3]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C_LANE; c++) {
            float4 * d4 = (float4 *) (state + (int64_t) (col0 + colb + c) * S_v + RL * rg);
#pragma unroll
            for (int r4 = 0; r4 < RL / 4; r4++) {
                d4[r4] = make_float4(gm * s[c][4*r4 + 0], gm * s[c][4*r4 + 1], gm * s[c][4*r4 + 2], gm * s[c][4*r4 + 3]);
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * state_rows, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    if constexpr (!KDA) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        // STRIX_GDN_R16: waves a workgroup for the 16-row kernel (2, 4 or 8; 0 = the tiled kernel). At 2K tokens a layer
        // takes 2.7 ms with 4 (96 workgroups), 2.9 with 8 and 4.6 with 2 (each workgroup loads the whole q/k tile);
        // the tiled kernel 3.4
        static const int r16 = getenv("STRIX_GDN_R16") ? atoi(getenv("STRIX_GDN_R16")) : 4;
        // the tile loader reads q, k and v in float4s
        const bool vec4 = ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0 && ((uintptr_t) v_d) % 16 == 0 &&
            sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0;
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && warp_size == 32 && S_v == 128 && n_seqs == 1 && n_tokens >= 16 &&
                n_tokens <= 32768 && vec4 && (r16 == 2 || r16 == 4 || r16 == 8)) {
            static unsigned hits = 0;
            if (hits++ == 0) fprintf(stderr, "FORK_GDN_R16 tokens=%ld waves=%d snapshots=%d keep=%d\n", n_tokens, r16, K, int(keep_rs_t));
            const dim3 grid(H, n_seqs, 128 / (16 * r16)), block(32 * r16, 1, 1);
            const ggml_cuda_kernel_launch_params params(grid, block, 0, stream);
#define GDN_R16_LAUNCH(NW) ggml_cuda_kernel_launch(gated_delta_net_r16_cuda<4, NW, 16, keep_rs_t>, params,                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,                 neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows)
            if (r16 == 2) {
                GDN_R16_LAUNCH(2);
            } else if (r16 == 4) {
                GDN_R16_LAUNCH(4);
            } else {
                GDN_R16_LAUNCH(8);
            }
#undef GDN_R16_LAUNCH
            return;
        }
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && H == 48 && n_seqs == 1 && n_tokens >= 16 && n_tokens <= 32768) {
            const dim3 tiled_grid(H, n_seqs, 2);
            const dim3 tiled_block(warp_size, 8, 1);
            const ggml_cuda_kernel_launch_params tiled_params(tiled_grid, tiled_block, 0, stream);
            static unsigned hits = 0;
            if (hits++ == 0) fprintf(stderr, "FORK_GDN_TILE tokens=%ld snapshots=%d keep=%d\n", n_tokens, K, int(keep_rs_t));
            ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<128, 8, 8, 16, keep_rs_t>, tiled_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        }
        case 128: {
            // strixllama: two columns a warp for batches of a few tokens a sequence (decode, verify); longer ones keep
            // one, as they spend their time in the per-token chain. STRIX_GDN_COLS=1: one column always
            static const int gdn_cols = getenv("STRIX_GDN_COLS") ? (strcmp(getenv("STRIX_GDN_COLS"), "ab") ? atoi(getenv("STRIX_GDN_COLS")) : -1) : 2;
            if constexpr (!KDA) {
                // STRIX_GDN_COLS=ab: one column on side A of the in-run A/B (STRIX_AB), two on B
                if ((gdn_cols == 2 || (gdn_cols < 0 && ggml_cuda_strix_ab())) && warp_size == 32 && n_tokens <= 16) {
                    static unsigned hits = 0;
                    if (hits++ == 0) fprintf(stderr, "FORK_GDN_COLS2 seqs=%ld tokens=%ld keep=%d\n", n_seqs, n_tokens, int(keep_rs_t));
                    const dim3 grid2(H, n_seqs, S_v / (num_warps * 2));
                    const ggml_cuda_kernel_launch_params params2(grid2, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, 2>, params2,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct gdn_qk_src { const ggml_tensor * gdn; const void * data; const float * q, * k; int64_t s1, s2, s3; };
static std::vector<gdn_qk_src> g_gdn_qk;

void ggml_cuda_gdn_qk_from(const ggml_tensor * gdn, const float * q, const float * k, int64_t s1, int64_t s2, int64_t s3) {
    for (auto it = g_gdn_qk.begin(); it != g_gdn_qk.end(); ++it) {
        if (it->gdn == gdn) { g_gdn_qk.erase(it); break; }
    }
    g_gdn_qk.push_back({gdn, gdn->data, q, k, s1, s2, s3});
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const gdn_qk_src * qk = nullptr;
    for (const auto & e : g_gdn_qk) {
        if (e.gdn == dst && e.data == dst->data) { qk = &e; break; }
    }
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    // strixllama: the states read straight from the cache rows the skipped gather would have copied
    const int32_t * state_rows = cache ? cache->state_rows : nullptr;
    const float *   s_d        = state_rows ? cache->state_base : (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    int64_t sq1 = nbq1 / sizeof(float);
    int64_t sq2 = nbq2 / sizeof(float);
    int64_t sq3 = nbq3 / sizeof(float);
    const bool qk_redirected = qk != nullptr;
    if (qk != nullptr) {   // ggml_cuda_gdn_qk_from: once
        q_d = qk->q; k_d = qk->k; sq1 = qk->s1; sq2 = qk->s2; sq3 = qk->s3;
        g_gdn_qk.erase(g_gdn_qk.begin() + (qk - g_gdn_qk.data()));
        qk = nullptr;
    }
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // strixllama: upstream's chunked MMA kernel takes q, k and the states from the tensors - not when they come from
    // elsewhere (ggml_cuda_gdn_qk_from, the cache rows) - and on HIP only with STRIX_GDN_MMA=1 until it is measured
    // against the r16 prefill kernel on gfx1151
#ifdef GGML_USE_HIP
    static const bool gdn_mma = getenv("STRIX_GDN_MMA") != nullptr && atoi(getenv("STRIX_GDN_MMA")) != 0;
#else
    static const bool gdn_mma = true;
#endif
    if (gdn_mma && !qk_redirected && state_rows == nullptr) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        }
    }
}

// strixllama: the net with deferred rollback (ggml_gated_delta_net_lazy) and the replay alone (ggml_gated_delta_net_replay).
// A verify batch wrote a full state snapshot per token per sequence per layer (3 MB each here) so that rejected drafts
// could be rolled back; ~0.7 ms a step for each 108 MB, 3-4% of a step at 1-4 conversations. Instead the state is left
// where the batch found it, and each token records what a replay needs - its delta [S_v, H_v], key [S_k, H_k] and gate
// [H_v], 33 KB - and the next batch starts by replaying the records the rollback kept, in registers, before it writes the
// state back. The update is fmaf(g, s, k * delta) in both, as in the plain kernel, so the bits are the plain net's.
// One warp per column, four columns a workgroup, as the plain kernel.
// COLS columns a warp, as the plain kernel.
template <int S_v, bool replay_only, int COLS = 1>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_lazy_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                          float * state_all, float * rec, const int32_t * info, float * dst,
                          int64_t H, int64_t H_k, int64_t n_tokens,
                          int64_t sq1, int64_t sq2, int64_t sq3,
                          int64_t sv1, int64_t sv2, int64_t sv3,
                          int64_t sb1, int64_t sb2, int64_t sb3,
                          const uint3 rq3_magic, float scale, int64_t rec_floats, int64_t R) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * COLS;
    const int64_t  kh       = h_idx % H_k;

    const int32_t * inf     = info + 5 * sequence;
    const int64_t   row_in  = inf[0];
    const int64_t   row_out = inf[1];
    const int       n_rep   = inf[2];
    if (replay_only && n_rep == 0 && row_in == row_out) {
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;

    const int64_t D     = H * S_v * S_v;
    const float * s_in  = state_all + row_in  * D + h_idx * S_v * S_v + col0 * S_v;
    float *       s_out = state_all + (row_out >= 0 ? row_out : row_in) * D + h_idx * S_v * S_v + col0 * S_v;

    float s_shard[COLS][rows_per_lane];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = s_in[c * S_v + r * warp_size + lane];
        }
    }

    // records by index: inf[3] the first replayed, inf[4] the first this batch writes (R: records a set)
    GGML_UNUSED(R);
    const float * rd = rec + (int64_t) inf[3] * rec_floats;
    // in blocks of RB records: a block's loads (gate, deltas, key) issued together, then its updates in record order -
    // records accumulate (up to STRIX_GDN_ACC a set), and one at a time each waited for its loads
    constexpr int RB = 6;
    for (int j0 = 0; j0 < n_rep; j0 += RB) {
        float gvb[RB], db[RB][COLS], kb[RB][rows_per_lane];
#pragma unroll
        for (int b = 0; b < RB; b++) {
            if (j0 + b < n_rep) {
                const float * rj = rd + (int64_t) (j0 + b) * rec_floats;
                gvb[b] = rj[H * S_v + H_k * S_v + h_idx];
#pragma unroll
                for (int c = 0; c < COLS; c++) {
                    db[b][c] = rj[h_idx * S_v + col0 + c];
                }
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kb[b][r] = rj[H * S_v + kh * S_v + r * warp_size + lane];
                }
            }
        }
#pragma unroll
        for (int b = 0; b < RB; b++) {
            if (j0 + b < n_rep) {
#pragma unroll
                for (int c = 0; c < COLS; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[c][r] = fmaf(gvb[b], s_shard[c][r], kb[b][r] * db[b][c]);
                    }
                }
            }
        }
    }
    // row written -1: the batch's records follow the replayed ones and the row stays as it is
    if (row_out >= 0 && (n_rep > 0 || row_in != row_out)) {
#pragma unroll
        for (int c = 0; c < COLS; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_out[c * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
    if constexpr (replay_only) {
        return;
    } else {
        const uint32_t iq3       = fastdiv(sequence, rq3_magic);
        float *        attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
        float *        wr        = rec + (int64_t) inf[4] * rec_floats;

        for (int t = 0; t < n_tokens; t++) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + kh * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + kh * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float   beta_val  = beta[gb_offset];
            const float   g_val     = expf(g[gb_offset]);

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * warp_size + lane];
                q_reg[r] = q_t[r * warp_size + lane];
            }

            // the record: the delta by each column's warp, the key and the gate once per head
            float * rt = wr + t * rec_floats;
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                const int col = col0 + c;

                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
                const float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                    attn_partial += s_shard[c][r] * q_reg[r];
                }
                const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
                if (lane == 0) {
                    attn_data[col]        = attn_col * scale;
                    rt[h_idx * S_v + col] = delta_col;
                }
            }
            attn_data += S_v * H;
            if (blockIdx.z == 0 && threadIdx.y == 0) {
                if ((int64_t) h_idx < H_k) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        rt[H * S_v + h_idx * S_v + r * warp_size + lane] = k_reg[r];
                    }
                }
                if (lane == 0) {
                    rt[H * S_v + H_k * S_v + h_idx] = g_val;
                }
            }
        }
    }
}

static void ggml_cuda_op_gated_delta_net_lazy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 1);
    const ggml_tensor * src_state = dst->src[5];
    const ggml_tensor * src_rec   = dst->src[6];
    const ggml_tensor * src_info  = dst->src[7];

    const int64_t S_v    = mode == 2 ? ggml_get_op_params_i32(dst, 2) : dst->src[2]->ne[0];
    const int64_t H      = mode == 2 ? ggml_get_op_params_i32(dst, 3) : dst->src[2]->ne[1];
    const int64_t H_k    = mode == 2 ? ggml_get_op_params_i32(dst, 4) : dst->src[1]->ne[1];
    const int64_t n_seqs = src_info->ne[1];

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    GGML_ASSERT(S_v == 128 && (warp_size == 32 || warp_size == 64));
    const int num_warps = 4;
    // strixllama: two columns a warp (STRIX_GDN_COLS=1: one), as the plain kernel for decode batches
    static const int gdn_cols = getenv("STRIX_GDN_COLS") ? (strcmp(getenv("STRIX_GDN_COLS"), "ab") ? atoi(getenv("STRIX_GDN_COLS")) : -1) : 2;
    const bool two = (gdn_cols == 2 || (gdn_cols < 0 && ggml_cuda_strix_ab())) && warp_size == 32;
    const dim3 grid(H, n_seqs, S_v / num_warps / (two ? 2 : 1));
    const dim3 block(warp_size, num_warps, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());

    float *         state_all  = (float *) src_state->data;
    float *         rec        = (float *) src_rec->data;
    const int32_t * info       = (const int32_t *) src_info->data;
    const int64_t   rec_floats = src_rec->ne[0];
    const int64_t   R          = src_rec->ne[1];

    if (mode == 2) {
        ggml_cuda_kernel_launch(two ? gated_delta_net_lazy_cuda<128, true, 2> : gated_delta_net_lazy_cuda<128, true, 1>, params,
            nullptr, nullptr, nullptr, nullptr, nullptr, state_all, rec, info, nullptr,
            H, H_k, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0,
            (int64_t) 0, (int64_t) 0, (int64_t) 0, init_fastdiv_values(1), 1.0f, rec_floats, R);
        return;
    }

    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_beta = dst->src[4];
    GGML_ASSERT(ggml_are_same_stride(src_q, dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[3]) && ggml_is_contiguous(src_beta));

    const int64_t n_tokens = src_v->ne[2];
    const int64_t rq3      = src_v->ne[3] / src_q->ne[3];

    // strixllama: q and k from where the fused conv + norms left them (ggml_cuda_gdn_qk_from), as the plain net
    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) dst->src[1]->data;
    int64_t sq1 = src_q->nb[1] / sizeof(float), sq2 = src_q->nb[2] / sizeof(float), sq3 = src_q->nb[3] / sizeof(float);
    for (auto it = g_gdn_qk.begin(); it != g_gdn_qk.end(); ++it) {
        if (it->gdn == dst && it->data == dst->data) {
            q_d = it->q; k_d = it->k; sq1 = it->s1; sq2 = it->s2; sq3 = it->s3;
            g_gdn_qk.erase(it);
            break;
        }
    }

    ggml_cuda_kernel_launch(two ? gated_delta_net_lazy_cuda<128, false, 2> : gated_delta_net_lazy_cuda<128, false, 1>, params,
        q_d, k_d, (const float *) src_v->data,
        (const float *) dst->src[3]->data, (const float *) src_beta->data, state_all, rec, info, (float *) dst->data,
        H, H_k, n_tokens,
        sq1, sq2, sq3,
        (int64_t) (src_v->nb[1] / sizeof(float)), (int64_t) (src_v->nb[2] / sizeof(float)), (int64_t) (src_v->nb[3] / sizeof(float)),
        (int64_t) (src_beta->nb[1] / sizeof(float)), (int64_t) (src_beta->nb[2] / sizeof(float)), (int64_t) (src_beta->nb[3] / sizeof(float)),
        init_fastdiv_values(rq3), 1.0f / sqrtf((float) S_v), rec_floats, R);
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 1) != 0) {
        ggml_cuda_op_gated_delta_net_lazy(ctx, dst);
        return;
    }
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

#include "ggml-backend-impl.h"

// strixllama: see gated_delta_net.cuh. The stream is drained first: the pool block the info goes to may still be read
// by the last graph's kernels, and the host copy of the rows that follows runs on another stream
void ggml_backend_cuda_gdn_replay(ggml_backend_t backend, int n_layer, ggml_tensor ** states, ggml_tensor ** recs,
                                  int32_t row, int32_t n_rep, int32_t first, int S_v, int H_v, int H_k) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();
    const int warp_size = ggml_cuda_info().devices[ctx->device].warp_size;
    GGML_ASSERT(S_v == 128 && (warp_size == 32 || warp_size == 64));
    const bool two = warp_size == 32;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ggml_cuda_pool_alloc<int32_t> info(ctx->pool(), 5);
    const int32_t h_info[5] = { row, row, n_rep, first, first };
    CUDA_CHECK(cudaMemcpy(info.get(), h_info, sizeof(h_info), cudaMemcpyHostToDevice));
    const dim3 grid(H_v, 1, S_v / 4 / (two ? 2 : 1));
    const dim3 block(warp_size, 4, 1);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, stream);
    for (int il = 0; il < n_layer; ++il) {
        ggml_cuda_kernel_launch(two ? gated_delta_net_lazy_cuda<128, true, 2> : gated_delta_net_lazy_cuda<128, true, 1>, params,
            nullptr, nullptr, nullptr, nullptr, nullptr, (float *) states[il]->data, (float *) recs[il]->data,
            (const int32_t *) info.get(), nullptr, (int64_t) H_v, (int64_t) H_k, (int64_t) 0, (int64_t) 0, (int64_t) 0,
            (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0, (int64_t) 0,
            init_fastdiv_values(1), 1.0f, (int64_t) recs[il]->ne[0], (int64_t) recs[il]->ne[1]);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
