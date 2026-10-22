#include "idx-relu-sum.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(GGML_USE_HIP) && !defined(cudaEventElapsedTime)
#define cudaEventElapsedTime hipEventElapsedTime
#endif

// Lightning-indexer head reduction: relu each head's block score, then sum the heads in graph order.
// Reproduces unary op_relu (fmaxf(x, 0)) followed by (((r0 + r1) + r2) + ...) with the same rounding, so the result is
// bitwise identical to the RELU + CONT + ADD chain it replaces, while reading the scores once instead of four times.
static __global__ void idx_relu_sum_f32(const float * __restrict__ src, float * __restrict__ dst,
                                        const int n_blocks, const int heads) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int64_t tt = blockIdx.y;                         // token within stream, streams stacked
    const float * p  = src + tt * (int64_t) heads * n_blocks + b;
    float acc = fmaxf(p[0], 0.0f);
    for (int h = 1; h < heads; ++h) {
        acc = acc + fmaxf(p[(int64_t) h * n_blocks], 0.0f);
    }
    dst[tt * (int64_t) n_blocks + b] = acc;
}

bool ggml_cuda_idx_relu_sum_enabled() {
    static const int v = getenv("LLAMA_IDX_RELU_SUM") ? atoi(getenv("LLAMA_IDX_RELU_SUM")) : 0;
    return v != 0;
}

void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args) {
    const ggml_tensor * s = args.score;
    const int n_blocks = (int) s->ne[0];
    const int64_t rows = s->ne[2] * s->ne[3];
    constexpr int threads = 256;
    const dim3 grid((n_blocks + threads - 1) / threads, (unsigned) rows, 1);
    idx_relu_sum_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) s->data, (float *) args.dst->data, n_blocks, args.heads);
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_RELU_SUM blocks=%d heads=%d rows=%lld\n", n_blocks, args.heads, (long long) rows);
}

// strixllama: the lightning indexer's score for one query strip, whole. out[t][b] = tails[t] > starts[b] ?
// ((relu(s0) + relu(s1)) + relu(s2)) + relu(s3) : -inf, s_h = q[t][h] . keys[b] over 128 dims - what the graph computes
// as MUL_MAT -> RELU -> head sum -> score + log(step(tail - start)), without its [blocks x heads x queries] F32
// intermediate (137 MB a strip at 64K tokens) and the four passes over it. The dot products are taken as the tile GEMM's
// two-product mode takes them (q split into F16 hi + lo, the keys - F16 values out of the block-key cache - as F16, F32
// accumulation over 16-deep steps in order), so the result is the unfused chain's.
// A block: 8 waves x 4 queries against 128 keys staged in LDS. A wave's 4 queries x 4 heads are the 16 rows of its WMMA
// tile (row = head * 4 + query) against 16 keys at a time; each lane then holds all four heads of two queries for one key.
typedef short idxs_v16s __attribute__((ext_vector_type(16)));
typedef float idxs_v8f  __attribute__((ext_vector_type(8)));
#define IDXS_KB 128
#define IDXS_LS (128 + 8)

static __device__ __forceinline__ uint32_t idxs_h2(const float a, const float b) {
    return (uint32_t) __builtin_bit_cast(uint16_t, (_Float16) a) | ((uint32_t) __builtin_bit_cast(uint16_t, (_Float16) b) << 16);
}

static __global__ void __launch_bounds__(256) k_idx_score(const float * __restrict__ keys, const int64_t ks, const int nb,
        const float * __restrict__ q, const int64_t qs, const int nq,
        const int32_t * __restrict__ starts, const int32_t * __restrict__ tails, float * __restrict__ out) {
    __shared__ __align__(16) uint16_t kt[IDXS_KB * IDXS_LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int b0 = blockIdx.y * IDXS_KB;
    for (int i = tid; i < IDXS_KB * 32; i += 256) {                    // 32 float4 a key
        const int r = i >> 5, c4 = (i & 31) * 4;
        const float4 v = *(const float4 *) (keys + (int64_t) min(b0 + r, nb - 1) * ks + c4);
        *(uint2 *) (kt + r * IDXS_LS + c4) = make_uint2(idxs_h2(v.x, v.y), idxs_h2(v.z, v.w));
    }
    const int r = lane & 15, h = r >> 2, tq = r & 3;
    const int t0 = blockIdx.x * 32 + wave * 4;
    const float * qr = q + ((int64_t) min(t0 + tq, nq - 1) * 4 + h) * qs;
    idxs_v16s ah[8], al[8];
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
        uint32_t ph[8], pl[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (qr + kc * 16 + j * 4);
            const float x[4] = {v.x, v.y, v.z, v.w};
            uint16_t hi[4], lo[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                hi[c] = __builtin_bit_cast(uint16_t, (_Float16) x[c]);
                lo[c] = __builtin_bit_cast(uint16_t, (_Float16) (x[c] - (float) __builtin_bit_cast(_Float16, hi[c])));
            }
            ph[2 * j] = hi[0] | ((uint32_t) hi[1] << 16); ph[2 * j + 1] = hi[2] | ((uint32_t) hi[3] << 16);
            pl[2 * j] = lo[0] | ((uint32_t) lo[1] << 16); pl[2 * j + 1] = lo[2] | ((uint32_t) lo[3] << 16);
        }
        ah[kc] = __builtin_bit_cast(idxs_v16s, (uint4[2]){make_uint4(ph[0], ph[1], ph[2], ph[3]), make_uint4(ph[4], ph[5], ph[6], ph[7])});
        al[kc] = __builtin_bit_cast(idxs_v16s, (uint4[2]){make_uint4(pl[0], pl[1], pl[2], pl[3]), make_uint4(pl[4], pl[5], pl[6], pl[7])});
    }
    __syncthreads();
    const int half = lane >> 4;
    const int ta = t0 + half, tb = t0 + 2 + half;
    const int tail_a = ta < nq ? tails[ta] : 0, tail_b = tb < nq ? tails[tb] : 0;
    for (int st = 0; st < IDXS_KB / 16; ++st) {
        idxs_v8f acc;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        const uint16_t * kr = kt + (st * 16 + r) * IDXS_LS;
#pragma unroll
        for (int kc = 0; kc < 8; ++kc) {
            const idxs_v16s bk = __builtin_bit_cast(idxs_v16s, (uint4[2]){*(const uint4 *) (kr + kc * 16), *(const uint4 *) (kr + kc * 16 + 8)});
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[kc], bk, acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[kc], bk, acc);
        }
        const int b = b0 + st * 16 + r;
        if (b < nb) {
            const int sb = starts[b];
            // rows 2e + half: e = 2h' (+1) is head h', query half (2 + half)
            if (ta < nq) {
                float s = fmaxf(acc[0], 0.0f);
                s = s + fmaxf(acc[2], 0.0f); s = s + fmaxf(acc[4], 0.0f); s = s + fmaxf(acc[6], 0.0f);
                out[(int64_t) ta * nb + b] = tail_a > sb ? s : -INFINITY;
            }
            if (tb < nq) {
                float s = fmaxf(acc[1], 0.0f);
                s = s + fmaxf(acc[3], 0.0f); s = s + fmaxf(acc[5], 0.0f); s = s + fmaxf(acc[7], 0.0f);
                out[(int64_t) tb * nb + b] = tail_b > sb ? s : -INFINITY;
            }
        }
    }
}

void ggml_cuda_op_idx_score(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_args & a) {
    const int nb = (int) a.keys->ne[1], nq = (int) (a.q->ne[1] / 4);
    const dim3 grid((unsigned) ((nq + 31) / 32), (unsigned) ((nb + IDXS_KB - 1) / IDXS_KB));
    k_idx_score<<<grid, 256, 0, ctx.stream()>>>((const float *) a.keys->data, (int64_t) (a.keys->nb[1] / sizeof(float)), nb,
        (const float *) a.q->data, (int64_t) (a.q->nb[1] / sizeof(float)), nq, a.starts, a.tails, (float *) a.out->data);
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_SCORE fused: blocks=%d queries=%d\n", nb, nq);
}

// strixllama: the indexer score of a decode-sized strip (at most 8 queries), whole: out[t][b] = s + log(step(tails[t] - starts[b]))
// with s = ((relu(s0) + relu(s1)) + relu(s2)) + relu(s3), s_h = q[t][h] . kb[rows[b]] - what the graph computes as GET_ROWS (the
// F16 block keys out of the cache as F32) -> MUL_MAT -> RELU -> the head sum -> the visibility, one pass over each key instead of
// an F32 copy of every key and seven passes over the scores. Each dot product is the one the vector kernel takes for K = 128
// (mul_mat_vec_f, 64 threads a row): thread l accumulates dims 2l and 2l + 1 with ggml_cuda_mad from 0, each wave of 32 sums its
// partials by butterfly (offsets 16, 8, 4, 2, 1: lane 0 ends with ((((p0 + p16) + (p8 + p24)) + ...)), and the two waves' sums
// are reduced again by butterfly with zeros in the other lanes, (w0 + 0) + (w1 + 0). Here one thread takes a key and spells that
// tree out, so the scores are the chain's, bit for bit. `zero` is 0.0f, passed in so those additions stay in the code.
#define IDXD_MAXQ 8

static __device__ __forceinline__ float idxd_partial(const half2 k, const float * __restrict__ q) {
    const float2 kx = __half22float2(k);
    float s = 0.0f;
    ggml_cuda_mad(s, kx.x, q[0]);
    ggml_cuda_mad(s, kx.y, q[1]);
    return s;
}

// one wave's butterfly sum of its 32 partials: k and q hold that wave's 64 dims
static __device__ __forceinline__ float idxd_wave(const half2 * __restrict__ k, const float * __restrict__ q) {
    float a[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = idxd_partial(k[i], q + 2 * i) + idxd_partial(k[i + 16], q + 2 * (i + 16));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = a[i] + a[i + 8]; }
#pragma unroll
    for (int i = 0; i < 4; ++i) { a[i] = a[i] + a[i + 4]; }
#pragma unroll
    for (int i = 0; i < 2; ++i) { a[i] = a[i] + a[i + 2]; }
    return a[0] + a[1];
}

// thread pair (2b, 2b + 1) takes key b: thread w the dims [64 w, 64 w + 64), the vector kernel's wave w
//
// strixllama: with several sequences in a strip a block is seen by the queries of its own sequence only, so each pair tests a
// query's visibility first and skips the dot products of an invisible one (the pair shares its key and query, so both skip
// together, and the shuffle stays inside it). A computed score is the same as before and a skipped one was -inf regardless.
// The membership product's terms are 0 or 1, so it is above 0.5 exactly when the block and the query share a slot: with at
// most 32 slots that is a test of two bit masks, the queries' made once a workgroup and a block's once a pair.
static __global__ void __launch_bounds__(256) k_idx_score_dec(const half * __restrict__ kb, const int64_t kbs,
        const int32_t * __restrict__ rows, const int nb, const float * __restrict__ q, const int64_t qs, const int nq,
        const int32_t * __restrict__ starts, const int32_t * __restrict__ tails, float * __restrict__ out, const float zero,
        const float * __restrict__ seq_blk, const float * __restrict__ seq_tok, const int n_slots) {
    __shared__ float qsh[IDXD_MAXQ * 4 * 128];
    __shared__ uint32_t tok_mask[IDXD_MAXQ];
    const bool use_mask = seq_blk != nullptr && n_slots <= 32;
    for (int i = threadIdx.x; i < nq * 4 * 128; i += blockDim.x) {
        qsh[i] = q[(int64_t) (i / 128) * qs + i % 128];
    }
    if (use_mask && threadIdx.x < (unsigned) nq) {
        uint32_t m = 0;
        for (int sl = 0; sl < n_slots; ++sl) {
            m |= (seq_tok[(int64_t) threadIdx.x * n_slots + sl] > 0.5f ? 1u : 0u) << sl;
        }
        tok_mask[threadIdx.x] = m;
    }
    __syncthreads();
    const int w = threadIdx.x & 1;
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 1;
    const bool valid = b < nb;
    uint32_t blk_mask = 0;
    if (valid && use_mask) {
        for (int sl = 0; sl < n_slots; ++sl) {
            blk_mask |= (seq_blk[(int64_t) b * n_slots + sl] > 0.5f ? 1u : 0u) << sl;
        }
    }
    half2 k[32];
    if (valid) {
        const uint4 * kp = (const uint4 *) (kb + (int64_t) rows[b] * kbs + 64 * w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint4 v = kp[i];
            k[4 * i + 0] = __builtin_bit_cast(half2, v.x);
            k[4 * i + 1] = __builtin_bit_cast(half2, v.y);
            k[4 * i + 2] = __builtin_bit_cast(half2, v.z);
            k[4 * i + 3] = __builtin_bit_cast(half2, v.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) { k[i] = __builtin_bit_cast(half2, 0u); }
    }
    const int st = valid ? starts[b] : 0;
    for (int t = 0; t < nq; ++t) {
        bool vis = valid && tails[t] > st;
        if (vis && seq_blk) {
            if (use_mask) {
                vis = (blk_mask & tok_mask[t]) != 0;
            } else {
                // the membership product's 0/1 terms: exact in any order
                float m = 0.0f;
                for (int sl = 0; sl < n_slots; ++sl) {
                    m += seq_blk[(int64_t) b * n_slots + sl] * seq_tok[(int64_t) t * n_slots + sl];
                }
                vis = m > 0.5f;
            }
        }
        if (!vis) {
            if (valid && w == 0) {
                out[(int64_t) t * nb + b] = -INFINITY;
            }
            continue;
        }
        float acc = 0.0f;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float mine  = idxd_wave(k, qsh + (t * 4 + h) * 128 + 64 * w);
            const float other = __shfl_xor_sync(0xffffffff, mine, 1, 32);
            const float w0 = w == 0 ? mine : other, w1 = w == 0 ? other : mine;
            const float v = (w0 + zero) + (w1 + zero);
            const float r = fmaxf(v, 0.0f);
            acc = h == 0 ? r : acc + r;
        }
        if (w == 0) {
            out[(int64_t) t * nb + b] = acc + zero;
        }
    }
}

// strixllama: the decode score with each thread pair computing only what its block's own queries see. With several
// conversations in a strip the block list is interleaved - a position bucket at a time, one block of each sequence (see
// llama_memory_hybrid_idx::set_input_qsa_run) - so the 16 blocks of a wave belong to up to 16 conversations, and
// k_idx_score_dec, which loops over the queries with the visible lanes doing the work, ran a wave's dot products once for
// every query: 8 times at 8 conversations. With the block keys in adjacent rows (llama_memory_hybrid_idx::kb_row) that
// is 205 us a strip at 8 x 20K against 48 us here, and one conversation at 110K takes 17 us against 22.
// Here each lane first marks which queries see its block, then takes those one after another with the query as its own
// index into LDS - one dot product a lane at 8 conversations without drafts, however the wave is mixed. The loop runs
// as long as its busiest lane (the pair shares a block, so the shuffle stays inside it). LDS holds the queries head by
// head, a query's two halves 66 floats apart and the queries 132 apart, so the 16 (query, half) reads of a head start
// on 16 different even banks. The arithmetic is k_idx_score_dec's, so the scores are the same bits.
#define IDXD3_TS 132
#define IDXD3_HS (IDXD_MAXQ * IDXD3_TS)
static __global__ void __launch_bounds__(256) k_idx_score_dec3(const half * __restrict__ kb, const int64_t kbs,
        const int32_t * __restrict__ rows, const int nb, const float * __restrict__ q, const int64_t qs, const int nq,
        const int32_t * __restrict__ starts, const int32_t * __restrict__ tails, float * __restrict__ out, const float zero,
        const float * __restrict__ seq_blk, const float * __restrict__ seq_tok, const int n_slots) {
    __shared__ float qsh[4 * IDXD3_HS];
    __shared__ uint32_t tok_mask[IDXD_MAXQ];
    __shared__ int32_t tail_sh[IDXD_MAXQ];
    const bool use_mask = seq_blk != nullptr && n_slots <= 32;
    for (int i = threadIdx.x; i < nq * 4 * 128; i += blockDim.x) {
        const int r = i / 128, c = i % 128;   // q row r = query r / 4, head r % 4
        qsh[(r % 4) * IDXD3_HS + (r / 4) * IDXD3_TS + (c / 64) * 66 + c % 64] = q[(int64_t) r * qs + c];
    }
    if (threadIdx.x < (unsigned) nq) {
        tail_sh[threadIdx.x] = tails[threadIdx.x];
        if (use_mask) {
            uint32_t m = 0;
            for (int sl = 0; sl < n_slots; ++sl) {
                m |= (seq_tok[(int64_t) threadIdx.x * n_slots + sl] > 0.5f ? 1u : 0u) << sl;
            }
            tok_mask[threadIdx.x] = m;
        }
    }
    __syncthreads();
    const int w = threadIdx.x & 1;
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 1;
    const bool valid = b < nb;
    uint32_t blk_mask = 0;
    if (valid && use_mask) {
        for (int sl = 0; sl < n_slots; ++sl) {
            blk_mask |= (seq_blk[(int64_t) b * n_slots + sl] > 0.5f ? 1u : 0u) << sl;
        }
    }
    half2 k[32];
    if (valid) {
        const uint4 * kp = (const uint4 *) (kb + (int64_t) rows[b] * kbs + 64 * w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint4 v = kp[i];
            k[4 * i + 0] = __builtin_bit_cast(half2, v.x);
            k[4 * i + 1] = __builtin_bit_cast(half2, v.y);
            k[4 * i + 2] = __builtin_bit_cast(half2, v.z);
            k[4 * i + 3] = __builtin_bit_cast(half2, v.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) { k[i] = __builtin_bit_cast(half2, 0u); }
    }
    const int st = valid ? starts[b] : 0;
    // the queries that see this block, by k_idx_score_dec's test; the others get -inf now
    uint32_t vis_bits = 0;
    for (int t = 0; t < nq; ++t) {
        bool vis = valid && tail_sh[t] > st;
        if (vis && seq_blk) {
            if (use_mask) {
                vis = (blk_mask & tok_mask[t]) != 0;
            } else {
                // the membership product's 0/1 terms: exact in any order
                float m = 0.0f;
                for (int sl = 0; sl < n_slots; ++sl) {
                    m += seq_blk[(int64_t) b * n_slots + sl] * seq_tok[(int64_t) t * n_slots + sl];
                }
                vis = m > 0.5f;
            }
        }
        if (vis) {
            vis_bits |= 1u << t;
        } else if (valid && w == 0) {
            out[(int64_t) t * nb + b] = -INFINITY;
        }
    }
    // then the visible ones, a lane its own query at a time
    while (__any(vis_bits != 0)) {
        const bool active = vis_bits != 0;
        const int t = active ? __ffs(vis_bits) - 1 : 0;
        vis_bits &= vis_bits - 1;
        float acc = 0.0f;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float mine  = idxd_wave(k, qsh + h * IDXD3_HS + t * IDXD3_TS + 66 * w);
            const float other = __shfl_xor_sync(0xffffffff, mine, 1, 32);
            const float w0 = w == 0 ? mine : other, w1 = w == 0 ? other : mine;
            const float v = (w0 + zero) + (w1 + zero);
            const float r = fmaxf(v, 0.0f);
            acc = h == 0 ? r : acc + r;
        }
        if (active && w == 0) {
            out[(int64_t) t * nb + b] = acc + zero;
        }
    }
}

static void idx_score_dec_launch(const int variant, const ggml_cuda_idx_score_dec_args & a, float * out, cudaStream_t stream) {
    const int nb = (int) a.rows->ne[0], nq = (int) (a.q->ne[1] / 4);
    const half *    kb   = (const half *) a.kb->data;
    const int64_t   kbs  = (int64_t) (a.kb->nb[1] / sizeof(half));
    const int32_t * rows = (const int32_t *) a.rows->data;
    const float *   q    = (const float *) a.q->data;
    const int64_t   qs   = (int64_t) (a.q->nb[1] / sizeof(float));
    if (variant == 3) {
        k_idx_score_dec3<<<(unsigned) ((2 * (int64_t) nb + 255) / 256), 256, 0, stream>>>(kb, kbs, rows, nb, q, qs, nq,
            a.starts, a.tails, out, 0.0f, a.seq_blk, a.seq_tok, a.n_slots);
    } else {
        k_idx_score_dec<<<(unsigned) ((2 * (int64_t) nb + 255) / 256), 256, 0, stream>>>(kb, kbs, rows, nb, q, qs, nq,
            a.starts, a.tails, out, 0.0f, a.seq_blk, a.seq_tok, a.n_slots);
    }
}

void ggml_cuda_op_idx_score_dec(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_dec_args & a) {
    const int nb = (int) a.rows->ne[0], nq = (int) (a.q->ne[1] / 4);
    GGML_ASSERT(nq >= 1 && nq <= IDXD_MAXQ && a.kb->nb[1] % 16 == 0 && ((uintptr_t) a.kb->data) % 16 == 0);
    // STRIX_IDXD_V=1: k_idx_score_dec (every query's dot products run for the whole wave); 3 (default): k_idx_score_dec3
    static const int variant = getenv("STRIX_IDXD_V") ? atoi(getenv("STRIX_IDXD_V")) : 3;
    // DEV: STRIX_IDXD_BENCH=n (graphs off) - at the first strips of 8192 blocks and more with n queries or more, both
    // kernels 50 times each, timed with events, and their scores compared bit for bit
    static const int bench = getenv("STRIX_IDXD_BENCH") ? atoi(getenv("STRIX_IDXD_BENCH")) : 0;
    static const int bench_skip = getenv("STRIX_IDXD_BENCH_SKIP") ? atoi(getenv("STRIX_IDXD_BENCH_SKIP")) : 0;
    static int benched = 0, seen = 0;
    if (bench > 0 && nb >= 8192 && nq >= bench && seen++ >= bench_skip && benched < 4) {
        ++benched;
        CUDA_CHECK(cudaDeviceSynchronize());
        const size_t n = (size_t) nb * nq;
        ggml_cuda_pool_alloc<float> o1(ctx.pool(), n), o2(ctx.pool(), n);
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreateWithFlags(&e0, 0));
        CUDA_CHECK(cudaEventCreateWithFlags(&e1, 0));
        float ms[3] = {0.0f, 0.0f, 0.0f};
        for (int v = 1; v <= 2; ++v) {
            float * o = v == 1 ? o1.get() : o2.get();
            const int kern = v == 1 ? 1 : 3;
            idx_score_dec_launch(kern, a, o, ctx.stream());   // warm
            CUDA_CHECK(cudaEventRecord(e0, ctx.stream()));
            for (int r = 0; r < 50; ++r) {
                idx_score_dec_launch(kern, a, o, ctx.stream());
            }
            CUDA_CHECK(cudaEventRecord(e1, ctx.stream()));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms[v], e0, e1));
        }
        std::vector<float> h1(n), h2(n);
        CUDA_CHECK(cudaMemcpy(h1.data(), o1.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h2.data(), o2.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
        const bool same = memcmp(h1.data(), h2.data(), n * sizeof(float)) == 0;
        fprintf(stderr, "IDXD_BENCH blocks=%d queries=%d slots=%d: v1 %.1f us, v3 %.1f us, v1 %.0f GB/s of keys, scores %s\n",
                nb, nq, a.n_slots, 1000.0f * ms[1] / 50, 1000.0f * ms[2] / 50, (double) nb * 256 / (1e6 * ms[1] / 50),
                same ? "identical" : "DIFFER");
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
    }
    idx_score_dec_launch(variant, a, (float *) a.out->data, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_SCORE_DEC fused: blocks=%d queries=%d kernel v%d\n", nb, nq, variant);
}
