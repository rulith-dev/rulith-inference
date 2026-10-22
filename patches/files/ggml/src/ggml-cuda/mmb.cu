#include "mmb.cuh"
#include "unary.cuh"
#include <unordered_map>
#include <map>
#include <utility>
#include "mmid.cuh"
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <unordered_set>

namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
typedef uint32_t mmb_u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t mmb_u32x4 __attribute__((ext_vector_type(4)));
constexpr int MMB_BK = 64, MMB_NT = 256, MMB_LDS_STRIDE = MMB_BK + 8;

__device__ __forceinline__ uint16_t mmb_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ uint32_t mmb_pack2(float a, float b) { return (uint32_t)mmb_f2bf(a) | ((uint32_t)mmb_f2bf(b) << 16); }
__constant__ int8_t mmb_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ float mmb_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

// strixllama: a block barrier that waits for LDS traffic only. __syncthreads() is a release/acquire fence as well, which
// here waits for every outstanding global load (vmcnt(0)): the next step's weights, issued before the WMMAs, had to land
// before the barrier after them, so a step's load latency was hidden by one step's compute at most, whatever the
// prefetch depth (STRIX_MMB_PF=2 measured the same). The tile loops share nothing through global memory, so LDS
// visibility (lgkmcnt(0)) is all they need, and the loads a store_lds consumes are waited for by their own registers'
// s_waitcnt. As composable kernels' block_sync_lds
__device__ __forceinline__ void mmb_lds_sync() {
#if defined(__HIP_PLATFORM_AMD__)
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

__global__ void mmb_cvt_f32_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o; o.x = mmb_pack2(a.x, a.y); o.y = mmb_pack2(a.z, a.w); o.z = mmb_pack2(b.x, b.y); o.w = mmb_pack2(b.z, b.w);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmb_f2bf(x[i]);
    }
}

// dequantize one weight row's two consecutive IQ4_NL blocks (36 bytes) into 64 bf16 in LDS.
// LUT held in registers as (kv + 128) bytes and applied with v_perm_b32 (4 nibbles per op pair) instead of a per-lane
// indexed constant array (which lowers to one scalar-byte memory load per element). The value kv*d is produced as
// fma(kv+128, d, -128*d): -128*d is exact, so the single rounding equals RN(kv*d) -> bitwise the same BF16 as before.
__device__ __forceinline__ void mmb_dq_row36(const uint4 w0, const uint4 w1, const uint32_t w2, uint32_t * arow) {
    const uint32_t ws[9] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[4] >> 16));
    const uint32_t q0[4] = { (ws[0] >> 16) | (ws[1] << 16), (ws[1] >> 16) | (ws[2] << 16), (ws[2] >> 16) | (ws[3] << 16), (ws[3] >> 16) | (ws[4] << 16) };
    const uint32_t q1[4] = { ws[5], ws[6], ws[7], ws[8] };
    // kv + 128 = {1,24,45,63,79,93,106,118,129,141,153,166,181,197,217,241} packed little-endian, 4 per dword
    const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; const float md = -128.0f * d; const uint32_t * q = blk ? q1 : q0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v = q[w];
            const uint32_t nib[2] = { v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu };
            float x[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t n = nib[h];
                const uint32_t sel = n & 0x07070707u;
                const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel);   // entries 0..7
                const uint32_t pB = __builtin_amdgcn_perm(L3, L2, sel);   // entries 8..15
                const uint32_t m  = ((n >> 3) & 0x01010101u) * 0xFFu;      // 0xFF where the nibble >= 8
                const uint32_t u  = (pA & ~m) | (pB & m);
                x[h][0] = fmaf((float)(u & 0xFFu), d, md);
                x[h][1] = fmaf((float)((u >> 8) & 0xFFu), d, md);
                x[h][2] = fmaf((float)((u >> 16) & 0xFFu), d, md);
                x[h][3] = fmaf((float)(u >> 24), d, md);
            }
            out[2*w] = mmb_pack2(x[0][0], x[0][1]); out[2*w + 1] = mmb_pack2(x[0][2], x[0][3]);
            out[8 + 2*w] = mmb_pack2(x[1][0], x[1][1]); out[8 + 2*w + 1] = mmb_pack2(x[1][2], x[1][3]);
        }
    }
}

// dequantize one weight row's two consecutive Q8_0 blocks (68 bytes: d0 qs0[32] d1 qs1[32]) into 64 bf16 in LDS
__device__ __forceinline__ void mmb_dq_row68(const uint4 w0, const uint4 w1, const uint4 w2, const uint4 w3, const uint32_t w4, uint32_t * arow) {
    const uint32_t ws[17] = {w0.x,w0.y,w0.z,w0.w, w1.x,w1.y,w1.z,w1.w, w2.x,w2.y,w2.z,w2.w, w3.x,w3.y,w3.z,w3.w, w4};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[8] >> 16));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = blk ? ws[9 + w] : ((ws[w] >> 16) | (ws[w + 1] << 16));
            const float e0 = d * (float)(int8_t)(v      ), e1 = d * (float)(int8_t)(v >>  8);
            const float e2 = d * (float)(int8_t)(v >> 16), e3 = d * (float)(int8_t)(v >> 24);
            out[2*w] = mmb_pack2(e0, e1); out[2*w + 1] = mmb_pack2(e2, e3);
        }
    }
}

// strixllama: dequantize one weight row's two consecutive Q5_1 blocks (48 bytes: d0 m0 qh0 qs0[16] d1 m1 qh1 qs1[16])
// into 64 bf16 in LDS. Weight j of a block is (qs[j] & 15 | bit j of qh << 4) * d + m and weight j + 16 is
// (qs[j] >> 4 | bit j + 16 of qh << 4) * d + m, as dequantize_row_q5_1 (one rounding: an fma). Four high bits go to
// bit 4 of four bytes with one multiply: b * 0x00204081 puts bit i of b at bit 8i, nothing overlapping.
__device__ __forceinline__ void mmb_dq_row48(const uint4 w0, const uint4 w1, const uint4 w2, uint32_t * arow) {
    const uint32_t ws[12] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const uint32_t * b = ws + 6 * blk;
        const float d = mmb_h2f((uint16_t)(b[0] & 0xffff)), m = mmb_h2f((uint16_t)(b[0] >> 16));
        const uint32_t qh = b[1];
        uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v  = b[2 + w];
            const uint32_t h0 = ((((qh >> (4 * w)) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
            const uint32_t h1 = ((((qh >> (16 + 4 * w)) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
            const uint32_t x0 = (v & 0x0f0f0f0fu) | h0, x1 = ((v >> 4) & 0x0f0f0f0fu) | h1;
            float a[4], c[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                a[k] = fmaf((float)((x0 >> (8 * k)) & 0xffu), d, m);
                c[k] = fmaf((float)((x1 >> (8 * k)) & 0xffu), d, m);
            }
            out[2*w] = mmb_pack2(a[0], a[1]); out[2*w + 1] = mmb_pack2(a[2], a[3]);
            out[8 + 2*w] = mmb_pack2(c[0], c[1]); out[8 + 2*w + 1] = mmb_pack2(c[2], c[3]);
        }
    }
}

// strixllama: dequantize 64 weights of one IQ3_S row into 64 bf16 in LDS.
//
// Unlike IQ4_NL and Q8_0, whose 64 weights are two whole 32-weight blocks lying in 36 / 68
// contiguous bytes, IQ3_S is a 256-weight superblock of 110 bytes
//     d(2) qs[64] qh[8] signs[32] scales[4]
// so a 64-weight step is a QUARTER of one superblock and its inputs are five scattered pieces.
// For quarter q that is ib = 2q and 2q+1 in the reference's indexing, hence 16 qs bytes at 2+16q,
// 2 qh bytes at 66+2q, 8 sign bytes at 74+8q, and ONE scales byte at 106+q whose low nibble scales
// the first half and whose high nibble scales the second.
//
// The values themselves come from iq3s_grid, a 512-entry table of four packed uint8 each, indexed
// by a qs byte plus one bit lifted out of qh; signs are one bit per weight. Same arithmetic as
// dequantize_iq3_s in dequantize.cuh, just emitting the LDS bf16 pairs the MMB tile wants.
__device__ __forceinline__ void mmb_dq_row_iq3s(const uint8_t * __restrict__ p, const int q, uint32_t * arow) {
    const float d0 = mmb_h2f(*(const uint16_t *) p);
    const uint8_t * qs = p + 2  + 16 * q;
    const uint8_t * qh = p + 66 + 2  * q;
    const uint8_t * sg = p + 74 + 8  * q;
    const uint32_t  sc = p[106 + q];
#pragma unroll
    for (int h = 0; h < 2; ++h) {                       // h picks ib = 2q + h: 32 weights
        const float d = d0 * (float) (1 + 2 * ((sc >> (4 * h)) & 0xf));
        const uint32_t qhb = qh[h];
        const uint8_t * qsb = qs + 8 * h;
#pragma unroll
        for (int il = 0; il < 4; ++il) {                // il picks 8 weights: 4 from each grid entry
            const uint32_t g1 = iq3s_grid[qsb[2 * il + 0] | ((qhb << (8 - 2 * il)) & 256)];
            const uint32_t g2 = iq3s_grid[qsb[2 * il + 1] | ((qhb << (7 - 2 * il)) & 256)];
            const uint32_t s  = sg[4 * h + il];
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float a = d * (float) ((g1 >> (8 * j)) & 0xff);
                const float b = d * (float) ((g2 >> (8 * j)) & 0xff);
                v[j]     = (s & (1u <<  j))      ? -a : a;
                v[j + 4] = (s & (1u << (j + 4))) ? -b : b;
            }
            uint32_t * out = arow + (32 * h + 8 * il) / 2;
            out[0] = mmb_pack2(v[0], v[1]); out[1] = mmb_pack2(v[2], v[3]);
            out[2] = mmb_pack2(v[4], v[5]); out[3] = mmb_pack2(v[6], v[7]);
        }
    }
}

template <typename DRowFn>
__device__ __forceinline__ void mmb_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D, uint16_t * __restrict__ Dh,
        const bool store_f32, const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[(2 * e + cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (store_f32) {
            if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; } }
        }
        if (Dh) {
            if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(mmb_pack2(v0.x, v0.y), mmb_pack2(v0.z, v0.w), mmb_pack2(v1.x, v1.y), mmb_pack2(v1.z, v1.w)); }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) Dh[base + k] = mmb_f2bf(vv[k]); } }
        }
    }
    __syncthreads();
}
// strixllama: PF - steps of weights and activations in flight in registers: 1 (the next step, loaded while this one
// computes) or 2 (the next two; a step's loads then get a whole step's compute more to land, for the routed experts of
// a prefill's few rows an expert, where a step's WMMAs are over before its loads return)
template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, int PF = 1, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm(const uint8_t * __restrict__ Wbase, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * As, uint16_t * Bs, const int ldx = 0) {
    // strixllama: ldx - the activations' row stride when K is a slice of it (split K); 0 = K
    const size_t xs = ldx > 0 ? (size_t) ldx : (size_t) K;
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    static_assert(PF == 1 || PF == 2, "one or two steps in flight");
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    struct regs_t { uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS], a5[A_ITEMS], a6[A_ITEMS], a7[A_ITEMS], a8[A_ITEMS]; uint32_t a2[A_ITEMS]; uint4 bst[B_ITEMS]; };
    regs_t R0, R1;
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }

    auto load_regs = [&](const int ks, regs_t & r) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    r.a0[i] = *(const uint4 *)(p); r.a1[i] = *(const uint4 *)(p + 16); r.a2[i] = *(const uint32_t *)(p + 32); }
                else if constexpr (WTYPE == 1) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    r.a0[i] = *(const uint4 *)(p); r.a1[i] = *(const uint4 *)(p + 16); r.a3[i] = *(const uint4 *)(p + 32); r.a4[i] = *(const uint4 *)(p + 48); r.a2[i] = *(const uint32_t *)(p + 64); }
                else if constexpr (WTYPE == 4) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 48;
                    r.a0[i] = *(const uint4 *)(p); r.a1[i] = *(const uint4 *)(p + 16); r.a3[i] = *(const uint4 *)(p + 32); }
                else { const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 128);
                    r.a0[i] = p[0]; r.a1[i] = p[1]; r.a3[i] = p[2]; r.a4[i] = p[3]; r.a5[i] = p[4]; r.a6[i] = p[5]; r.a7[i] = p[6]; r.a8[i] = p[7]; }
            } else { r.a0[i] = make_uint4(0,0,0,0); r.a1[i] = make_uint4(0,0,0,0); r.a3[i] = make_uint4(0,0,0,0); r.a4[i] = make_uint4(0,0,0,0); r.a5[i] = r.a6[i] = r.a7[i] = r.a8[i] = make_uint4(0,0,0,0); r.a2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            r.bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * xs + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&](const regs_t & r) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = tid + i * MMB_NT; if (row < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(r.a0[i], r.a1[i], r.a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 1) mmb_dq_row68(r.a0[i], r.a1[i], r.a3[i], r.a4[i], r.a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 4) mmb_dq_row48(r.a0[i], r.a1[i], r.a3[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else { uint4 * d = (uint4 *)(As + row * MMB_LDS_STRIDE); d[0] = r.a0[i]; d[1] = r.a1[i]; d[2] = r.a3[i]; d[3] = r.a4[i]; d[4] = r.a5[i]; d[5] = r.a6[i]; d[6] = r.a7[i]; d[7] = r.a8[i]; } } }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = r.bst[i]; }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;   // whole fragments past the valid columns are never stored
    const int nks = K / MMB_BK;
    auto compute = [&]() __attribute__((always_inline)) {
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]); }
        }
    };
    if constexpr (PF == 1) {
        load_regs(0, R0); store_lds(R0); mmb_lds_sync();
        for (int ks = 0; ks < nks; ++ks) {
            if (ks + 1 < nks) load_regs(ks + 1, R0);
            __builtin_amdgcn_sched_barrier(0);   // the loads' consumers stay after the WMMAs
            compute();
            mmb_lds_sync();
            __builtin_amdgcn_sched_barrier(0);
            if (ks + 1 < nks) store_lds(R0);
            mmb_lds_sync();
        }
    } else {
        // two steps in flight: step ks + 1 was loaded an iteration ago, ks + 2 is loaded now; the register sets alternate,
        // the loop unrolled by two so the compiler sees which set each use means
        load_regs(0, R0); if (nks > 1) load_regs(1, R1); store_lds(R0); mmb_lds_sync();
        for (int ks = 0; ks < nks; ks += 2) {
            if (ks + 2 < nks) load_regs(ks + 2, R0);
            __builtin_amdgcn_sched_barrier(0);
            compute();
            mmb_lds_sync();
            __builtin_amdgcn_sched_barrier(0);
            if (ks + 1 < nks) store_lds(R1);
            mmb_lds_sync();
            if (ks + 1 >= nks) break;
            if (ks + 3 < nks) load_regs(ks + 3, R1);
            __builtin_amdgcn_sched_barrier(0);
            compute();
            mmb_lds_sync();
            __builtin_amdgcn_sched_barrier(0);
            if (ks + 2 < nks) store_lds(R0);
            mmb_lds_sync();
        }
    }
    // epilogue through LDS (per-wave 1 KB stage in the now-free A/B tile area); tiles are 16 rows, a_rows is a
    // multiple of 32 in this model, so whole tiles are either valid or beyond a_rows
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : As) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) mmb_store_tile(acc[i][j], stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, const int K, const int T, const int ldx = 0) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = WTYPE == 2 ? (size_t) K * 2 : (size_t)(K / 32) * (WTYPE == 0 ? 18 : 34);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, false>(W + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, D, Dh, store_f32, M, [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, m0, T - t0, As, Bs, ldx);
}

// strixllama: the GDN output gate's projection z = W x [K -> heads * 128] with the gated RMS norm that is z's only
// reader as its epilogue: out = (scale * o * g) * sigmoid(z), scale = rsqrt(mean(o^2) + eps) over each head's 128
// values of the delta net's output o. The main loop is mmb_tile_gemm's (WTYPE 0 IQ4_NL, 1 Q8_0) and the epilogue is
// rms_rows_f32<true>'s arithmetic in its order, so out is bitwise what the GEMM followed by ggml_cuda_op_norm_gated
// write - without z going to memory and back (400 MB a layer at 8K tokens). A 128-row tile is one head; z passes
// through LDS a 16-token slice at a time, and each wave then normalises whole token rows of the head.
__device__ __forceinline__ float mmb_xor_tree(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor(v, off); }
    return v;
}
// the square as its own rounded product: left to the compiler, xi * xi fused into the tree's first add here (one
// rounding fewer than rms_rows_f32, where it does not)
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float mmb_sq_rn(const float a) { float r; asm("v_mul_f32_e32 %0, %1, %1" : "=v"(r) : "v"(a)); return r; }
#else
__device__ __forceinline__ float mmb_sq_rn(const float a) { return __fmul_rn(a, a); }
#endif
template <int WTYPE>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_dense_gnorm_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, const int M, const int K, const int T, const int ldx,
        const float * __restrict__ O, const float * __restrict__ G, float * __restrict__ Out,
        const int64_t nrows, const int64_t nchannels, const int64_t s_row, const int64_t s_ch, const int64_t s_sample, const float eps, const int ncols) {
    constexpr int BM = 128, BN = 256, WTM = 64, WTN = 64;
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    static_assert(WTYPE == 0 || WTYPE == 1, "IQ4_NL or Q8_0");
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = (size_t)(K / 32) * (WTYPE == 0 ? 18 : 34);
    const uint8_t * Wbase = W + (size_t) m0 * wrow_bytes;
    const int a_rows = M - m0;
    const size_t xs = ldx > 0 ? (size_t) ldx : (size_t) K;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS]; uint32_t a2[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int t = t0 + (c >> 3); brow[i] = t < T ? t : -1; }

    auto load_regs = [&](const int ks) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a2[i] = *(const uint32_t *)(p + 32); }
                else { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a3[i] = *(const uint4 *)(p + 32); a4[i] = *(const uint4 *)(p + 48); a2[i] = *(const uint32_t *)(p + 64); }
            } else { a0[i] = make_uint4(0,0,0,0); a1[i] = make_uint4(0,0,0,0); a3[i] = make_uint4(0,0,0,0); a4[i] = make_uint4(0,0,0,0); a2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * xs + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = tid + i * MMB_NT; if (row < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(a0[i], a1[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else                      mmb_dq_row68(a0[i], a1[i], a3[i], a4[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE)); } }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]);
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }

    // epilogue: lane holds channel wm*64 + i*16 + (lane & 15) of the head and token wn*64 + j*16 + 2e + (lane >> 4)
    constexpr int LD = 132;                      // z slice in LDS: [4 token groups x 16 tokens][128 channels], padded
    static_assert(4 * 16 * LD * sizeof(float) <= BN * MMB_LDS_STRIDE * sizeof(uint16_t), "the z slice fits in Bs");
    float * zs = (float *) Bs;
    const int64_t heads = M / 128, head = m0 / 128;
    float gw[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) gw[q] = G[32 * q + lane];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        if (j > 0) __syncthreads();
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) zs[(wn * 16 + 2 * e + (lane >> 4)) * LD + wm * WTM + i * 16 + (lane & 15)] = acc[i][j][e];
        __syncthreads();
        // the slice's 64 tokens, 8 to a wave, each a whole row of the head as rms_rows_f32 takes it
        for (int q8 = 0; q8 < 8; ++q8) {
            const int slot = wave * 8 + q8, tg = slot >> 4, tl = slot & 15;
            const int t = t0 + tg * WTN + j * 16 + tl;
            if (t >= T) continue;
            const int64_t g = (int64_t) t * heads + head;
            const int64_t row = g % nrows, channel = (g / nrows) % nchannels, sample = g / (nrows * nchannels);
            const float * x = O + sample * s_sample + channel * s_ch + row * s_row;
            float part[8];
#pragma unroll
            for (int wv = 0; wv < 8; ++wv) {
                const int col = 32 * wv + lane;
                const float xi = col < 128 ? x[col] : 0.0f;
                part[wv] = mmb_xor_tree(mmb_sq_rn(xi));
            }
            float v = 0.0f;
#pragma unroll
            for (int wv = 0; wv < 8; ++wv) { v = lane == wv ? part[wv] : v; }
            const float tmp = mmb_xor_tree(v);
            const float mean = tmp / ncols;   // ncols = 128 at run time: the division rms_rows_f32 does, not a product
            const float scale = rsqrtf(mean + eps);
            float * dst = Out + g * 128;
#pragma unroll
            for (int wv = 0; wv < 4; ++wv) {
                const int col = 32 * wv + lane;
                const float tt = scale * x[col] * gw[wv];
                const float s = 1.0f / (1.0f + expf(-zs[slot * LD + col]));
                dst[col] = tt * s;
            }
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif
__device__ __forceinline__ float gm_sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float gm_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }

// strixllama: WTYPE 0 is IQ4_NL (the author's file), 1 is Q8_0 (Unsloth's, whose HC weights are Q8_0). XF32 reads
// the normalised streams as F32 and applies the sigmoid to the F32 accumulator, which is exactly what the unfused
// MMB GEMM + hc_mix_reduce do when the BF16 intermediates (LLAMA_MMB_HC16) are off - so it is bitwise the same
// result, only without writing the [hc*E, T] gate out and reading it back.
// XRES (strixllama): the F32 streams formed from the combine's F32 residual output Res, its per-row scale Rs and gamma
// Gm - x * scale * gamma, the combine's own two products in its order - instead of read back from a copy it wrote
template <int HC, int WTYPE = 0, bool XF32 = false, bool XRES = false>
__global__ void __launch_bounds__(MMB_NT, 2)
hc_gate_mix_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Lo, const uint16_t * __restrict__ Xn,
        const float * __restrict__ XnF, float * __restrict__ Out,
        uint16_t * __restrict__ OutH, const bool store_f32,
        const int E, const int K, const int T, const float scale, const float bias, const int64_t xn_ld = 0,
        const float * __restrict__ Res = nullptr, const float * __restrict__ Rs = nullptr, const float * __restrict__ Gm = nullptr) {
    constexpr int CH = 32, BN = 128, BM = HC * CH;
    static_assert(BM <= MMB_NT, "one A row per thread");
    static_assert(WTYPE == 0 || WTYPE == 1, "IQ4_NL or Q8_0");
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 1, wn = wave >> 1;                 // wave: 16 channels (all HC streams) x 32 tokens
    const int e0 = blockIdx.x * CH, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = (size_t)(K / 32) * (WTYPE == 0 ? 18 : 34);
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    uint4 a0 = make_uint4(0,0,0,0), a1 = make_uint4(0,0,0,0), a3 = make_uint4(0,0,0,0), a4 = make_uint4(0,0,0,0); uint32_t a2 = 0;
    uint4 bst[B_ITEMS]; int brow[B_ITEMS];
    const uint8_t * arow = W;
    if (tid < BM) { const int c = tid / CH, i = tid - c * CH; arow = W + (size_t)(c * E + e0 + i) * wrow_bytes; }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int t = t0 + (c >> 3); brow[i] = t < T ? t : -1; }
    auto load_regs = [&](const int ks) {
        if (tid < BM) {
            if constexpr (WTYPE == 0) { const uint8_t * p = arow + (size_t)ks * 36; a0 = *(const uint4 *)(p); a1 = *(const uint4 *)(p + 16); a2 = *(const uint32_t *)(p + 32); }
            else { const uint8_t * p = arow + (size_t)ks * 68;
                   a0 = *(const uint4 *)(p); a1 = *(const uint4 *)(p + 16); a3 = *(const uint4 *)(p + 32); a4 = *(const uint4 *)(p + 48); a2 = *(const uint32_t *)(p + 64); }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Lo + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0); }
    };
    auto store_lds = [&]() {
        if (tid < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(a0, a1, a2, (uint32_t *)(As + tid * MMB_LDS_STRIDE));
            else                      mmb_dq_row68(a0, a1, a3, a4, a2, (uint32_t *)(As + tid * MMB_LDS_STRIDE));
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f acc[HC][2];
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[c][j][e] = 0.f;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[HC], b[2]; const int r = lane & 15;
#pragma unroll
            for (int c = 0; c < HC; ++c) { const uint16_t * p = As + (c * CH + wm * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[c] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < 2; ++j) { const uint16_t * p = Bs + (wn * 32 + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int c = 0; c < HC; ++c)
#pragma unroll
                for (int j = 0; j < 2; ++j) acc[c][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[c], acc[c][j]);
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    // epilogue: lane holds channel (lane & 15) of the wave's 16 and tokens 2e + (lane >> 4) of each 16-token fragment
    const int cm = lane & 15, cn = lane >> 4; const int ch = e0 + wm * 16 + cm;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = t0 + wn * 32 + j * 16 + 2 * e + cn;
            if (t >= T) continue;
            float s = 0.f;
            if constexpr (XRES) {
                const float * rr = Res + (size_t)t * ((size_t)HC * E) + ch;
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float v    = gm_mul_rn(gm_mul_rn(Rs[(size_t)t * HC + c], rr[(size_t)c * E]), Gm[(size_t)c * E + ch]);
                    const float term = gm_mul_rn(v, gm_sigmoid(acc[c][j][e]));
                    s = (c == 0) ? term : gm_add_rn(s, term);
                }
            } else if constexpr (XF32) {
                const float * xr = XnF + (size_t)t * ((size_t)HC * E) + ch;
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float term = gm_mul_rn(xr[(size_t)c * E], gm_sigmoid(acc[c][j][e]));
                    s = (c == 0) ? term : gm_add_rn(s, term);
                }
            } else {
                const uint16_t * xr = Xn + (size_t)t * (xn_ld ? (size_t) xn_ld : (size_t)HC * E) + ch;
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float g = __uint_as_float(((uint32_t) mmb_f2bf(acc[c][j][e])) << 16);   // the gate GEMM's BF16 epilogue rounding
                    const float term = gm_mul_rn(gm_bf2f(xr[(size_t)c * E]), gm_sigmoid(g));
                    s = (c == 0) ? term : gm_add_rn(s, term);
                }
            }
            const float o = scale * s + bias;
            if (store_f32) Out[(size_t)t * E + ch] = o;
            if (OutH) OutH[(size_t)t * E + ch] = mmb_f2bf(o);
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0, int PF = 1>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;   // uniform across the block, before any barrier
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = (size_t)(K / 32) * (WTYPE == 4 ? 24 : WTYPE == 1 ? 34 : 18);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, true, PF>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs);
}

// WT selects the weight encoding: 0 = IQ4_NL (two 18-byte blocks per 64-weight step),
// 3 = IQ3_S (a quarter of a 110-byte 256-weight superblock). gate and up always share a type.
template <int BM, int BN, int WTM, int WTN, bool TAIL, int WT, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 g0[A_ITEMS], g1[A_ITEMS], u0[A_ITEMS], u1[A_ITEMS]; uint32_t g2[A_ITEMS], u2[A_ITEMS];
    // IQ3_S: prefetch the 29 bytes a quarter-superblock needs into registers in load_regs, the same
    // way the IQ4_NL path does, so the global latency is hidden behind the previous tile's math.
    // The offsets are not dword aligned, so these are byte loads the compiler coalesces itself.
    struct iq3s_regs { uint16_t d; uint8_t qs[16], qh[2], sg[8], sc; };
    iq3s_regs r3g[A_ITEMS], r3u[A_ITEMS]; bool ok3[A_ITEMS]; int q3 = 0;
    auto fetch3 = [](const uint8_t * __restrict__ p, const int q, iq3s_regs & r) {
        r.d = *(const uint16_t *) p;
        const uint8_t * qs = p + 2 + 16 * q;
#pragma unroll
        for (int z = 0; z < 16; ++z) { r.qs[z] = qs[z]; }
        r.qh[0] = p[66 + 2 * q]; r.qh[1] = p[67 + 2 * q];
        const uint8_t * sg = p + 74 + 8 * q;
#pragma unroll
        for (int z = 0; z < 8; ++z) { r.sg[z] = sg[z]; }
        r.sc = p[106 + q];
    };
    auto dq3 = [](const iq3s_regs & r, uint32_t * arow) {
        const float d0 = mmb_h2f(r.d);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float d = d0 * (float) (1 + 2 * ((r.sc >> (4 * h)) & 0xf));
            const uint32_t qhb = r.qh[h];
#pragma unroll
            for (int il = 0; il < 4; ++il) {
                const uint32_t g1 = iq3s_grid[r.qs[8 * h + 2 * il + 0] | ((qhb << (8 - 2 * il)) & 256)];
                const uint32_t g2 = iq3s_grid[r.qs[8 * h + 2 * il + 1] | ((qhb << (7 - 2 * il)) & 256)];
                const uint32_t s  = r.sg[4 * h + il];
                float v[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float a = d * (float) ((g1 >> (8 * j)) & 0xff);
                    const float b = d * (float) ((g2 >> (8 * j)) & 0xff);
                    v[j]     = (s & (1u <<  j))      ? -a : a;
                    v[j + 4] = (s & (1u << (j + 4))) ? -b : b;
                }
                uint32_t * out = arow + (32 * h + 8 * il) / 2;
                out[0] = mmb_pack2(v[0], v[1]); out[1] = mmb_pack2(v[2], v[3]);
                out[2] = mmb_pack2(v[4], v[5]); out[3] = mmb_pack2(v[6], v[7]);
            }
        }
    };
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    auto load_regs = [&](const int ks) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WT == 3) {
                    const size_t sb = (size_t)(ks >> 2) * 110;
                    q3 = ks & 3;
                    fetch3(Wg + (size_t)row * wrow_bytes + sb, q3, r3g[i]);
                    fetch3(Wu + (size_t)row * wrow_bytes + sb, q3, r3u[i]);
                    ok3[i] = true;
                } else {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g2[i] = *(const uint32_t *)(pg + 32);
                    u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u2[i] = *(const uint32_t *)(pu + 32);
                }
            } else if constexpr (WT == 3) { ok3[i] = false; }
            else { g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); g2[i] = u2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = tid + i * MMB_NT; if (row < BM) {
            if constexpr (WT == 3) {
                uint32_t * ag = (uint32_t *)(Ag + row * MMB_LDS_STRIDE), * au = (uint32_t *)(Au + row * MMB_LDS_STRIDE);
                if (ok3[i]) { dq3(r3g[i], ag); dq3(r3u[i], au); }
                else {
#pragma unroll
                    for (int z = 0; z < MMB_BK / 2; ++z) { ag[z] = 0; au[z] = 0; }
                }
            } else {
                mmb_dq_row36(g0[i], g1[i], g2[i], (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                mmb_dq_row36(u0[i], u1[i], u2[i], (uint32_t *)(Au + row * MMB_LDS_STRIDE));
            } } }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s ag[TM], au[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                ag[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                au[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (TAIL) { if (!jact[j]) continue; }
                    accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], ag[i], accg[i][j]);
                    accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], au[i], accu[i][j]);
                }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) {
                v8f v;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(accg[i][j][e]) * accu[i][j][e]; }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WT = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = (WT == 3) ? (size_t)(K / 256) * 110 : (size_t)(K / 32) * 18;
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, true, WT>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs);
}

// strixllama: the IQ3_S gate/up tile. BM = 64 and the step's dequantization is spread over the whole block, four
// threads per row with 16 of its 64 weights each (one thread per row left six of the eight waves waiting at the
// barrier). On RDNA3.5 the VALU and the WMMA unit do not run at the same time on a SIMD, so every instruction of
// the dequantization costs WMMA time; the rest is about spending fewer of them and never waiting on a load:
// - weight = bf16(d * (+-g)): g and -g are exact in F16 and d * g is exact in F32 (at most 20 significant bits),
//   so the sign goes onto the F16 grid value (an LDS copy of iq3s_grid with each byte as an F16) and the product
//   is one v_fma_mix: the same bits as bf16(+-(d * (float) g)) with no byte convert and no compare/select;
// - a thread's raw bytes stay whole in registers until it dequantizes them after the step's WMMAs: split into
//   bytes at the load, they made every step wait for its weight loads before the first WMMA;
// - the loads address the tile from a uniform base (the superblock of row 0, the step's column 0 of the
//   activations) with 32-bit per-thread offsets, and the prefetch registers are native vectors (a HIP uint4 array
//   here was not promoted to registers and went through scratch);
// - a column past the expert's tokens reads token 0: finite values in a column that is never stored, instead of
//   a branch and a zero fill per load;
// - the step loop is instantiated per count of this wave's valid fragments (a uniform branch outside the loop,
//   where a test per fragment branched around every WMMA pair), and a scheduling barrier keeps one 16-deep slice
//   of fragments in registers at a time.
// Each output element is the same WMMA chain over K as in mmb_tile_gemm_glu, so the results are bitwise the same.
template <int N, typename F> __device__ __forceinline__ void mmb_with_na0(const int na, F && f) {
    if (na == N) { f(std::integral_constant<int, N>{}); return; }
    if constexpr (N > 0) { mmb_with_na0<N - 1>(na, f); }
}
template <int BN, int WTM, int WTN, int WQ, int PF, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu3(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const uint32_t wrow_bytes,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int M, DRowFn drow, const int m0, const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs, const mmb_u32x2 * grid16) {
    constexpr int BM = 64, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    static_assert(BM * 4 == MMB_NT, "four threads per row of a 64-row tile");
    const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    // this thread's row, and its 16 weights of a step: half ph of the quarter superblock, grid pairs pil and pil + 1
    const int prow = tid >> 2, ph = (tid & 3) >> 1, pil = (tid & 1) * 2;
    const uint32_t ro = (uint32_t) prow * wrow_bytes;
    const uint32_t o_qs = ro + 2 + 8 * ph + 2 * pil, o_qh = ro + 66 + ph, o_sg = ro + 74 + 4 * ph + pil, o_sc = ro + 106;
    // whole registers per field: sub-dword fields packed two to a register made each load wait where it was issued
    struct raw3 { uint32_t qs, d, sg, qh, sc; };
    // strixllama: Q4_K (WQ 12). Quarter q of a superblock is sub-blocks 2q (the low nibbles of qs[32q..32q + 31]) and
    // 2q + 1 (their high nibbles), so this thread's 16 weights are one nibble of 16 bytes: half ph picks the nibble and
    // the sub-block, tid & 1 the bytes. Its scale and min (6 bits each, get_scale_min_k4) are decoded at load time.
    struct raw4 { uint4 qs; uint32_t dm; uint32_t sc[3]; uint32_t j; };   // the 12 scale bytes raw, decoded in dq4
    // strixllama: IQ4_XS (WQ 4). Quarter q of a superblock is sub-blocks ib = 2q and 2q + 1 (32 weights each: the low
    // nibbles of qs[16 ib..16 ib + 15], then their high nibbles), so half ph picks the sub-block and tid & 1 the nibble.
    // d and scales_h share a dword, scales_l is the next one; the 6-bit scale is decoded in dqx
    struct rawx { uint4 qs; uint32_t dh; uint32_t sl; uint32_t ib; };
    constexpr int SB = WQ == 12 ? 144 : WQ == 4 ? 136 : 110;   // superblock bytes
    std::conditional_t<WQ == 12, raw4, std::conditional_t<WQ == 4, rawx, raw3>> rg, ru, rg2, ru2;   // two register sets: PF 2 keeps two steps in flight
    const uint32_t k_qs = ro + 16 + 16 * (uint32_t) (tid & 1);
    auto ld4 = [&](const uint8_t * __restrict__ sb, const int q, raw4 & r) __attribute__((always_inline)) {
        r.dm = *(const uint32_t *)(sb + ro);
        const uint32_t * sc = (const uint32_t *)(sb + ro + 4);
        r.sc[0] = sc[0]; r.sc[1] = sc[1]; r.sc[2] = sc[2];
        r.j = (uint32_t) (2 * q + ph);
        r.qs = *(const uint4 *)((sb + 32 * q) + k_qs);
    };
    auto dq4 = [&](const raw4 & r, uint32_t * arow) __attribute__((always_inline)) {
        // get_scale_min_k4 on the three raw dwords (bytes 0-3, 4-7, 8-11); the byte index j is per thread (ph), so the
        // dword is picked by a condition, never by indexing the register array (that went to scratch: 2x slower)
        const int j = (int) r.j, sh = 8 * (j & 3);
        const uint32_t b0 = (r.sc[0] >> sh) & 0xffu, b1 = (r.sc[1] >> sh) & 0xffu, b2 = (r.sc[2] >> sh) & 0xffu;
        uint32_t a, m;
        if (j < 4) { a = b0 & 63; m = b1 & 63; }                                        // sc[j], sc[j + 4]
        else       { a = (b2 & 0xf) | ((b0 >> 6) << 4); m = (b2 >> 4) | ((b1 >> 6) << 4); }   // sc[j + 4], sc[j - 4], sc[j]
        const float d1 = mmb_h2f((uint16_t)(r.dm & 0xffff)) * (float) a;
        const float m1 = mmb_h2f((uint16_t)(r.dm >> 16)) * (float) m;
        const uint32_t qs[4] = {r.qs.x, r.qs.y, r.qs.z, r.qs.w};
        uint32_t * out = arow + (32 * ph + 16 * (tid & 1)) / 2;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t x = (qs[w] >> (4 * ph)) & 0x0f0f0f0fu;
            float a[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) { a[k] = fmaf((float)((x >> (8 * k)) & 0xffu), d1, -m1); }
            out[2*w] = mmb_pack2(a[0], a[1]); out[2*w + 1] = mmb_pack2(a[2], a[3]);
        }
    };
    auto ldx = [&](const uint8_t * __restrict__ sb, const int q, rawx & r) __attribute__((always_inline)) {
        r.dh = *(const uint32_t *)(sb + ro);
        r.sl = *(const uint32_t *)(sb + ro + 4);
        r.ib = (uint32_t) (2 * q + ph);
        r.qs = *(const uint4 *)(sb + ro + 8 + 32 * q + 16 * ph);
    };
    auto dqx = [&](const rawx & r, uint32_t * arow) __attribute__((always_inline)) {
        // dequantize_row_iq4_xs: dl = d * (ls - 32), weight = dl * kvalues_iq4nl[nibble]; d is F16 and ls - 32 six
        // bits, so dl is exact in F32 and so is dl * kv (24 bits at most): the one fma below rounds nothing
        const int ib = (int) r.ib;
        const int ls = (int) (((r.sl >> (4 * ib)) & 0xfu) | (((r.dh >> (16 + 2 * ib)) & 3u) << 4));
        const float dl = mmb_h2f((uint16_t)(r.dh & 0xffff)) * (float) (ls - 32);
        const float md = -128.0f * dl;
        // kv + 128 = {1,24,45,63,79,93,106,118,129,141,153,166,181,197,217,241} packed little-endian, 4 per dword
        const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
        const uint32_t qs[4] = {r.qs.x, r.qs.y, r.qs.z, r.qs.w};
        uint32_t * out = arow + (32 * ph + 16 * (tid & 1)) / 2;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t n = (qs[w] >> (4 * (tid & 1))) & 0x0f0f0f0fu;
            const uint32_t sel = n & 0x07070707u;
            const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel), pB = __builtin_amdgcn_perm(L3, L2, sel);
            const uint32_t m = ((n >> 3) & 0x01010101u) * 0xFFu;
            const uint32_t u = (pA & ~m) | (pB & m);
            float a[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) { a[k] = fmaf((float)((u >> (8 * k)) & 0xffu), dl, md); }
            out[2*w] = mmb_pack2(a[0], a[1]); out[2*w + 1] = mmb_pack2(a[2], a[3]);
        }
    };
    auto ld3 = [&](const uint8_t * __restrict__ sb, const int q, raw3 & r) __attribute__((always_inline)) {   // sb: the step's superblock of row 0
        r.d = *(const uint16_t *)(sb + ro);
        __builtin_memcpy(&r.qs, (sb + 16 * q) + o_qs, 4);
        r.qh = (sb + 2 * q)[o_qh];
        r.sg = *(const uint16_t *)((sb + 8 * q) + o_sg);
        r.sc = (sb + q)[o_sc];
    };
    auto dq3 = [&](const raw3 & r, uint32_t * arow) __attribute__((always_inline)) {
        typedef uint16_t u16x2 __attribute__((ext_vector_type(2)));
        const float d0 = mmb_h2f(r.d);
        const float d = d0 * (float) (1 + 2 * ((r.sc >> (4 * ph)) & 0xf));
        const uint32_t qhb = r.qh;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int il = pil + t;
            const mmb_u32x2 g1 = grid16[((r.qs >> (16 * t)) & 0xff) | ((qhb << (8 - 2 * il)) & 256)];
            const mmb_u32x2 g2 = grid16[((r.qs >> (16 * t + 8)) & 0xff) | ((qhb << (7 - 2 * il)) & 256)];
            const uint32_t sgn = (r.sg >> (8 * t)) & 0xff;
            const u16x2 s2 = __builtin_bit_cast(u16x2, sgn | (sgn << 15));   // lanes: the signs, and the signs >> 1
            const uint32_t gp[4] = {g1.x, g1.y, g2.x, g2.y};
            uint32_t * out = arow + (32 * ph + 8 * il) / 2;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint16_t sh = (uint16_t)(15 - 2 * q);
                const uint32_t m = __builtin_bit_cast(uint32_t, s2 << (u16x2){sh, sh});   // bits 15, 31: signs 2q, 2q + 1
                const uint32_t gs = (m & 0x80008000u) | (gp[q] & 0x7fff7fffu);
                const float a = d * (float) __builtin_bit_cast(_Float16, (uint16_t)(gs & 0xffff));
                const float b = d * (float) __builtin_bit_cast(_Float16, (uint16_t)(gs >> 16));
                out[q] = mmb_pack2(a, b);
            }
        }
    };
    uint32_t boff[B_ITEMS];   // byte offset of this thread's 16 bytes of a step from the step's column 0 of token 0
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int t = xrow(c >> 3);
        boff[i] = ((uint32_t) (t >= 0 ? t : 0) * (uint32_t) K + (uint32_t) (c & 7) * 8) * 2; }
    mmb_u32x4 bst[B_ITEMS];
    uint32_t * const agr = (uint32_t *)(Ag + prow * MMB_LDS_STRIDE), * const aur = (uint32_t *)(Au + prow * MMB_LDS_STRIDE);
    // the weights of a step (two sets: PF 2 has two steps of them in flight; the activations, L2-hot, stay one ahead -
    // a second set of those spilled the big tile)
    auto load_w = [&](const int ks, auto & rg_, auto & ru_) __attribute__((always_inline)) {
        const size_t sbo = (size_t)(ks >> 2) * SB;
        if constexpr (WQ == 12)     { ld4(Wg + sbo, ks & 3, rg_); ld4(Wu + sbo, ks & 3, ru_); }
        else if constexpr (WQ == 4) { ldx(Wg + sbo, ks & 3, rg_); ldx(Wu + sbo, ks & 3, ru_); }
        else                        { ld3(Wg + sbo, ks & 3, rg_); ld3(Wu + sbo, ks & 3, ru_); }
    };
    auto load_x = [&](const int ks) __attribute__((always_inline)) {
        const uint8_t * xb = (const uint8_t *)(Xh + ks * MMB_BK);
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { bst[i] = *(const mmb_u32x4 *)(xb + boff[i]); }
    };
    auto store_lds = [&](const auto & rg_, const auto & ru_) __attribute__((always_inline)) {
        if constexpr (WQ == 12) { dq4(rg_, agr); dq4(ru_, aur); }
        else if constexpr (WQ == 4)  { dqx(rg_, agr); dqx(ru_, aur); }
        else                         { dq3(rg_, agr); dq3(ru_, aur); }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(mmb_u32x4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    // this wave's fragments that hold valid columns: a prefix of its TN (the rest are never computed or stored)
    const int rem = n_cols - wn * WTN;
    const int na = rem <= 0 ? 0 : rem >= WTN ? TN : (rem + 15) / 16;
    const int nks = K / MMB_BK;
    mmb_with_na0<TN>(na, [&](auto nac) __attribute__((always_inline)) {
        constexpr int NA = decltype(nac)::value;
        auto compute = [&]() __attribute__((always_inline)) {
            if constexpr (NA > 0) {
#pragma unroll
                for (int kk = 0; kk < MMB_BK; kk += 16) {
                    v16s ag[TM], au[TM], b[NA]; const int r = lane & 15;
#pragma unroll
                    for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                        ag[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                        au[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)}); }
#pragma unroll
                    for (int j = 0; j < NA; ++j) { const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                        b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int j = 0; j < NA; ++j) {
                            accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], ag[i], accg[i][j]);
                            accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], au[i], accu[i][j]);
                        }
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
        };
        if constexpr (PF == 1) {
            load_w(0, rg, ru); load_x(0); store_lds(rg, ru); mmb_lds_sync();
            for (int ks = 0; ks < nks; ++ks) {
                if (ks + 1 < nks) { load_w(ks + 1, rg, ru); load_x(ks + 1); }
                __builtin_amdgcn_sched_barrier(0);   // the loads' consumers stay after the WMMAs
                compute();
                mmb_lds_sync();
                __builtin_amdgcn_sched_barrier(0);
                if (ks + 1 < nks) store_lds(rg, ru);
                mmb_lds_sync();
            }
        } else {
            // two steps of weights in flight (see mmb_tile_gemm): the sets alternate, the loop unrolled by two
            load_w(0, rg, ru); load_x(0); store_lds(rg, ru); mmb_lds_sync();
            if (nks > 1) load_w(1, rg2, ru2);
            for (int ks = 0; ks < nks; ks += 2) {
                if (ks + 2 < nks) load_w(ks + 2, rg, ru);
                if (ks + 1 < nks) load_x(ks + 1);
                __builtin_amdgcn_sched_barrier(0);
                compute();
                mmb_lds_sync();
                __builtin_amdgcn_sched_barrier(0);
                if (ks + 1 < nks) store_lds(rg2, ru2);
                mmb_lds_sync();
                if (ks + 1 >= nks) break;
                if (ks + 3 < nks) load_w(ks + 3, rg2, ru2);
                if (ks + 2 < nks) load_x(ks + 2);
                __builtin_amdgcn_sched_barrier(0);
                compute();
                mmb_lds_sync();
                __builtin_amdgcn_sched_barrier(0);
                if (ks + 2 < nks) store_lds(rg, ru);
                mmb_lds_sync();
            }
        }
    });
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < M - m0;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && j < na) {
                v8f v;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(accg[i][j][e]) * accu[i][j][e]; }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BN, int WTM, int WTN, int WQ = 3, int PF = 1>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu3_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    constexpr int BM = 64;
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    __shared__ mmb_u32x2 grid16[512];   // iq3s_grid with each byte as an F16
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    for (int i = threadIdx.x; WQ == 3 && i < 512; i += MMB_NT) {
        const uint32_t g = iq3s_grid[i];
        uint32_t h[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { h[j] = __builtin_bit_cast(uint16_t, (_Float16) (float) ((g >> (8 * j)) & 0xff)); }
        grid16[i] = (mmb_u32x2){h[0] | (h[1] << 16), h[2] | (h[3] << 16)};
    }
    __syncthreads();
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const uint32_t wrow_bytes = (uint32_t)(K / 256) * (WQ == 12 ? 144 : WQ == 4 ? 136 : 110);
    mmb_tile_gemm_glu3<BN, WTM, WTN, WQ, PF>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs, grid16);
}

// F32 x F32 -> F32 GEMM with F32-equivalent precision on WMMA: each operand is split into F16 hi + F16 lo at tile load and
// the product is accumulated as hi*hi + hi*lo + lo*hi in F32 (lo*lo ~2^-22 relative, dropped). Used for the F32 MoE router.
__device__ __forceinline__ void mmb_split2(float x, uint16_t & hi, uint16_t & lo) {
    hi = __builtin_bit_cast(uint16_t, (_Float16) x); lo = __builtin_bit_cast(uint16_t, (_Float16) (x - (float) __builtin_bit_cast(_Float16, hi)));
}
template <int BM, int BN, int WTM, int WTN, bool TWO>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_f32split_kernel(const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K, const int T) {
    constexpr int BKs = 32, LS = BKs + 8, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    __shared__ __align__(16) uint16_t Ah[BM * LS], Al[BM * LS], Bh[BN * LS], Bl[BN * LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    constexpr int A_CH = BM * BKs / 4, B_CH = BN * BKs / 4;
    for (int k0 = 0; k0 < K; k0 += BKs) {
        for (int idx = tid; idx < A_CH; idx += MMB_NT) { const int row = idx >> 3, c4 = (idx & 7) * 4; const int m = m0 + row;
            float4 v = make_float4(0.f,0.f,0.f,0.f); if (m < M) v = *(const float4 *)(W + (size_t) m * K + k0 + c4);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Ah + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Al + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
        for (int idx = tid; idx < B_CH; idx += MMB_NT) { const int row = idx >> 3, c4 = (idx & 7) * 4; const int t = t0 + row;
            float4 v = make_float4(0.f,0.f,0.f,0.f); if (t < T) v = *(const float4 *)(X + (size_t) t * K + k0 + c4);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Bh + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Bl + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
        __syncthreads();
        const int r = lane & 15;
#pragma unroll
        for (int kk = 0; kk < BKs; kk += 16) {
            v16s ah[TM], al[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + kk;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)});
                al[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Al + off), *(const uint4 *)(Al + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + kk;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)});
                bl[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bl + off), *(const uint4 *)(Bl + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], ah[i], acc[i][j]);
                    if constexpr (!TWO) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], al[i], acc[i][j]);
                }
        }
        __syncthreads();
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + cn; if (t < T) D[(size_t) t * M + m] = acc[i][j][e]; } }
}

// strixllama: the same GEMM for a narrow weight, M <= 64 - the hyper-connection inject [10240 -> 4] and the GDN beta and
// alpha [2560 -> 48], which Unsloth's files keep F32. The tile kernel above pads M to 128 rows and stages every step
// through LDS behind two barriers: the inject took 1.40 ms a call at 2030 tokens, where reading its activations once
// takes 0.36. Here a wave takes 16 columns and all of M (MT tiles of 16 rows), its operands straight from memory into
// registers, the next step's loaded while this one computes. Every output element sees the WMMA sequence it sees in the
// tile kernel - the same hi/lo split, the same 16-deep steps in the same order, the same operand roles - so the result
// is bitwise the tile kernel's (checked on 11 shapes, T 512..8192, both product counts). The inject at 2030 tokens:
// 0.61 ms, at 8192: 3.30 -> 2.69; beta/alpha at 2030: 0.36 -> 0.14. What is left is the memory channels: all waves walk
// K in step and their loads meet on the same channels (a staggered start reads ~190 GB/s instead of ~140), but a
// stagger would reorder the sums. STRIX_MMB_F32_NARROW=0 keeps the tile kernel.
typedef float mmb_f4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ v16s mmb_pack16(const uint16_t (&h)[16]) {
    uint32_t p[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = (uint32_t) h[2 * i] | ((uint32_t) h[2 * i + 1] << 16);
    return __builtin_bit_cast(v16s, (uint4[2]){make_uint4(p[0], p[1], p[2], p[3]), make_uint4(p[4], p[5], p[6], p[7])});
}
template <int MT, bool TWO, int U>
__global__ void __launch_bounds__(MMB_NT)
mmb_f32narrow_kernel(const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K, const int T) {
    // K % (16 * U) == 0. Rows past T (M) read row T-1 (M-1): output (t, m) depends only on row t of X and row m of W,
    // and those outputs are not stored
    const int lane = threadIdx.x & 31, r = lane & 15;
    const int t0 = (blockIdx.x * (MMB_NT / 32) + (threadIdx.x >> 5)) * 16;
    if (t0 >= T) return;
    const mmb_f4 * xr = (const mmb_f4 *) (X + (size_t) min(t0 + r, T - 1) * K);
    const mmb_f4 * wr[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) wr[i] = (const mmb_f4 *) (W + (size_t) min(i * 16 + r, M - 1) * K);
    v8f acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
    const int K4 = K / 4;
    mmb_f4 xv[U][4], wv[MT][U][4];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            xv[u][q] = xr[4 * u + q];
#pragma unroll
            for (int i = 0; i < MT; ++i) wv[i][u][q] = wr[i][4 * u + q];
        }
    for (int k4 = 0; k4 < K4; k4 += 4 * U) {
        const int n4 = k4 + 4 * U < K4 ? k4 + 4 * U : 0;   // the last step's loads read the start again, unused
        mmb_f4 xn[U][4], wn[MT][U][4];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                xn[u][q] = xr[n4 + 4 * u + q];
#pragma unroll
                for (int i = 0; i < MT; ++i) wn[i][u][q] = wr[i][n4 + 4 * u + q];
            }
        // a compiler-only barrier: without it the next step's loads and the first step's, the two values of xv / wv
        // at the loop head, are folded into one load of the phi of their addresses there, just before the use
        asm volatile("" ::: "memory");
#pragma unroll
        for (int u = 0; u < U; ++u) {
            uint16_t h[16], l[16];
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int c = 0; c < 4; ++c) mmb_split2(xv[u][q][c], h[4 * q + c], l[4 * q + c]);
            const v16s xh = mmb_pack16(h), xl = mmb_pack16(l);
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                uint16_t wh[16], wl[16];
#pragma unroll
                for (int q = 0; q < 4; ++q)
#pragma unroll
                    for (int c = 0; c < 4; ++c) mmb_split2(wv[i][u][q][c], wh[4 * q + c], wl[4 * q + c]);
                const v16s bwh = mmb_pack16(wh);
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xh, bwh, acc[i]);
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xl, bwh, acc[i]);
                if constexpr (!TWO) acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xh, mmb_pack16(wl), acc[i]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                xv[u][q] = xn[u][q];
#pragma unroll
                for (int i = 0; i < MT; ++i) wv[i][u][q] = wn[i][u][q];
            }
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < MT; ++i) { const int m = i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int e = 0; e < 8; ++e) { const int t = t0 + 2 * e + cn; if (t < T) D[(size_t) t * M + m] = acc[i][e]; } }
}
template <bool TWO, int U>
static void mmb_f32narrow(const float * W, const float * X, float * D, const int M, const int K, const int T, cudaStream_t stream) {
    const dim3 grid((T + MMB_NT / 2 - 1) / (MMB_NT / 2));   // a wave per 16 columns
    if      (M <= 16) mmb_f32narrow_kernel<1, TWO, U><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
    else if (M <= 32) mmb_f32narrow_kernel<2, TWO, U><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
    else if (M <= 48) mmb_f32narrow_kernel<3, TWO, U><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
    else              mmb_f32narrow_kernel<4, TWO, U><<<grid, MMB_NT, 0, stream>>>(W, X, D, M, K, T);
}

// strixllama: the same F32 GEMM in two-product mode (LLAMA_MMB_F32SPLIT=2) on a 64 x 128 tile - the MoE router
// [2560 -> 512] took 2.0 ms a call at 8150 tokens on the tile kernel above, 1.4 on this one. Every output element sees
// the WMMA sequence it sees there - x_hi * w_hi then x_lo * w_hi per 16-deep step, K in order, the same operand roles -
// so the result is bitwise the tile kernel's. What differs: the weight is converted to F16 only (the two-product mode
// never reads its low half), and the next 64-deep step's operands are loaded into registers while this one computes.
// K % BKs == 0. STRIX_MMB_F32_TILE2=0 keeps the tile kernel
template <int BM, int BN, int WTM, int WTN, int BKs>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_f32split2_kernel(const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, const int M, const int K, const int T) {
    constexpr int LS = BKs + 8, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_PER = BM * BKs / 4 / MMB_NT, B_PER = BN * BKs / 4 / MMB_NT, C4 = BKs / 4;
    static_assert(A_PER * MMB_NT * 4 == BM * BKs && B_PER * MMB_NT * 4 == BN * BKs, "whole float4s per thread");
    static_assert(WAVES_M * (BN / WTN) == MMB_NT / 32, "8 waves");
    __shared__ __align__(16) uint16_t Ah[BM * LS], Bh[BN * LS], Bl[BN * LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    float4 ra[A_PER], rb[B_PER];
    auto load = [&](const int k0) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) { const int idx = tid + p * MMB_NT, row = idx / C4, c4 = (idx % C4) * 4, m = m0 + row;
            ra[p] = m < M ? *(const float4 *)(W + (size_t) m * K + k0 + c4) : make_float4(0.f, 0.f, 0.f, 0.f); }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) { const int idx = tid + p * MMB_NT, row = idx / C4, c4 = (idx % C4) * 4, t = t0 + row;
            rb[p] = t < T ? *(const float4 *)(X + (size_t) t * K + k0 + c4) : make_float4(0.f, 0.f, 0.f, 0.f); }
    };
    auto store = [&]() {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) { const int idx = tid + p * MMB_NT, row = idx / C4, c4 = (idx % C4) * 4; const float4 v = ra[p];
            const uint32_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y);
            const uint32_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w);
            *(uint2 *)(Ah + row * LS + c4) = make_uint2(h0 | (h1 << 16), h2 | (h3 << 16)); }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) { const int idx = tid + p * MMB_NT, row = idx / C4, c4 = (idx % C4) * 4; const float4 v = rb[p];
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Bh + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Bl + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
    };
    load(0); store(); __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BKs) {
        if (k0 + BKs < K) load(k0 + BKs);
        const int r = lane & 15;
#pragma unroll
        for (int kk = 0; kk < BKs; kk += 16) {
            v16s ah[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + kk;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + kk;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)});
                bl[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bl + off), *(const uint4 *)(Bl + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], ah[i], acc[i][j]);
                }
        }
        __syncthreads();
        if (k0 + BKs < K) store();
        __syncthreads();
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + cn; if (t < T) D[(size_t) t * M + m] = acc[i][j][e]; } }
}

// strixllama: the narrow kernel for one or two weights in two-product mode. Tiles [0, MT1) are rows of W1, [MT1, MT1 +
// MT2) rows of W2, both against the same activations - the GDN beta and alpha [2560 -> 48] read them once instead of
// twice. WH: the weights come as F16 copies (mmb_f16w), which are exactly the high halves the split takes of them, so
// nothing is split or read twice. Each output sees the narrow kernel's WMMA sequence: bitwise its result. At 8150
// tokens beta + alpha took 0.83 + 0.84 ms, one pass with the F16 copies 0.65
template <int MT1, int MT2, int U, bool WH>
__global__ void __launch_bounds__(MMB_NT)
mmb_f32narrow2_kernel(const float * __restrict__ W1, const float * __restrict__ W2, const _Float16 * __restrict__ H1, const _Float16 * __restrict__ H2,
        const float * __restrict__ X, float * __restrict__ D1, float * __restrict__ D2, const int M1, const int M2, const int K, const int T) {
    constexpr int MT = MT1 + MT2;
    const int lane = threadIdx.x & 31, r = lane & 15;
    const int t0 = (blockIdx.x * (MMB_NT / 32) + (threadIdx.x >> 5)) * 16;
    if (t0 >= T) return;
    const mmb_f4 * xr = (const mmb_f4 *) (X + (size_t) min(t0 + r, T - 1) * K);
    const mmb_f4 * wr[MT]; const uint2 * hr[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const bool second = i >= MT1; const int row = min((second ? i - MT1 : i) * 16 + r, (second ? M2 : M1) - 1);
        if constexpr (WH) { hr[i] = (const uint2 *) ((second ? H2 : H1) + (size_t) row * K); wr[i] = nullptr; }
        else              { wr[i] = (const mmb_f4 *) ((second ? W2 : W1) + (size_t) row * K); hr[i] = nullptr; }
    }
    v8f acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
    const int K4 = K / 4;
    mmb_f4 xv[U][4], wv[WH ? 1 : MT][U][4]; uint2 hv[WH ? MT : 1][U][4];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            xv[u][q] = xr[4 * u + q];
#pragma unroll
            for (int i = 0; i < MT; ++i) { if constexpr (WH) hv[i][u][q] = hr[i][4 * u + q]; else wv[i][u][q] = wr[i][4 * u + q]; }
        }
    for (int k4 = 0; k4 < K4; k4 += 4 * U) {
        const int n4 = k4 + 4 * U < K4 ? k4 + 4 * U : 0;   // the last step's loads read the start again, unused
        mmb_f4 xn[U][4], wn[WH ? 1 : MT][U][4]; uint2 hn[WH ? MT : 1][U][4];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                xn[u][q] = xr[n4 + 4 * u + q];
#pragma unroll
                for (int i = 0; i < MT; ++i) { if constexpr (WH) hn[i][u][q] = hr[i][n4 + 4 * u + q]; else wn[i][u][q] = wr[i][n4 + 4 * u + q]; }
            }
        asm volatile("" ::: "memory");   // see mmb_f32narrow_kernel
#pragma unroll
        for (int u = 0; u < U; ++u) {
            uint16_t h[16], l[16];
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int c = 0; c < 4; ++c) mmb_split2(xv[u][q][c], h[4 * q + c], l[4 * q + c]);
            const v16s xh = mmb_pack16(h), xl = mmb_pack16(l);
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                v16s bwh;
                if constexpr (WH) {
                    bwh = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(hv[i][u][0].x, hv[i][u][0].y, hv[i][u][1].x, hv[i][u][1].y),
                                                              make_uint4(hv[i][u][2].x, hv[i][u][2].y, hv[i][u][3].x, hv[i][u][3].y)});
                } else {
                    uint16_t wh[16], wl[16];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
#pragma unroll
                        for (int c = 0; c < 4; ++c) mmb_split2(wv[i][u][q][c], wh[4 * q + c], wl[4 * q + c]);
                    bwh = mmb_pack16(wh);
                }
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xh, bwh, acc[i]);
                acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xl, bwh, acc[i]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                xv[u][q] = xn[u][q];
#pragma unroll
                for (int i = 0; i < MT; ++i) { if constexpr (WH) hv[i][u][q] = hn[i][u][q]; else wv[i][u][q] = wn[i][u][q]; }
            }
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const bool second = i >= MT1; const int m = (second ? i - MT1 : i) * 16 + cm; const int Mi = second ? M2 : M1; float * Di = second ? D2 : D1;
        if (m >= Mi) continue;
#pragma unroll
        for (int e = 0; e < 8; ++e) { const int t = t0 + 2 * e + cn; if (t < T) Di[(size_t) t * Mi + m] = acc[i][e]; }
    }
}
template <int MT1, int MT2, bool WH>
static void mmb_f32narrow2_launch(const float * W1, const float * W2, const _Float16 * H1, const _Float16 * H2, const float * X,
        float * D1, float * D2, const int M1, const int M2, const int K, const int T, cudaStream_t stream) {
    const dim3 grid((T + MMB_NT / 2 - 1) / (MMB_NT / 2));   // a wave per 16 columns
    mmb_f32narrow2_kernel<MT1, MT2, 2, WH><<<grid, MMB_NT, 0, stream>>>(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T);
}

// F16 copies of narrow F32 weights for the two-product mode, keyed by data pointer, made before the graph runs
// (ggml_cuda_mmb_f16w_prepare). STRIX_MMB_F16W=0: none
static std::unordered_map<const void *, _Float16 *> g_mmb_f16w;
static bool mmb_f32_two() { static const bool v = getenv("LLAMA_MMB_F32SPLIT") && atoi(getenv("LLAMA_MMB_F32SPLIT")) >= 2; return v; }
static bool mmb_f16w_on() { static const bool v = mmb_f32_two() && (!getenv("STRIX_MMB_F16W") || atoi(getenv("STRIX_MMB_F16W")) != 0); return v; }
static const _Float16 * mmb_f16w_lookup(const void * data) { auto it = g_mmb_f16w.find(data); return it == g_mmb_f16w.end() ? nullptr : it->second; }
__global__ void mmb_cvt_f32_f16(const float * __restrict__ x, _Float16 * __restrict__ y, const size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (_Float16) x[i];
}
// one or two narrow weights in two-product mode through mmb_f32narrow2_kernel: false (nothing run) when the shapes have
// no instance here or a single weight has no F16 copy
static bool mmb_f32narrow2(const float * W1, const float * W2, const int M1, const int M2, const float * X, float * D1, float * D2,
        const int K, const int T, cudaStream_t stream) {
    const _Float16 * H1 = mmb_f16w_on() ? mmb_f16w_lookup(W1) : nullptr;
    const _Float16 * H2 = W2 && mmb_f16w_on() ? mmb_f16w_lookup(W2) : nullptr;
    const int mt1 = (M1 + 15) / 16, mt2 = W2 ? (M2 + 15) / 16 : 0;
    if (!mmb_f32_two() || K % 32 != 0 || mt1 < 1 || mt1 > 4) return false;
    if (!W2) {
        if (!H1) return false;
        switch (mt1) {
            case 1:  mmb_f32narrow2_launch<1, 0, true>(W1, W1, H1, H1, X, D1, D1, M1, M1, K, T, stream); break;
            case 2:  mmb_f32narrow2_launch<2, 0, true>(W1, W1, H1, H1, X, D1, D1, M1, M1, K, T, stream); break;
            case 3:  mmb_f32narrow2_launch<3, 0, true>(W1, W1, H1, H1, X, D1, D1, M1, M1, K, T, stream); break;
            default: mmb_f32narrow2_launch<4, 0, true>(W1, W1, H1, H1, X, D1, D1, M1, M1, K, T, stream); break;
        }
        return true;
    }
    if (mt2 != mt1) return false;
    const bool wh = H1 && H2;
    switch (mt1 * 2 + (wh ? 1 : 0)) {
        case 2:  mmb_f32narrow2_launch<1, 1, false>(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        case 3:  mmb_f32narrow2_launch<1, 1, true >(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        case 4:  mmb_f32narrow2_launch<2, 2, false>(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        case 5:  mmb_f32narrow2_launch<2, 2, true >(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        case 6:  mmb_f32narrow2_launch<3, 3, false>(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        case 7:  mmb_f32narrow2_launch<3, 3, true >(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        case 8:  mmb_f32narrow2_launch<4, 4, false>(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
        default: mmb_f32narrow2_launch<4, 4, true >(W1, W2, H1, H2, X, D1, D2, M1, M2, K, T, stream); break;
    }
    return true;
}

// two tile classes: experts with >= thresh rows get BN_BIG-row tiles, the rest BN_SMALL-row tiles (fewer wasted rows on tiny experts)
__global__ void mmb_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
}

// strixllama: three tile classes - an expert of up to BN_SMALL rows gets one BN_SMALL-row tile, of up to BN_MID one
// BN_MID-row tile, a bigger one BN_BIG-row tiles. A tile reads its expert's weights once whatever its width, so an
// expert under BN_BIG rows in BN_SMALL tiles read them up to four times: at 2K tokens (median 31 rows an expert) the
// routed down projection read its weights 1.59 times an expert, here 1.15. Every row sees the same WMMA sequence in
// any class, so results do not depend on the split.
__global__ void mmb_build_desc3(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_mid,
        uint32_t * __restrict__ desc_small, const int E, const int nbig_max, const int nmid_max, const int nsmall_max,
        const int BN_BIG, const int BN_MID, const int BN_SMALL) {
    __shared__ int sb[1024], sm[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nmid_max;   i += blockDim.x) desc_mid[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    const int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const int tb = cnt > BN_MID ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int tm = cnt > BN_SMALL && cnt <= BN_MID ? 1 : 0;
    const int ts = cnt > 0 && cnt <= BN_SMALL ? 1 : 0;
    sb[e] = tb; sm[e] = tm; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vm = (e >= off) ? sm[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; sm[e] += vm; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bm = sm[e] - tm, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max) desc_big[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
    if (tm && bm < nmid_max)   desc_mid[bm]   = (uint32_t)e;
    if (ts && bs < nsmall_max) desc_small[bs] = (uint32_t)e;
}
static bool mmb_down_tiles3() { static const bool v = !getenv("STRIX_MMB_DOWN_TILES") || atoi(getenv("STRIX_MMB_DOWN_TILES")) != 2; return v; }
// strixllama: the fused gate/up keeps two steps of weights in flight (mmb_tile_gemm PF; q4_K -2%, iq3_s -1.5% at 2K
// tokens); STRIX_MMB_PF=1: one. The down kernel measured the same either way and keeps one
static bool mmb_pf2() { static const bool v = !getenv("STRIX_MMB_PF") || atoi(getenv("STRIX_MMB_PF")) == 2; return v; }

// strixllama: ld - the copy's row stride in elements when its rows are padded (mmb_pad_ld), else 0 (rows of K)
struct mmb_cache_entry { const ggml_tensor * root; const void * data; size_t n; ggml_cuda_pool_alloc<uint16_t> * buf; int64_t ld; int64_t k; };
static std::vector<mmb_cache_entry> g_mmb_cache;
static mmb_cache_entry g_mmb_slots[4] = {{nullptr,nullptr,0,nullptr,0,0},{nullptr,nullptr,0,nullptr,0,0},{nullptr,nullptr,0,nullptr,0,0},{nullptr,nullptr,0,nullptr,0,0}};

// strixllama: BF16 activation rows a multiple of 4 KB long (K 2048, 6144, 10240) are stored 64 elements longer. A dense
// GEMM's blocks walk K in step, and with such a row stride every block's loads fall on the same memory channels: the
// HC down [10240 -> 320] 0.78 -> 0.55 ms at 2K tokens, 2.99 -> 1.97 at 8K; the GDN out (K 6144) 2.04 -> 1.94 ms.
// STRIX_MMB_PAD=0 keeps them dense.
static int64_t mmb_pad_ld(const int64_t K) {
    static const bool on = !getenv("STRIX_MMB_PAD") || atoi(getenv("STRIX_MMB_PAD")) != 0;
    // a row longer than this is a shape mistake (the model's longest is 10240): refuse it rather than index with it
    GGML_ASSERT(K > 0 && K <= (1 << 20));
    return on && (K * 2) % 4096 == 0 ? K + 64 : 0;
}
// row by row into rows of ld elements
__global__ void mmb_cvt_f32_bf16_ld(const float * __restrict__ x, uint16_t * __restrict__ y, const int64_t K, const int64_t ld) {
    const int64_t row = blockIdx.y;
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 > K) return;
    const float4 a = *(const float4 *)(x + row * K + i), b = *(const float4 *)(x + row * K + i + 4);
    uint4 o; o.x = mmb_pack2(a.x, a.y); o.y = mmb_pack2(a.z, a.w); o.z = mmb_pack2(b.x, b.y); o.w = mmb_pack2(b.z, b.w);
    *(uint4 *)(y + row * ld + i) = o;
}
static std::unordered_set<const ggml_tensor *> g_mmb_bf16_only;

static size_t mmb_cache_max() { static const int v = getenv("LLAMA_MMB_CACHE") ? atoi(getenv("LLAMA_MMB_CACHE")) : 4; return (size_t) v; }
static const ggml_tensor * mmb_root(const ggml_tensor * t) { return t->view_src ? t->view_src : t; }
static uint16_t * mmb_cache_insert(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const size_t n, const size_t alloc_n, const int64_t ld, const int64_t k) {
    if (g_mmb_cache.size() >= mmb_cache_max()) { delete g_mmb_cache.front().buf; g_mmb_cache.erase(g_mmb_cache.begin()); }
    auto * buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), alloc_n);
    g_mmb_cache.push_back({mmb_root(t), t->data, n, buf, ld, k});
    return buf->get();
}
// K and ld: a caller that passes ld can read padded rows (its row stride comes back in *ld, 0 = K); one that does not
// only ever gets dense rows
static const uint16_t * mmb_bf16_activation(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, cudaStream_t stream,
        const int64_t K = 0, int64_t * ld = nullptr) {
    const ggml_tensor * root = mmb_root(src1);
    // a padded copy only for a caller that reads rows of the length it was padded for
    auto usable = [&](const mmb_cache_entry & e) { return !e.ld || (ld && e.k == K); };
    for (auto & e : g_mmb_slots) if (e.buf && e.root == root && e.data == src1->data && e.n == n && usable(e)) { if (ld) *ld = e.ld; return e.buf->get(); }
    for (auto & e : g_mmb_cache) if (e.root == root && e.data == src1->data && e.n == n && usable(e)) { if (ld) *ld = e.ld; return e.buf->get(); }
    const int64_t rows = K > 0 ? (int64_t) (n / (size_t) K) : 0;
    const int64_t pad = ld && K > 0 && n % (size_t) K == 0 && rows <= 65535 && ggml_is_contiguous(src1) ? mmb_pad_ld(K) : 0;
    if (ld) *ld = pad;
    if (pad) {
        uint16_t * buf = mmb_cache_insert(ctx, src1, n, (size_t) rows * pad, pad, K);
        mmb_cvt_f32_bf16_ld<<<dim3((unsigned) ((K / 8 + 255) / 256), (unsigned) rows), 256, 0, stream>>>((const float *) src1->data, buf, K, pad);
        return buf;
    }
    uint16_t * buf = mmb_cache_insert(ctx, src1, n, n, 0, 0);
    { static const int lg = getenv("LLAMA_MMB_CVT_LOG") ? atoi(getenv("LLAMA_MMB_CVT_LOG")) : 0; static unsigned cnt = 0;
      if (lg && cnt++ < 200) fprintf(stderr, "MMB_CVT %s op=%s ne=[%lld,%lld,%lld,%lld] view_src=%s n=%zu\n", src1->name, ggml_op_name(src1->op), (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3], src1->view_src ? src1->view_src->name : "-", n); }
    mmb_cvt_f32_bf16<<<(unsigned)((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, buf, n);
    return buf;
}

// Shadow BF16 copies of IQ4_NL dense weights: dequantised once (same LUT*scale -> BF16 RNE as mmb_dq_row36, so the
// WMMA inputs are bitwise identical) so the dense GEMM runs the dequant-free WTYPE=2 path.
__global__ void mmb_dq_q6k_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p  = W + b * 210;
    const uint8_t * ql = p, * qh = p + 128;
    const int8_t  * sc = (const int8_t *) (p + 192);
    const float d = mmb_h2f(*(const uint16_t *) (p + 208));
    uint16_t * o = out + b * 256;
    for (int n = 0; n < 2; ++n) {
        const uint8_t * QL = ql + 64 * n; const uint8_t * QH = qh + 32 * n; const int8_t * S = sc + 8 * n; uint16_t * Y = o + 128 * n;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((QL[l +  0] & 0xF) | (((QH[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((QL[l +  0] >>  4) | (((QH[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((QL[l + 32] >>  4) | (((QH[l] >> 6) & 3) << 4)) - 32;
            Y[l +  0] = mmb_f2bf(d * S[is + 0] * q1);
            Y[l + 32] = mmb_f2bf(d * S[is + 2] * q2);
            Y[l + 64] = mmb_f2bf(d * S[is + 4] * q3);
            Y[l + 96] = mmb_f2bf(d * S[is + 6] * q4);
        }
    }
}

__global__ void mmb_dq_iq4nl_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p = W + b * 18;
    const float d = mmb_h2f(*(const uint16_t *) p);
    const uint32_t * q = (const uint32_t *) (p + 2);
    uint32_t * o = (uint32_t *) (out + b * 32);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = q[w];
        const float l0 = d * mmb_kv_iq4nl[(v      ) & 0xF], h0 = d * mmb_kv_iq4nl[(v >>  4) & 0xF];
        const float l1 = d * mmb_kv_iq4nl[(v >>  8) & 0xF], h1 = d * mmb_kv_iq4nl[(v >> 12) & 0xF];
        const float l2 = d * mmb_kv_iq4nl[(v >> 16) & 0xF], h2 = d * mmb_kv_iq4nl[(v >> 20) & 0xF];
        const float l3 = d * mmb_kv_iq4nl[(v >> 24) & 0xF], h3 = d * mmb_kv_iq4nl[(v >> 28) & 0xF];
        o[2*w] = mmb_pack2(l0, l1); o[2*w + 1] = mmb_pack2(l2, l3); o[8 + 2*w] = mmb_pack2(h0, h1); o[8 + 2*w + 1] = mmb_pack2(h2, h3);
    }
}
static std::unordered_map<const void *, uint16_t *> g_mmb_shadow;
static std::map<std::pair<const void *, const void *>, uint16_t *> g_mmb_shadow_pair;   // concat(w0, w1) along rows -> BF16 copy
static size_t g_mmb_shadow_bytes = 0;
int    mmb_shadow_mode(){ static const int v = getenv("LLAMA_MMB_SHADOW") ? atoi(getenv("LLAMA_MMB_SHADOW")) : 0; return v; }
bool   mmb_shadow()    { return mmb_shadow_mode() != 0; }
bool   mmb_shadow_q6k(){ return mmb_shadow_mode() >= 1; }
size_t mmb_shadow_cap(){ static const long v = getenv("LLAMA_MMB_SHADOW_MB") ? atol(getenv("LLAMA_MMB_SHADOW_MB")) : 6144; return (size_t) v << 20; }
static bool mmb_is_resident_q6k(const ggml_tensor * w) { return w && w->type == GGML_TYPE_Q6_K && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && w->ne[0] % 256 == 0 && w->ne[1] <= 32768; }
static bool mmb_is_resident_iq4(const ggml_tensor * w) { return w && w->type == GGML_TYPE_IQ4_NL && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w); }
static bool mmb_is_row_concat(const ggml_tensor * w) {
    return w && w->op == GGML_OP_CONCAT && w->type == GGML_TYPE_IQ4_NL && ggml_get_op_params_i32(w, 0) == 1 && mmb_is_resident_iq4(w->src[0]) && mmb_is_resident_iq4(w->src[1]) &&
           w->src[0]->ne[0] == w->src[1]->ne[0] && w->ne[0] == w->src[0]->ne[0] && w->ne[1] == w->src[0]->ne[1] + w->src[1]->ne[1];
}
static const uint16_t * mmb_shadow_lookup(const ggml_tensor * w) {
    if (w->op == GGML_OP_CONCAT) { auto it = g_mmb_shadow_pair.find({w->src[0]->data, w->src[1]->data}); return it == g_mmb_shadow_pair.end() ? nullptr : it->second; }
    auto it = g_mmb_shadow.find(w->data); return it == g_mmb_shadow.end() ? nullptr : it->second;
}

bool mmb_enabled() { static const int v = getenv("LLAMA_MMB") ? atoi(getenv("LLAMA_MMB")) : 0; return v != 0; }
int  mmb_min_t()   { static const int v = getenv("LLAMA_MMB_MIN_T") ? atoi(getenv("LLAMA_MMB_MIN_T")) : 512; return v; }
int  mmb_f32split_mode(){ static const int v = getenv("LLAMA_MMB_F32SPLIT") ? atoi(getenv("LLAMA_MMB_F32SPLIT")) : 0; return v; }
bool mmb_f32split() { static const int v = getenv("LLAMA_MMB_F32SPLIT") ? atoi(getenv("LLAMA_MMB_F32SPLIT")) : 0; return v != 0; }
bool mmb_bf16w()    { static const int v = getenv("LLAMA_MMB_BF16W") ? atoi(getenv("LLAMA_MMB_BF16W")) : 0; return v != 0; }
bool mmb_hc16()    { static const int v = getenv("LLAMA_MMB_HC16") ? atoi(getenv("LLAMA_MMB_HC16")) : 0; return v != 0; }
int  mmb_tall_mode(){ static const int v = getenv("LLAMA_MMB_TALL") ? atoi(getenv("LLAMA_MMB_TALL")) : 0; return v; }
bool mmb_tall()    { return mmb_tall_mode() != 0; }
bool mmb_gatemix_flag() { static const int v = getenv("LLAMA_HC_GATEMIX") ? atoi(getenv("LLAMA_HC_GATEMIX")) : 0; return v != 0; }
bool mmb_down16_flag() { static const int v = getenv("LLAMA_MMB_DOWN16") ? atoi(getenv("LLAMA_MMB_DOWN16")) : 0; return v != 0; }
bool mmb_glu()     { static const int v = getenv("LLAMA_MMB_GLU") ? atoi(getenv("LLAMA_MMB_GLU")) : 0; return v != 0; }

} // namespace

const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t, int64_t * ld) {
    const ggml_tensor * root = mmb_root(t);
    for (auto & e : g_mmb_slots) if (e.buf && e.root == root && e.data == t->data && (ld || !e.ld)) { if (ld) *ld = e.ld; return e.buf->get(); }
    for (auto & e : g_mmb_cache) if (e.root == root && e.data == t->data && (ld || !e.ld)) { if (ld) *ld = e.ld; return e.buf->get(); }
    return nullptr;
}
static size_t g_mmb_slot_cap[4] = {0, 0, 0};
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n, int64_t ld, size_t alloc_n) {
    mmb_cache_entry & e = g_mmb_slots[slot];
    const size_t need = alloc_n > n ? alloc_n : n;
    if (e.buf && g_mmb_slot_cap[slot] < need) { delete e.buf; e.buf = nullptr; }
    if (!e.buf) { e.buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), need); g_mmb_slot_cap[slot] = need; }
    e.root = mmb_root(t); e.data = t->data; e.n = n; e.ld = ld; e.k = ld && alloc_n ? (int64_t) (n / (alloc_n / ld)) : 0;
    return e.buf->get();
}
void ggml_cuda_mmb_marks_clear() { g_mmb_bf16_only.clear(); }
size_t ggml_cuda_mmb_marks_count() { return g_mmb_bf16_only.size(); }
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t) { g_mmb_bf16_only.insert(t); }
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t) { return g_mmb_bf16_only.count(t) > 0; }
void ggml_cuda_mmb_begin_graph() { for (auto & e : g_mmb_cache) delete e.buf; g_mmb_cache.clear(); for (auto & e : g_mmb_slots) { e.root = nullptr; e.data = nullptr; e.n = 0; e.ld = 0; e.k = 0; } }
void ggml_cuda_mmb_release_all() {
    ggml_cuda_mmb_begin_graph();
    for (int i = 0; i < 4; ++i) { if (g_mmb_slots[i].buf) delete g_mmb_slots[i].buf; g_mmb_slots[i].buf = nullptr; g_mmb_slot_cap[i] = 0; }
    // the shadow weights are raw cudaMalloc, keyed by data pointer and held for the life of the
    // process. A model has finitely many weights so this never mattered, but a long-lived process
    // that sees many distinct tensors (test-backend-ops) keeps every one of them.
    for (auto & e : g_mmb_shadow)      { if (e.second) cudaFree(e.second); }
    for (auto & e : g_mmb_shadow_pair) { if (e.second) cudaFree(e.second); }
    g_mmb_shadow.clear();
    g_mmb_shadow_pair.clear();
    g_mmb_shadow_bytes = 0;
    for (auto & e : g_mmb_f16w) { if (e.second) cudaFree(e.second); }
    g_mmb_f16w.clear();
}

void ggml_cuda_mmb_f16w_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!mmb_f16w_on() || !w || w->type != GGML_TYPE_F32 || w->op != GGML_OP_NONE || !w->data || !w->buffer || !ggml_is_contiguous(w) ||
            w->ne[1] > 64 || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] % 32 != 0 || g_mmb_f16w.count(w->data) > 0) {
        return;
    }
    const size_t n = (size_t) w->ne[0] * w->ne[1];
    _Float16 * buf = nullptr;
    if (cudaMalloc((void **) &buf, n * sizeof(_Float16)) != cudaSuccess) { fprintf(stderr, "MMB_F16W alloc failed (%zu bytes)\n", n * 2); return; }
    mmb_cvt_f32_f16<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) w->data, buf, n);
    CUDA_CHECK(cudaGetLastError());
    g_mmb_f16w[w->data] = buf;
    static size_t total = 0; static unsigned hits = 0; total += n * 2;
    if (hits++ < 2) fprintf(stderr, "MMB_F16W %s [%lld x %lld] -> F16 (%.1f MB so far)\n", w->name, (long long) w->ne[0], (long long) w->ne[1], total / 1048576.0);
}

bool ggml_cuda_mmb_f32_narrow_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * w1, const ggml_tensor * w2, const ggml_tensor * x,
        ggml_tensor * d1, ggml_tensor * d2) {
    static const bool on = !getenv("STRIX_MMB_NARROW_PAIR") || atoi(getenv("STRIX_MMB_NARROW_PAIR")) != 0;
    if (!on || w1->ne[0] != w2->ne[0]) return false;
    const int K = (int) w1->ne[0], T = (int) x->ne[1];
    if (!mmb_f32narrow2((const float *) w1->data, (const float *) w2->data, (int) w1->ne[1], (int) w2->ne[1], (const float *) x->data,
            (float *) d1->data, (float *) d2->data, K, T, ctx.stream())) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_NARROW_PAIR %s + %s: K=%d T=%d\n", w1->name, w2->name, K, T);
    return true;
}
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n) {
    if (!mmb_enabled() || ggml_nrows(t) < mmb_min_t()) return nullptr;
    return ggml_cuda_mmb_slot_reserve(ctx, 0, t, n);
}
uint16_t * ggml_cuda_mmb_cache_reserve_ld(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int64_t rows, int64_t K, int64_t * ld) {
    *ld = 0;
    if (!mmb_enabled() || ggml_nrows(t) < mmb_min_t()) return nullptr;
    GGML_ASSERT(rows > 0 && K > 0 && (size_t) (rows * K) == (size_t) ggml_nelements(t));
    *ld = mmb_pad_ld(K);
    return ggml_cuda_mmb_slot_reserve(ctx, 0, t, (size_t) (rows * K), *ld, *ld ? (size_t) (rows * *ld) : 0);
}

bool ggml_cuda_mmb_supported_mm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!mmb_enabled()) return false;
    const bool quant = src0->type == GGML_TYPE_IQ4_NL || src0->type == GGML_TYPE_Q8_0 ||
                       (src0->type == GGML_TYPE_Q6_K && mmb_shadow_q6k() && mmb_is_resident_q6k(src0) &&
                        mmb_shadow_lookup(src0) != nullptr);
    const bool bf16w = src0->type == GGML_TYPE_BF16 && mmb_bf16w();
    const bool f32w  = src0->type == GGML_TYPE_F32 && mmb_f32split();
    if ((!quant && !bf16w && !f32w) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if ((f32w ? K % 32 : K % 64) != 0 || src1->ne[0] != K || dst->ne[0] != M) return false;
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    // strixllama: F32 weights (the MoE router, the indexer's decode-time scores) from 9 columns: below 512 they went
    // to hipBLAS, which loads each GEMM kernel from disk on its first use - 20 to 400 ms stalls in the first requests,
    // and again whenever a context length or batch size reaches a new kernel. Same numerics as the long batches get.
    static const int64_t f32_min_t = getenv("STRIX_MMB_F32_MIN_T") ? atoll(getenv("STRIX_MMB_F32_MIN_T")) : 9;
    // strixllama: BF16 weights (the sparse-attention indexer's projections) likewise: mul_mat_f takes up to 16 columns
    // and this kernel took 512 on, so a 17-511 token batch - most chat turns - loaded hipBLAS's BF16 kernels, ~0.4 s
    static const int64_t bf16_min_t = getenv("STRIX_MMB_BF16_MIN_T") ? atoll(getenv("STRIX_MMB_BF16_MIN_T")) : 9;
    if (T < (f32w ? f32_min_t : bf16w ? bf16_min_t : (int64_t) mmb_min_t()) || T > INT32_MAX / 4) return false;
    return ggml_nrows(dst) == T;
}

// allow_iq3s is set only by the fused-GLU caller: the routed GLU kernel can dequantize IQ3_S but
// the plain mmid kernel still assumes the 36-byte IQ4_NL step, so the default must stay strict.
// strixllama: STRIX_MMB_IQ3S=0 sends IQ3_S weights back to the stock MMQ path, so the kernel can be
// A/B'd on one binary. It carries about half the model body here, so this is the switch to reach for
// when a numerics or long-generation regression has to be attributed.
static bool mmb_iq3s_enabled() {
    static const bool e = [] {
        const char * v = getenv("STRIX_MMB_IQ3S");
        return !v || atoi(v) != 0;
    }();
    return e;
}

// strixllama: UD-Q4_K_XL's experts - gate/up Q4_K (through the fused GLU only) and down Q5_1. STRIX_MMB_KQ=0 sends
// them back to the stock MMQ path
static bool mmb_kq_enabled() {
    static const bool e = !getenv("STRIX_MMB_KQ") || atoi(getenv("STRIX_MMB_KQ")) != 0;
    return e;
}
// strixllama: the experts UD-IQ4_XS left on MMQ - the Q8_0 down projection of five layers (either file, and the MTP
// head's) through the routed kernel, dequantized to bf16 as the dense Q8_0 GEMMs are (WTYPE 1), and the IQ4_XS
// gate/up of layer 2 through glu3: at 2K tokens 12.1 -> 5.7 ms and 11.3 -> 7.1 ms. STRIX_MMB_Q8=0 sends both back to MMQ
static bool mmb_q8_enabled() {
    static const bool e = !getenv("STRIX_MMB_Q8") || atoi(getenv("STRIX_MMB_Q8")) != 0;
    return e;
}

bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, bool allow_iq3s) {
    if (!mmb_enabled()) return false;
    const bool iq3s = allow_iq3s && mmb_iq3s_enabled() && src0->type == GGML_TYPE_IQ3_S;
    const bool q4k  = allow_iq3s && mmb_kq_enabled() && src0->type == GGML_TYPE_Q4_K;
    const bool q51  = mmb_kq_enabled() && src0->type == GGML_TYPE_Q5_1;
    const bool q80  = mmb_q8_enabled() && src0->type == GGML_TYPE_Q8_0;
    const bool iq4x = allow_iq3s && mmb_q8_enabled() && src0->type == GGML_TYPE_IQ4_XS;   // through the fused GLU only
    if ((src0->type != GGML_TYPE_IQ4_NL && !iq3s && !q4k && !q51 && !q80 && !iq4x) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1], E = src0->ne[2];
    if (src0->ne[3] != 1 || K % 64 != 0 || E < 1 || E > 1024) return false;
    // a 64-weight step must not straddle two 256-weight superblocks
    if ((iq3s || q4k || iq4x) && K % 256 != 0) return false;
    const int64_t n_used = ids->ne[0], T = ids->ne[1];
    if (src1->ne[0] != K || src1->ne[3] != 1 || src1->ne[2] != T) return false;
    if (src1->ne[1] != 1 && src1->ne[1] != n_used) return false;
    if (dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != T || dst->ne[3] != 1) return false;
    if (ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) return false;
    if (T < mmb_min_t() || n_used > 64 || (T * n_used) >> 16 >= 1024) return false;   // tile index must fit in 16 bits per expert
    return true;
}

bool ggml_cuda_mmb_f32_narrow(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0], M = src0->ne[1], T = src1->ne[1];
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || M > 64 || K % 32 != 0 ||
            T < 1 || T > INT32_MAX / 4 || src1->ne[0] != K || dst->ne[0] != M || dst->ne[1] != T || ggml_nrows(src0) != M ||
            ggml_nrows(src1) != T || ggml_nrows(dst) != T || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
            !ggml_is_contiguous(dst)) {
        return false;
    }
    // the product count of the tile GEMM (LLAMA_MMB_F32SPLIT: 2 = two, else three)
    static const bool two = getenv("LLAMA_MMB_F32SPLIT") && atoi(getenv("LLAMA_MMB_F32SPLIT")) >= 2;
    if (two && mmb_f32narrow2((const float *) src0->data, nullptr, (int) M, 0, (const float *) src1->data, (float *) dst->data, nullptr, (int) K, (int) T, ctx.stream())) {
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (two) mmb_f32narrow<true, 2>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, (int) T, ctx.stream());
    else     mmb_f32narrow<false, 1>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, (int) T, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int T = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    if (src0->type == GGML_TYPE_F32) {
        dim3 grid((M + 127) / 128, (T + 127) / 128);
        static const bool two = getenv("LLAMA_MMB_F32SPLIT") && atoi(getenv("LLAMA_MMB_F32SPLIT")) >= 2;
        static const bool narrow = !getenv("STRIX_MMB_F32_NARROW") || atoi(getenv("STRIX_MMB_F32_NARROW")) != 0;
        if (narrow && M <= 64) {
            if (two && mmb_f32narrow2((const float *) src0->data, nullptr, M, 0, (const float *) src1->data, (float *) dst->data, nullptr, K, T, stream)) {
                CUDA_CHECK(cudaGetLastError()); return;
            }
            // three products hold one more weight fragment a step: one step in flight (registers)
            if (two) mmb_f32narrow<true, 2>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T, stream);
            else     mmb_f32narrow<false, 1>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T, stream);
            CUDA_CHECK(cudaGetLastError()); return;
        }
        static const bool tile2 = !getenv("STRIX_MMB_F32_TILE2") || atoi(getenv("STRIX_MMB_F32_TILE2")) != 0;
        if (two && tile2 && K % 64 == 0 && T >= mmb_min_t()) {   // prompt batches: decode-sized ones keep the tile kernel
            mmb_f32split2_kernel<64, 128, 16, 64, 64><<<dim3((M + 63) / 64, (T + 127) / 128), MMB_NT, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T);
            CUDA_CHECK(cudaGetLastError()); return;
        }
        if (two) mmb_f32split_kernel<128, 128, 32, 64, true ><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T);
        else     mmb_f32split_kernel<128, 128, 32, 64, false><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, K, T);
        CUDA_CHECK(cudaGetLastError()); return;
    }
    int64_t ldx = 0;   // the activations' row stride when padded (mmb_pad_ld)
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) T * K, stream, K, &ldx);
    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data;
    // tall-M tile: HC down|inject [10240 -> 324], activations read once. strixllama: Q8_0 as well as IQ4_NL - Unsloth's
    // HC weights are Q8_0. Every output element sees the same WMMA sequence as in the 128-row tile, so the result is
    // bitwise the one the separate down and inject GEMMs give.
    if (mmb_tall() && (src0->type == GGML_TYPE_IQ4_NL || src0->type == GGML_TYPE_Q8_0) && M <= 384 && K >= 4096 && T >= 2048) {
        static const int wide = mmb_tall_mode() >= 2;
        const bool q8 = src0->type == GGML_TYPE_Q8_0;
        if (wide) { dim3 grid(1, (T + 63) / 64);
            if (q8) mmb_dense_kernel<384, 64, 96, 32, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T, (int) ldx);
            else    mmb_dense_kernel<384, 64, 96, 32, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T, (int) ldx); }
        else      { dim3 grid(1, (T + 31) / 32);
            if (q8) mmb_dense_kernel<384, 32, 96, 16, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T, (int) ldx);
            else    mmb_dense_kernel<384, 32, 96, 16, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T, (int) ldx); }
        CUDA_CHECK(cudaGetLastError());
        static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_TALL%s dense M=%d K=%d T=%d type=%s\n", wide ? "(wide 384x64)" : "(384x32)", M, K, T, ggml_type_name(src0->type));
        return;
    }
    const uint16_t * shadow_pre = ((src0->type == GGML_TYPE_IQ4_NL && mmb_shadow()) || src0->type == GGML_TYPE_Q6_K) ? mmb_shadow_lookup(src0) : nullptr;
    const bool big = (M >= 6144 && K >= 2560) || (shadow_pre && K >= 2560 && T >= 4096);
    uint16_t * Dh = (mmb_hc16() && K == 320 && M == 10240) ? ggml_cuda_mmb_slot_reserve(ctx, 1, dst, (size_t) T * M) : nullptr;
    bool store_f32 = !(Dh && ggml_cuda_mmb_is_bf16_only(dst));
    if (ggml_cuda_mmb_blk16() && !Dh && ggml_cuda_mmb_is_bf16_only(dst) && (M & 7) == 0) {
        Dh = (uint16_t *) dst->data; store_f32 = false;
        static unsigned h = 0; if (h++ < 2) fprintf(stderr, "MMB_BLK16 dense BF16 in place: M=%d K=%d T=%d\n", M, K, T);
    }
    dim3 grid((M + 127) / 128, big ? (T + 255) / 256 : (T + 127) / 128);
    const uint16_t * shadow = shadow_pre;
    if (src0->type == GGML_TYPE_Q6_K && !shadow) { GGML_ABORT("MMB: Q6_K weight %s has no BF16 shadow", src0->name); }
    if (shadow) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 2><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) shadow, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
        else     mmb_dense_kernel<128, 128, 32, 64, 2><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) shadow, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
    } else if (src0->type == GGML_TYPE_IQ4_NL) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
        else     mmb_dense_kernel<128, 128, 32, 64, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
    } else if (src0->type == GGML_TYPE_Q8_0) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
        else     mmb_dense_kernel<128, 128, 32, 64, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
    } else {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 2><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
        else     mmb_dense_kernel<128, 128, 32, 64, 2><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T, (int) ldx);
    }
    CUDA_CHECK(cudaGetLastError());
}

// strixllama: z = w x as the dense MMB GEMM would compute it, with the gated RMS norm out = (scale * o * g) * sigmoid(z)
// in its epilogue (mmb_dense_gnorm_kernel). False (nothing run) unless the GEMM would have been that plain dense kernel
// and the norm rows are one head of 128 per 128 rows of w. STRIX_GNORM_FUSE=0: off
bool ggml_cuda_mmb_gemm_gnorm(ggml_backend_cuda_context & ctx, const ggml_tensor * gemm, const ggml_tensor * o, const ggml_tensor * g,
        const float eps, ggml_tensor * out) {
    static const bool on = !getenv("STRIX_GNORM_FUSE") || atoi(getenv("STRIX_GNORM_FUSE")) != 0;
    const ggml_tensor * w = gemm->src[0], * src1 = gemm->src[1];
    if (!on || (w->type != GGML_TYPE_Q8_0 && w->type != GGML_TYPE_IQ4_NL) || !ggml_cuda_mmb_supported_mm(w, src1, gemm)) return false;
    if (w->type == GGML_TYPE_IQ4_NL && mmb_shadow()) return false;   // that GEMM reads a BF16 shadow
    const int K = (int) w->ne[0], M = (int) w->ne[1];
    const int T = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    if (M % 128 != 0 || o->ne[0] != 128 || o->type != GGML_TYPE_F32 || o->nb[0] != sizeof(float) || g->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(g) || g->ne[0] != 128 || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
            ggml_nelements(out) != (int64_t) M * T || ggml_nelements(o) != (int64_t) M * T || ggml_nrows(gemm) != T) {
        return false;
    }
    if (mmb_tall() && M <= 384) return false;
    if (mmb_hc16() && K == 320 && M == 10240) return false;
    if (ggml_cuda_mmb_blk16() && ggml_cuda_mmb_is_bf16_only(gemm)) return false;
    cudaStream_t stream = ctx.stream();
    int64_t ldx = 0;
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) T * K, stream, K, &ldx);
    const dim3 grid(M / 128, (T + 255) / 256);
    const int64_t nrows = o->ne[1], nch = o->ne[2];
    if (w->type == GGML_TYPE_Q8_0) {
        mmb_dense_gnorm_kernel<1><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, xhp, M, K, T, (int) ldx, (const float *) o->data,
            (const float *) g->data, (float *) out->data, nrows, nch, (int64_t) (o->nb[1] / 4), (int64_t) (o->nb[2] / 4), (int64_t) (o->nb[3] / 4), eps, (int) o->ne[0]);
    } else {
        mmb_dense_gnorm_kernel<0><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, xhp, M, K, T, (int) ldx, (const float *) o->data,
            (const float *) g->data, (float *) out->data, nrows, nch, (int64_t) (o->nb[1] / 4), (int64_t) (o->nb[2] / 4), (int64_t) (o->nb[3] / 4), eps, (int) o->ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_GNORM gate GEMM + gated norm: %s M=%d K=%d T=%d\n", w->name, M, K, T);
    return true;
}

bool ggml_cuda_mmb_gatemix() { return mmb_gatemix_flag(); }
bool ggml_cuda_mmb_down16() { return mmb_down16_flag(); }
bool ggml_cuda_mmb_blk16() { static const int v = getenv("LLAMA_HC_BLK16") ? atoi(getenv("LLAMA_HC_BLK16")) : 0; return v != 0; }
bool ggml_cuda_mmb_res16()  { static const int v = getenv("LLAMA_HC_RES16") ? atoi(getenv("LLAMA_HC_RES16")) : 0; return v != 0; }
struct mmb_xres_entry { const void * xn_data; const float * res; const float * rs; const float * gamma; };
static mmb_xres_entry g_mmb_xres = { nullptr, nullptr, nullptr, nullptr };
void ggml_cuda_hc_xres_set(const void * xn_data, const float * res, const float * row_scale, const float * gamma) {
    g_mmb_xres = { xn_data, res, row_scale, gamma };
}

bool ggml_cuda_hc_gate_mix_ok(const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, const ggml_tensor * dst, const int hc, bool * f32) {
    // strixllama: Q8_0 too - Unsloth's file keeps the HC weights at Q8_0, so with IQ4_NL only this never fired there
    const bool q8 = w->type == GGML_TYPE_Q8_0;
    if (!mmb_gatemix_flag() || hc != 4 || (w->type != GGML_TYPE_IQ4_NL && !q8) || lo->type != GGML_TYPE_F32 || !ggml_is_contiguous(lo) || !ggml_is_contiguous(dst)) return false;
    const int K = (int) w->ne[0], M = (int) w->ne[1], E = (int) dst->ne[0]; const int T = (int) ggml_nrows(dst);
    if (K % MMB_BK != 0 || M != hc * E || E % 32 != 0 || lo->ne[0] != K || ggml_nrows(lo) != T || xn->ne[0] != M || ggml_nrows(xn) != T || T < mmb_min_t()) return false;
    if (f32) { *f32 = q8 && xn->type == GGML_TYPE_F32 && ggml_is_contiguous(xn) && !ggml_cuda_mmb_is_bf16_only(xn); }
    return true;
}

bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst,
        const int hc, const float scale, const float bias) {
    const bool q8 = w->type == GGML_TYPE_Q8_0;
    if (!ggml_cuda_hc_gate_mix_ok(w, lo, xn, dst, hc, nullptr)) return false;
    const int K = (int) w->ne[0], M = (int) w->ne[1], E = (int) dst->ne[0]; const int T = (int) ggml_nrows(dst);
    // strixllama: the combine that wrote xn left only its BF16 copy, the F32 residual and the row scales (STRIX_HC_XRES)
    const mmb_xres_entry xres = g_mmb_xres;
    g_mmb_xres = { nullptr, nullptr, nullptr, nullptr };
    if (xres.xn_data && xres.xn_data == xn->data) {
        GGML_ASSERT(q8 && xn->type == GGML_TYPE_F32 && "formed streams stand in for the F32 ones only");
        cudaStream_t stream = ctx.stream();
        const uint16_t * lo16 = mmb_bf16_activation(ctx, lo, (size_t) T * K, stream);
        uint16_t * outh = ggml_cuda_mmb_slot_reserve(ctx, 3, dst, (size_t) T * E);
        const bool store_f32 = !(outh && ggml_cuda_mmb_is_bf16_only(dst));
        dim3 grid(E / 32, (T + 127) / 128);
        hc_gate_mix_kernel<4, 1, true, true><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, nullptr, nullptr, (float *) dst->data,
            outh, store_f32, E, K, T, scale, bias, 0, xres.res, xres.rs, xres.gamma);
        CUDA_CHECK(cudaGetLastError());
        static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "HC_GATEMIX streams formed from the residual (XRES): E=%d K=%d T=%d\n", E, K, T);
        return true;
    }
    // the Q8_0 kernel reads the streams in F32 and keeps the gate in F32: the unfused path's own numerics, so it can
    // only be taken when the F32 streams are there - i.e. when xn is not a BF16-only tensor (LLAMA_MMB_HC16)
    const bool xf32 = q8 && xn->type == GGML_TYPE_F32 && ggml_is_contiguous(xn) && !ggml_cuda_mmb_is_bf16_only(xn);
    int64_t xn_ld = 0;
    const uint16_t * xn16 = xf32 ? nullptr : ggml_cuda_mmb_cache_lookup(xn, &xn_ld);
    GGML_ASSERT(xn_ld == 0 || xn_ld == (int64_t) M + 64);   // padded rows of the streams' length only
    if (!xf32 && !xn16) return false;
    cudaStream_t stream = ctx.stream();
    const uint16_t * lo16 = mmb_bf16_activation(ctx, lo, (size_t) T * K, stream);
    uint16_t * outh = ggml_cuda_mmb_slot_reserve(ctx, 3, dst, (size_t) T * E);
    const bool store_f32 = !(outh && ggml_cuda_mmb_is_bf16_only(dst));
    dim3 grid(E / 32, (T + 127) / 128);
    if (xf32) {
        hc_gate_mix_kernel<4, 1, true><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, nullptr, (const float *) xn->data, (float *) dst->data, outh, store_f32, E, K, T, scale, bias);
    } else if (q8) {
        hc_gate_mix_kernel<4, 1, false><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, xn16, nullptr, (float *) dst->data, outh, store_f32, E, K, T, scale, bias, xn_ld);
    } else {
        hc_gate_mix_kernel<4, 0, false><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, xn16, nullptr, (float *) dst->data, outh, store_f32, E, K, T, scale, bias, xn_ld);
    }
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "HC_GATEMIX fused gate GEMM + sigmoid + mix: E=%d K=%d T=%d type=%s%s\n", E, K, T, ggml_type_name(w->type), xf32 ? " f32-streams" : "");
    return true;
}

void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], E = (int) src0->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;

    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    constexpr int BN_SMALL = 32, THRESH = 128;
    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data; const size_t eb = (size_t) src0->nb[2];
    uint16_t * Dh = (mmb_down16_flag() && ggml_cuda_mmb_is_bf16_only(dst)) ? (uint16_t *) dst->data : nullptr;
    const bool store_f32 = Dh == nullptr;
    if (Dh) { static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_DOWN16 routed down BF16 in place: M=%d K=%d rows=%d\n", M, K, n_rows); }
    if (mmb_down_tiles3() && E <= 1024) {
        constexpr int BN_MID = 64;
        const int nbig_max = n_rows / BN + E + 1, nmid_max = E + 1, nsmall_max = E + 1;
        ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
        ggml_cuda_pool_alloc<uint32_t> desc_mid(ctx.pool(), nmid_max);
        ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
        mmb_build_desc3<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_mid.get(), desc_small.get(), E, nbig_max, nmid_max, nsmall_max, BN, BN_MID, BN_SMALL);
        dim3 gbig((M + 127) / 128, nbig_max), gmid((M + 127) / 128, nmid_max), gsmall((M + 127) / 128, nsmall_max);
        auto launch3 = [&](auto wt, auto pf) {
            constexpr int WT = decltype(wt)::value, PFV = decltype(pf)::value;
            mmb_routed_kernel<128, BN, 32, 64, WT, PFV><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
            mmb_routed_kernel<128, BN_MID, 32, 32, WT, PFV><<<gmid, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_mid.get(), M, K);
            mmb_routed_kernel<128, BN_SMALL, 32, 16, WT, PFV><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
        };
        using I0 = std::integral_constant<int, 0>; using I1 = std::integral_constant<int, 1>; using I2 = std::integral_constant<int, 2>; using I4 = std::integral_constant<int, 4>;
        if (src0->type == GGML_TYPE_Q5_1)      launch3(I4{}, I1{});
        else if (src0->type == GGML_TYPE_Q8_0) launch3(I1{}, I1{});
        else                                   launch3(I0{}, I1{});
        GGML_UNUSED(I2{});
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH);
    dim3 gbig((M + 127) / 128, nbig_max), gsmall((M + 127) / 128, nsmall_max);
    if (src0->type == GGML_TYPE_Q8_0) {
        mmb_routed_kernel<128, BN, 32, 64, 1><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_kernel<128, BN_SMALL, 32, 16, 1><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    } else if (src0->type == GGML_TYPE_Q5_1) {
        mmb_routed_kernel<128, BN, 32, 64, 4><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_kernel<128, BN_SMALL, 32, 16, 4><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    } else {
        mmb_routed_kernel<128, BN, 32, 64><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_kernel<128, BN_SMALL, 32, 16><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu) {
    if (!mmb_enabled() || !mmb_glu() || !gw || !uw || !src1 || !ids || !glu) return false;
    // gate and up must share one encoding: the fused kernel dequantizes both with the same step
    if (gw->type != uw->type) return false;
    if (gw->type != GGML_TYPE_IQ4_NL && gw->type != GGML_TYPE_IQ3_S && gw->type != GGML_TYPE_Q4_K && gw->type != GGML_TYPE_IQ4_XS) return false;
    if (!ggml_are_same_shape(gw, uw) || gw->nb[1] != uw->nb[1] || gw->nb[2] != uw->nb[2]) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) return false;
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !glu->src[0] || !glu->src[1]) return false;
    if (glu->src[0]->op != GGML_OP_MUL_MAT_ID || glu->src[1]->op != GGML_OP_MUL_MAT_ID) return false;
    if (glu->src[0]->src[0] != gw || glu->src[1]->src[0] != uw || glu->src[0]->src[1] != src1 || glu->src[1]->src[1] != src1 || glu->src[0]->src[2] != ids || glu->src[1]->src[2] != ids) return false;
    if (ggml_nelements(glu) != ggml_nelements(glu->src[0]) || glu->ne[0] != gw->ne[1]) return false;
    return ggml_cuda_mmb_supported_mmid(gw, src1, ids, glu->src[0], true) && ggml_cuda_mmb_supported_mmid(uw, src1, ids, glu->src[1], true);
}

void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) gw->ne[0], M = (int) gw->ne[1], E = (int) gw->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    const bool iq3s = gw->type == GGML_TYPE_IQ3_S;
    // strixllama: IQ3_S goes through mmb_tile_gemm_glu3; STRIX_MMB_GLU3=0 goes back to mmb_tile_gemm_glu (same results,
    // for A/B). glu3's tiles cost little besides their dequantization, so an expert of 32 rows or more is cheaper as one
    // big tile than as several small ones, each dequantizing the same weights again: its threshold is 32, not 128.
    static const bool glu3 = !getenv("STRIX_MMB_GLU3") || atoi(getenv("STRIX_MMB_GLU3")) != 0;
    const bool q4k = gw->type == GGML_TYPE_Q4_K;   // glu3 only: the older kernel has no Q4_K
    const bool iq4x = gw->type == GGML_TYPE_IQ4_XS;   // glu3 only, as Q4_K
    const bool use3 = (iq3s && glu3) || q4k || iq4x;
    constexpr int BN_SMALL = 32;
    const int thresh = use3 ? 32 : 128;
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((thresh + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, thresh);
    // strixllama: STRIX_MMB_GLU_DUMP=<file> appends every call's rows per expert ("n_rows n_experts count0 count1 ...",
    // what test-backend-ops' STRIX_MOE_IDS_FILE replays); it synchronises the stream, so it is for recording only
    static const char * dump = getenv("STRIX_MMB_GLU_DUMP");
    if (dump) {
        std::vector<int32_t> hb(E + 1);
        CUDA_CHECK(cudaMemcpyAsync(hb.data(), bounds.get(), (E + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (FILE * f = fopen(dump, "a")) {
            fprintf(f, "%d %d", n_rows, E);
            for (int e = 0; e < E; ++e) { fprintf(f, " %d", hb[e + 1] - hb[e]); }
            fputc('\n', f);
            fclose(f);
        }
    }
    uint16_t * Dh = ggml_cuda_mmb_slot_reserve(ctx, 2, glu, (size_t) n_rows * M);
    const bool store_f32 = !ggml_cuda_mmb_is_bf16_only(glu);
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_GLU fused gate/up+swiglu: M=%d K=%d rows=%d store_f32=%d type=%s\n", M, K, n_rows, (int) store_f32, ggml_type_name(gw->type));
    const uint8_t * Wg = (const uint8_t *) gw->data, * Wu = (const uint8_t *) uw->data; float * D = (float *) glu->data; const size_t eb = (size_t) gw->nb[2];
    dim3 gbig((M + 63) / 64, nbig_max), gsmall((M + 63) / 64, nsmall_max);
    auto launch_glu3 = [&](auto wq, auto pf) {
        constexpr int WQ = decltype(wq)::value, PFV = decltype(pf)::value;
        GGML_ASSERT((size_t) n_rows_x * K * 2 < ((size_t) 1 << 32));   // glu3's 32-bit activation offsets
        // the 128-token tile keeps one step in flight: two (220 VGPRs) measured 5% slower at 2K tokens
        mmb_routed_glu3_kernel<BN, 32, 32, WQ, 1><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_glu3_kernel<BN_SMALL, 16, 16, WQ, PFV><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    };
    using I1 = std::integral_constant<int, 1>; using I2 = std::integral_constant<int, 2>; using I3 = std::integral_constant<int, 3>; using I12 = std::integral_constant<int, 12>;
    using I4 = std::integral_constant<int, 4>;
    if (iq4x) {
        if (mmb_pf2()) launch_glu3(I4{}, I2{}); else launch_glu3(I4{}, I1{});
    } else if (use3 && q4k) {
        if (mmb_pf2()) launch_glu3(I12{}, I2{}); else launch_glu3(I12{}, I1{});
    } else if (use3) {
        if (mmb_pf2()) launch_glu3(I3{}, I2{}); else launch_glu3(I3{}, I1{});
    } else if (iq3s) {
        mmb_routed_glu_kernel<64, BN, 32, 32, 3><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_glu_kernel<64, BN_SMALL, 16, 16, 3><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    } else {
        mmb_routed_glu_kernel<64, BN, 32, 32><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmb_routed_glu_kernel<64, BN_SMALL, 16, 16><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Called from graph_optimize (outside stream capture): create the shadow for an eligible IQ4_NL dense weight.
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!w) return;
    if (mmb_is_resident_q6k(w)) {
        if (!mmb_shadow_q6k() || g_mmb_shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1], bytes = n * 2;
        if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { fprintf(stderr, "MMB_SHADOW cap reached; %s stays Q6_K\n", w->name); return; }
        uint16_t * buf = nullptr;
        if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { fprintf(stderr, "MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
        mmb_dq_q6k_bf16_kernel<<<(unsigned) ((n / 256 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 256);
        CUDA_CHECK(cudaGetLastError());
        g_mmb_shadow[w->data] = buf; g_mmb_shadow_bytes += bytes;
        static unsigned q6 = 0; if (q6++ < 3) fprintf(stderr, "MMB_SHADOW Q6_K %s [%lld x %lld] -> BF16 (%.1f MB total)\n", w->name, (long long) w->ne[0], (long long) w->ne[1], g_mmb_shadow_bytes / 1048576.0);
        return;
    }
    if (mmb_shadow_mode() != 1) return;               // mode 2: Q6_K only
    const bool concat = mmb_is_row_concat(w);
    if (!concat && !mmb_is_resident_iq4(w)) return;
    if (concat ? g_mmb_shadow_pair.count({w->src[0]->data, w->src[1]->data}) > 0 : g_mmb_shadow.count(w->data) > 0) return;
    const size_t n = (size_t) w->ne[0] * w->ne[1];
    const size_t bytes = n * 2;
    if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { static bool warned = false; if (!warned) { fprintf(stderr, "MMB_SHADOW cap reached at %.1f MB; further weights stay IQ4_NL\n", g_mmb_shadow_bytes / 1048576.0); warned = true; } return; }
    uint16_t * buf = nullptr;
    if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { fprintf(stderr, "MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
    if (concat) {
        const size_t n0 = (size_t) w->src[0]->ne[0] * w->src[0]->ne[1], n1 = (size_t) w->src[1]->ne[0] * w->src[1]->ne[1];
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n0 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[0]->data, buf, n0 / 32);
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n1 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[1]->data, buf + n0, n1 / 32);
        g_mmb_shadow_pair[{w->src[0]->data, w->src[1]->data}] = buf;
    } else {
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 32);
        g_mmb_shadow[w->data] = buf;
    }
    CUDA_CHECK(cudaGetLastError());
    g_mmb_shadow_bytes += bytes;
    static unsigned hits = 0; if (hits++ < 3 || (hits % 50) == 0) fprintf(stderr, "MMB_SHADOW %s [%lld x %lld] -> BF16 (%.1f MB total)\n", w->name, (long long) w->ne[0], (long long) w->ne[1], g_mmb_shadow_bytes / 1048576.0);
}
