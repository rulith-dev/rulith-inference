#include "common.cuh"
#include "dequantize.cuh"
#include "qsa.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));

#define QSA3_L2E 1.4426950408889634f
#define QSA3_G 4

static __device__ __forceinline__ uint16_t qsa3_f2h(const float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static __device__ __forceinline__ float qsa3_h2f(const uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

// ------------------------------------------------------------------------------------------------------------
// union builder, kernel A: per query row, check that the (validity-transformed) row is non-decreasing; if not,
// rank-sort it into srow[q] and set sflag[q]. Sentinel 0x7FFFFFFF for invalid (-1 / >= nk) entries.
#define QSA3_SENT 0x7FFFFFFF
__global__ __launch_bounds__(256) void qsa3_rows_kernel(
        const int * __restrict__ ids, const size_t i1, const int ns, const int nk, int * __restrict__ srow, int * __restrict__ sflag,
        const int ns_pow2) {
    extern __shared__ int ent[];
    __shared__ int unsorted;
    const int q = blockIdx.x, tid = threadIdx.x;
    const int * row = reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1);
    for (int j = tid; j < ns; j += 256) { const int key = row[j]; ent[j] = (key >= 0 && key < nk) ? key : QSA3_SENT; }
    for (int j = ns + tid; j < ns_pow2; j += 256) { ent[j] = QSA3_SENT; }
    if (tid == 0) { unsorted = 0; }
    __syncthreads();
    for (int j = tid; j + 1 < ns; j += 256) { if (ent[j] > ent[j+1]) { unsorted = 1; } }
    __syncthreads();
    if (unsorted) {
        // strixllama: a bitonic sort (after pwilkin/llama.cpp 38477e8c) instead of ranking every entry against every
        // other: O(ns log^2 ns) against O(ns^2), 4.2M comparisons a row at ns = 2048. The top-k rows arrive in score
        // order, so nearly every row takes this branch. Equal keys are equal values, so the sort need not be stable
        // and the row is the one the rank sort wrote; the padding is the sentinel, which sorts to the end.
        for (int k = 2; k <= ns_pow2; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < ns_pow2; i += 256) {
                    const int l = i ^ j;
                    if (l > i) {
                        const int a = ent[i], b = ent[l];
                        if ((a > b) == ((i & k) == 0)) { ent[i] = b; ent[l] = a; }
                    }
                }
                __syncthreads();
            }
        }
        int * dst = srow + (size_t) q * ns;
        for (int j = tid; j < ns; j += 256) { dst[j] = ent[j]; }
    }
    if (tid == 0) { sflag[q] = unsorted; }
}

// kernel B: one workgroup (128 lanes) per group of 4 queries. Lane = key sub-range; the range boundaries are the
// quantiles of the first non-empty row (block aligned), so the work per lane is balanced for the real (sorted,
// similar) rows. Each lane merges the 4 sorted rows inside its range: pass 1 counts, block-wide prefix sum,
// pass 2 writes.
#define QSA3_MERGE_LANES 128
static __device__ __forceinline__ int qsa3_get(const int * __restrict__ p, const int j, const int nk, const bool raw) {
    const int key = p[j];
    return raw ? ((key >= 0 && key < nk) ? key : QSA3_SENT) : key;
}
__global__ __launch_bounds__(QSA3_MERGE_LANES) void qsa3_merge_kernel(
        const int * __restrict__ ids, const size_t i1, const int n_q, const int ns, const int nk,
        const int * __restrict__ srow, const int * __restrict__ sflag,
        uint32_t * __restrict__ ublk, uint16_t * __restrict__ umask, int * __restrict__ ucount, const int cap) {
    extern __shared__ int rows_s[];
    __shared__ int scan_s[QSA3_MERGE_LANES];
    const int lane = threadIdx.x;
    const int g = blockIdx.x;
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        const int q = QSA3_G * g + qi;
        int * dst = rows_s + qi * ns;
        if (q < n_q) {
            const bool rw = sflag[q] == 0;
            const int * src = rw ? reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1) : srow + (size_t) q * ns;
            for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = qsa3_get(src, j, nk, rw); }
        } else {
            for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = QSA3_SENT; }
        }
    }
    __syncthreads();
    const int * rp[QSA3_G]; bool raw[QSA3_G]; int cnt[QSA3_G];
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        raw[qi] = false; rp[qi] = rows_s + qi * ns;
        int a = 0, b = ns;
        while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < QSA3_SENT) { a = mid + 1; } else { b = mid; } }
        cnt[qi] = a;
    }
    int kmax = -1, ref = -1;
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        if (cnt[qi] > 0) { kmax = max(kmax, qsa3_get(rp[qi], cnt[qi] - 1, nk, raw[qi])); if (ref < 0) { ref = qi; } }
    }
    uint32_t * ob = ublk + (size_t) g * cap;
    uint16_t * om = umask + (size_t) g * cap;
    if (kmax < 0) { if (lane == 0) { ucount[g] = 0; } return; }
    // lane range [lo, hi) in key space, block aligned, from the reference row's quantiles
    const int rc = cnt[ref];
    const int lo = lane == 0 ? 0 : (qsa3_get(rp[ref], (int) (((long long) rc * lane) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    const int hi = lane == QSA3_MERGE_LANES - 1 ? ((kmax + 4) & ~3)
                 : (qsa3_get(rp[ref], (int) (((long long) rc * (lane + 1)) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    int p[QSA3_G], pe[QSA3_G];
#pragma unroll
    for (int qi = 0; qi < QSA3_G; ++qi) {
        int a = 0, b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < lo) { a = mid + 1; } else { b = mid; } }
        p[qi] = a; b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < hi) { a = mid + 1; } else { b = mid; } }
        pe[qi] = a;
    }
    // pass 1: count union blocks in range
    int n = 0;
    {
        int pp[QSA3_G]; int head[QSA3_G];
#pragma unroll
        for (int qi = 0; qi < QSA3_G; ++qi) { pp[qi] = p[qi]; head[qi] = pp[qi] < pe[qi] ? qsa3_get(rp[qi], pp[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { pn += 4; }
                    else { do { ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    pp[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ++n;
        }
    }
    scan_s[lane] = n;
    __syncthreads();
    for (int off = 1; off < QSA3_MERGE_LANES; off <<= 1) {
        const int v = lane >= off ? scan_s[lane - off] : 0;
        __syncthreads();
        scan_s[lane] += v;
        __syncthreads();
    }
    const int total = scan_s[QSA3_MERGE_LANES - 1];
    int o = scan_s[lane] - n;
    {
        int head[QSA3_G];
#pragma unroll
        for (int qi = 0; qi < QSA3_G; ++qi) { head[qi] = p[qi] < pe[qi] ? qsa3_get(rp[qi], p[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
            uint32_t mk = 0;
#pragma unroll
            for (int qi = 0; qi < QSA3_G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = p[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { mk |= 0xFu << (4*qi); pn += 4; }
                    else { do { mk |= 1u << (4*qi + (rr[pn] & 3)); ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    p[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ob[o] = (uint32_t) b; om[o] = (uint16_t) mk; ++o;
        }
    }
    if (lane == QSA3_MERGE_LANES - 1) {
        int t = total;
        while (t & 3) { ob[t] = 0xFFFFFFFFu; om[t] = 0; ++t; }
        ucount[g] = t;
    }
}

struct qsa3_layout {
    size_t q1, q2, o1, o2, m1;
    int nk, n_q, gqa, cap;
    float scale;
};

__global__ __launch_bounds__(256) void qsa3_attn_kernel(
        const float * __restrict__ q, const uint16_t * __restrict__ pk, const uint16_t * __restrict__ pv,
        const uint16_t * __restrict__ mask, const uint32_t * __restrict__ ublk, const uint16_t * __restrict__ umask,
        const int * __restrict__ ucount, float * __restrict__ out, const qsa3_layout s) {
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y;
    __shared__ float4 part[3][8][64];
    __shared__ uint4  ptile[3][32];
    __shared__ float  alpha_s[48];
    __shared__ float  l_s[48];

    const size_t nblk = (size_t) s.nk / 4;
    const uint16_t * pkg = pk + (size_t) kvh * nblk * 1024;
    const uint16_t * pvg = pv + (size_t) kvh * nblk * 1024;
    const uint32_t * gblk = ublk + (size_t) g * s.cap;
    const uint16_t * gmsk = umask + (size_t) g * s.cap;
    const int nchunks = ucount[g] >> 2;

    v16s qf[3][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (48 * 32);   // 3 KB per wave, row stride 64 B
        const _Float16 hs = (_Float16) s.scale;
#pragma unroll
        for (int c = 0; c < 12; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < s.n_q) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * s.gqa + h) * s.q2 + (size_t) (32*w + dq) * 4);
            }
            const uint16_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
            const uint16_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        __syncthreads();
    }
    v8f O[3][2];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { O[i][t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;                       // owner state (waves 0..2): row 16w + r
    const int own_row = 16*w + r, own_qi = own_row / 12;
    const int own_query = QSA3_G * g + own_qi;
    const uint16_t * maskq = (mask && w < 3 && own_query < s.n_q)
        ? reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(mask) + (size_t) own_query * s.m1) : nullptr;

    auto load_desc = [&](const int c, uint4 & bb, uint32_t & m01, uint32_t & m23) {
        bb = *reinterpret_cast<const uint4 *>(gblk + 4*c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        m01 = mm.x; m23 = mm.y;
    };
    auto blk_of = [&](const uint4 & bb, const int j) -> int {
        const uint32_t v = j == 0 ? bb.x : j == 1 ? bb.y : j == 2 ? bb.z : bb.w;
        return v == 0xFFFFFFFFu ? 0 : (int) v;
    };
    auto load_k = [&](const uint4 & bb, v16s * kf) {
        const int kb = blk_of(bb, r >> 2);
        const uint16_t * krow = pkg + (size_t) kb * 1024 + (r & 3) * 16 + w * 128;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const uint4 lo  = *reinterpret_cast<const uint4 *>(krow + t*64);
            const uint4 hi4 = *reinterpret_cast<const uint4 *>(krow + t*64 + 8);
            kf[t] = __builtin_bit_cast(v16s, (uint4[2]){lo, hi4});
        }
    };

    uint4 bb = make_uint4(0, 0, 0, 0);
    uint32_t m01 = 0, m23 = 0;
    v16s kf[2];
    if (nchunks > 0) { load_desc(0, bb, m01, m23); load_k(bb, kf); }

    for (int c = 0; c < nchunks; ++c) {
        v8f sc[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { sc[i][e] = 0.f; }
#pragma unroll
            for (int t = 0; t < 2; ++t) { sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[i][t], sc[i]); }
        }
        v16s vf[2];
        {
            const int kb0 = blk_of(bb, 0), kb1 = blk_of(bb, 1), kb2 = blk_of(bb, 2), kb3 = blk_of(bb, 3);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                const uint2 v0 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb0 * 1024 + d * 4);
                const uint2 v1 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb1 * 1024 + d * 4);
                const uint2 v2 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb2 * 1024 + d * 4);
                const uint2 v3 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb3 * 1024 + d * 4);
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){v0, v1, v2, v3});
            }
        }
        // --- prefetch next chunk's descriptor and K fragments ---
        uint4 nbb = make_uint4(0, 0, 0, 0);
        uint32_t nm01 = 0, nm23 = 0;
        v16s kfn[2];
        if (c + 1 < nchunks) { load_desc(c + 1, nbb, nm01, nm23); load_k(nbb, kfn); }
        else { kfn[0] = kf[0]; kfn[1] = kf[1]; }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i != w) {
                part[i][w][lane*2+0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2+1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        if (w < 3) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { scf[e] = 0.f; }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { scf[e] += sc[w][e]; }
                } else {
                    const float4 a = part[w][w2][lane*2+0], b = part[w][w2][lane*2+1];
                    scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w; scf[4] += b.x; scf[5] += b.y; scf[6] += b.z; scf[7] += b.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1, cc = 2*(e & 1) + hi;
                const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xFFFFu;
                const bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
                if (maskq && kv) { scf[e] += qsa3_h2f(maskq[4 * blk_of(bb, j) + cc]); }
                if (!kv) { scf[e] = -INFINITY; }
            }
            float mloc = scf[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) { mloc = fmaxf(mloc, scf[e]); }
            mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
            const float mnew  = fmaxf(m, mloc);
            const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * QSA3_L2E);
            float p[8], ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * QSA3_L2E); ls += p[e]; }
            ls += __shfl_xor(ls, 16);
            l = l * alpha + ls;
            m = mnew;
            uint32_t pp[4], po[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                pp[e2] = (uint32_t) qsa3_f2h(p[2*e2]) | ((uint32_t) qsa3_f2h(p[2*e2+1]) << 16);
                po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
            }
            uint16_t ph[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint16_t mine  = (e & 1) ? (uint16_t) (pp[e>>1] >> 16) : (uint16_t) (pp[e>>1] & 0xffffu);
                const uint16_t other = (e & 1) ? (uint16_t) (po[e>>1] >> 16) : (uint16_t) (po[e>>1] & 0xffffu);
                ph[2*e]   = hi ? other : mine;
                ph[2*e+1] = hi ? mine  : other;
            }
            if (hi == 0) {
                ptile[w][r*2+0] = make_uint4((uint32_t) ph[0] | ((uint32_t) ph[1] << 16), (uint32_t) ph[2] | ((uint32_t) ph[3] << 16),
                                             (uint32_t) ph[4] | ((uint32_t) ph[5] << 16), (uint32_t) ph[6] | ((uint32_t) ph[7] << 16));
                ptile[w][r*2+1] = make_uint4((uint32_t) ph[8] | ((uint32_t) ph[9] << 16), (uint32_t) ph[10] | ((uint32_t) ph[11] << 16),
                                             (uint32_t) ph[12] | ((uint32_t) ph[13] << 16), (uint32_t) ph[14] | ((uint32_t) ph[15] << 16));
                alpha_s[own_row] = alpha;
            }
        }
        __syncthreads();
        float al[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) { al[i] = alpha_s[16*i + r]; }
        if (__ballot(al[0] != 1.f || al[1] != 1.f || al[2] != 1.f)) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int t = 0; t < 2; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) { O[i][t][e] *= al[i]; }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){ptile[i][r*2+0], ptile[i][r*2+1]});
#pragma unroll
            for (int t = 0; t < 2; ++t) { O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, O[i][t]); }
        }
        kf[0] = kfn[0]; kf[1] = kfn[1];
        bb = nbb; m01 = nm01; m23 = nm23;
    }

    if (w < 3 && hi == 0) { l_s[own_row] = l; }
    __syncthreads();
    // output: per tile, stage [16 rows x 32 dims] F32 per wave in LDS (2 KB), then coalesced 16-byte row stores
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f; }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            if (query < s.n_q) {
                float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) (kvh * s.gqa + h) * s.o1) + 32*w + dq;
                *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
            }
        }
        __syncthreads();
    }
}

static bool enabled(const char * name) { const char * value = getenv(name); return value && atoi(value) != 0; }

// strixllama: the kernel reads K and V only through their packed f16 copies (src 6 and 7), so the cache itself may
// be Q8_0 as well - the graph dequantizes it into them
static bool qsa3_kv_type_ok(const ggml_tensor * t) {
    return (t->type == GGML_TYPE_F16 && t->nb[0] == 2) || t->type == GGML_TYPE_Q8_0;
}

bool ggml_cuda_flash_attn_ext_qsa_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    const auto * packed = dst->src[6], * pv = dst->src[7];
    if (!enabled("LLAMA_QSA_FA_V3") || !q || !k || !v || !ids || dst->src[4] || !packed || !pv ||
        !GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc)) { return false; }
    if (q->ne[1] < 128 && !enabled("QSA3_FORCE")) { return false; }
    // strixllama: the union list numbers blocks of 4 cells in 32 bits since 0.2.8. In 16 bits it stopped at 262140
    // cells, and a larger window went to the generic kernels, which ignore the selection - dense attention over the
    // whole window, and without the mask a maskless graph leaves out, over other conversations and later positions
    // too. A ubatch of several sequences spans all of their cells (0.2.7's ragged ubatches, a 512K pool).
    float bias, softcap; memcpy(&bias, (const char *) dst->op_params + 4, 4); memcpy(&softcap, (const char *) dst->op_params + 8, 4);
    if (bias != 0 || softcap != 0 || q->type != GGML_TYPE_F32 || !qsa3_kv_type_ok(k) || !qsa3_kv_type_ok(v) ||
        dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
        q->ne[1] < 1 || q->ne[1] > INT_MAX || k->ne[1] < 1 || k->ne[1] > INT_MAX - 3 || k->ne[1] % 4 ||
        k->ne[2] < 1 || q->ne[2] != 12*k->ne[2] || v->ne[2] != k->ne[2] || v->ne[1] != k->ne[1] ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || q->nb[0] != 4 ||
        ids->nb[0] != 4 || ids->ne[1] < q->ne[1] || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[0] < 1 || ids->ne[0] > 2560 ||
        ids->nb[1] % 4 || uintptr_t(ids->data) % 4 || q->nb[1] % 16 || q->nb[2] % 16 || uintptr_t(q->data) % 16 ||
        !ggml_is_contiguous(dst)) { return false; }
    if (packed->type != GGML_TYPE_F16 || !ggml_is_contiguous(packed) || packed->ne[0] != 16 || packed->ne[1] != 4 ||
        packed->ne[2] != 16 || packed->ne[3] != k->ne[1]/4*k->ne[2] || uintptr_t(packed->data) % 16) { return false; }
    if (pv->type != GGML_TYPE_F16 || !ggml_is_contiguous(pv) || pv->ne[0] != 4 || pv->ne[1] != 256 ||
        pv->ne[2] != k->ne[1]/4*k->ne[2] || pv->ne[3] != 1 || uintptr_t(pv->data) % 8) { return false; }
    return !m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] >= k->ne[1] && m->ne[1] >= q->ne[1] && m->ne[2] == 1 && m->ne[3] == 1);
}

void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * m = dst->src[3], * ids = dst->src[5], * pk = dst->src[6], * pv = dst->src[7];
    float scale; memcpy(&scale, dst->op_params, 4);
    const int n_q = (int) q->ne[1], ns = (int) ids->ne[0], nk = (int) k->ne[1];
    const int ngroups = (n_q + QSA3_G - 1) / QSA3_G;
    const int cap = (QSA3_G * ns + 3) & ~3;
    ggml_cuda_pool_alloc<uint32_t> ublk(ctx.pool(), (size_t) ngroups * cap);
    ggml_cuda_pool_alloc<uint16_t> umask(ctx.pool(), (size_t) ngroups * cap);
    ggml_cuda_pool_alloc<int>      ucount(ctx.pool(), (size_t) ngroups);
    ggml_cuda_pool_alloc<int>      srow(ctx.pool(), (size_t) n_q * ns);
    ggml_cuda_pool_alloc<int>      sflag(ctx.pool(), (size_t) n_q);
    {
        int ns_pow2 = 1;
        while (ns_pow2 < ns) { ns_pow2 <<= 1; }
        // the bitonic sort pads the row to a power of two: up to twice the shared memory the rank sort took
        GGML_ASSERT((size_t) ns_pow2 * sizeof(int) <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo &&
                    "QSA selection row too large for the shared memory sort");
        const ggml_cuda_kernel_launch_params launch(dim3(n_q), dim3(256), (size_t) ns_pow2 * sizeof(int), ctx.stream());
        ggml_cuda_kernel_launch(qsa3_rows_kernel, launch, (const int *) ids->data, ids->nb[1], ns, nk, srow.get(), sflag.get(), ns_pow2);
        CUDA_CHECK(cudaGetLastError());
        const ggml_cuda_kernel_launch_params launch2(dim3(ngroups), dim3(QSA3_MERGE_LANES), (size_t) QSA3_G * ns * sizeof(int), ctx.stream());
        ggml_cuda_kernel_launch(qsa3_merge_kernel, launch2, (const int *) ids->data, ids->nb[1], n_q, ns, nk,
                                (const int *) srow.get(), (const int *) sflag.get(), ublk.get(), umask.get(), ucount.get(), cap);
        CUDA_CHECK(cudaGetLastError());
    }
    qsa3_layout layout{q->nb[1], q->nb[2], dst->nb[1], dst->nb[2], m ? m->nb[1] : 0, nk, n_q, 12, cap, scale};
    const ggml_cuda_kernel_launch_params launch(dim3(ngroups, k->ne[2]), dim3(256), 0, ctx.stream());
    ggml_cuda_kernel_launch(qsa3_attn_kernel, launch, (const float *) q->data, (const uint16_t *) pk->data, (const uint16_t *) pv->data,
                            m ? (const uint16_t *) m->data : nullptr, (const uint32_t *) ublk.get(), (const uint16_t *) umask.get(),
                            (const int *) ucount.get(), (float *) dst->data, layout);
    CUDA_CHECK(cudaGetLastError());
}

// strixllama: the packed K and V layouts in one pass from the cache. The graph (models/pack.inc) builds each as
// CONT(permuted cache view) -> RESHAPE -> PERMUTE -> CONT: two full copies of the layer's cache, ~8 ms a layer at
// 64K context through the generic copy kernel. The two paths below write the same bytes from the cache view directly.
//   keys   [16, 4, 16, heads * n_kv / 4]: (a, b, c, h * nblk + n) = K[dim 16c + a][cell 4n + b][head h]
//   values [4, 256, heads * n_kv / 4]:    (b, d, h * nblk + n)     = V[dim d][cell 4n + b][head h]
static __global__ void qsa_pack_keys_kernel(const char * __restrict__ src, uint4 * __restrict__ dst, const int64_t nblk,
        const int64_t nb_cell, const int64_t nb_head, const int64_t n_chunks) {
    const int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one 32-byte run of 16 dims
    if (q >= n_chunks) {
        return;
    }
    const int     b = (int) (q & 3), c = (int) ((q >> 2) & 15);
    const int64_t m = q >> 6, h = m / nblk, n = m - h * nblk;
    const uint4 * s = (const uint4 *) (src + h * nb_head + (4 * n + b) * nb_cell + c * 32);
    dst[2 * q]     = s[0];
    dst[2 * q + 1] = s[1];
}

static __global__ void qsa_pack_values_kernel(const char * __restrict__ src, uint4 * __restrict__ dst, const int64_t nblk,
        const int64_t nb_cell, const int64_t nb_head, const int64_t n_pairs) {
    const int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // dims 2d, 2d+1 of the block's four cells
    if (q >= n_pairs) {
        return;
    }
    const int     d2 = (int) (q & 127);
    const int64_t m  = q >> 7, h = m / nblk, n = m - h * nblk;
    const char *  s  = src + h * nb_head + 4 * n * nb_cell + d2 * 4;
    const uint32_t v0 = *(const uint32_t *) (s);
    const uint32_t v1 = *(const uint32_t *) (s + nb_cell);
    const uint32_t v2 = *(const uint32_t *) (s + 2 * nb_cell);
    const uint32_t v3 = *(const uint32_t *) (s + 3 * nb_cell);
    // out: dim 2d cells 0..3, then dim 2d+1 cells 0..3
    uint4 o;
    o.x = (v0 & 0xffffu) | (v1 << 16);
    o.y = (v2 & 0xffffu) | (v3 << 16);
    o.z = (v0 >> 16) | (v1 & 0xffff0000u);
    o.w = (v2 >> 16) | (v3 & 0xffff0000u);
    dst[q] = o;
}

// strixllama: the same layouts from a Q8_0 cache, dequantized value for value as get_rows / cpy do
// (dequantize_q8_0, then __float2half), so it packs to the bytes cast(F16) + pack gave
template <ggml_type T>
static __device__ __forceinline__ half qsa_deq(const char * row, const int d) {
    static_assert(T == GGML_TYPE_Q8_0, "a Q8_0 cache");
    float2 v;
    dequantize_q8_0(row, d / QK8_0, (d % QK8_0) & ~1, v);
    return __float2half((d & 1) ? v.y : v.x);
}

template <ggml_type T>
static __global__ void qsa_pack_keys_q_kernel(const char * __restrict__ src, half * __restrict__ dst, const int64_t nblk,
        const int64_t nb_cell, const int64_t nb_head, const int64_t n_chunks) {
    const int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one run of 16 dims
    if (q >= n_chunks) {
        return;
    }
    const int     b = (int) (q & 3), c = (int) ((q >> 2) & 15);
    const int64_t m = q >> 6, h = m / nblk, n = m - h * nblk;
    const char * row = src + h * nb_head + (4 * n + b) * nb_cell;
    half v[16];
#pragma unroll
    for (int a = 0; a < 16; ++a) {
        v[a] = qsa_deq<T>(row, 16 * c + a);
    }
    uint4 w[2];
    memcpy(w, v, sizeof(w));
    uint4 * o = (uint4 *) (dst + 16 * q);
    o[0] = w[0];
    o[1] = w[1];
}

template <ggml_type T>
static __global__ void qsa_pack_values_q_kernel(const char * __restrict__ src, half * __restrict__ dst, const int64_t nblk,
        const int64_t nb_cell, const int64_t nb_head, const int64_t n_pairs) {
    const int64_t q = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // dims 2d, 2d+1 of the block's four cells
    if (q >= n_pairs) {
        return;
    }
    const int     d2 = (int) (q & 127);
    const int64_t m  = q >> 7, h = m / nblk, n = m - h * nblk;
    const char *  s  = src + h * nb_head + 4 * n * nb_cell;
    half v[8];   // dim 2d cells 0..3, then dim 2d+1 cells 0..3
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        v[b]     = qsa_deq<T>(s + b * nb_cell, 2 * d2);
        v[4 + b] = qsa_deq<T>(s + b * nb_cell, 2 * d2 + 1);
    }
    uint4 w;
    memcpy(&w, v, sizeof(w));
    *(uint4 *) (dst + 8 * q) = w;
}

bool ggml_cuda_qsa_pack_supported(const ggml_tensor * op) {
    const ggml_tensor * s = op->src[0];
    if (s->ne[0] != 256 || s->ne[2] % 4 != 0 || s->ne[3] != 1 || s->nb[0] != ggml_type_size(s->type)) {
        return false;
    }
    if (s->type == GGML_TYPE_F16) {
        return s->nb[1] % 16 == 0 && s->nb[2] % 16 == 0;
    }
    return s->type == GGML_TYPE_Q8_0;
}

void ggml_cuda_qsa_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * s = dst->src[0];
    const bool values = ggml_get_op_params_i32(dst, 0) != 0;
    const int64_t heads = s->ne[1], nblk = s->ne[2] / 4, N = heads * nblk;
    const int64_t nb_head = s->nb[1], nb_cell = s->nb[2];
    GGML_ASSERT(ggml_nelements(dst) == 256 * s->ne[2] * heads && ggml_is_contiguous(dst));
    const char * src = (const char *) s->data;
    cudaStream_t st = ctx.stream();
    const int64_t n = values ? N * 128 : N * 64;
    const unsigned grid = (unsigned) ((n + 255) / 256);
    switch (s->type) {
        case GGML_TYPE_F16:
            GGML_ASSERT(((uintptr_t) src) % 16 == 0 && ((uintptr_t) dst->data) % 16 == 0);
            if (values) {
                qsa_pack_values_kernel<<<grid, 256, 0, st>>>(src, (uint4 *) dst->data, nblk, nb_cell, nb_head, n);
            } else {
                qsa_pack_keys_kernel<<<grid, 256, 0, st>>>(src, (uint4 *) dst->data, nblk, nb_cell, nb_head, n);
            }
            break;
        case GGML_TYPE_Q8_0:
            if (values) {
                qsa_pack_values_q_kernel<GGML_TYPE_Q8_0><<<grid, 256, 0, st>>>(src, (half *) dst->data, nblk, nb_cell, nb_head, n);
            } else {
                qsa_pack_keys_q_kernel<GGML_TYPE_Q8_0><<<grid, 256, 0, st>>>(src, (half *) dst->data, nblk, nb_cell, nb_head, n);
            }
            break;
        default:
            GGML_ABORT("qsa_pack: unsupported cache type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// the tensors' memory [first byte, one past the last) as a view reaches it
static void qsa_extent(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    size_t last = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        last += (size_t) (t->ne[d] - 1) * t->nb[d];
    }
    lo = (uintptr_t) t->data;
    hi = lo + last + ggml_type_size(t->type);
}

static bool qsa_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    uintptr_t alo, ahi, blo, bhi;
    qsa_extent(a, alo, ahi);
    qsa_extent(b, blo, bhi);
    return alo < bhi && blo < ahi;
}

// nodes[i] = CONT(permuted cache view), then RESHAPE, PERMUTE, CONT(packed); the graph's structure only (graph_optimize
// runs it before memory is assigned)
static bool qsa_pack_match(const ggml_cgraph * g, const int i, const ggml_tensor ** src, ggml_tensor ** out, bool * keys) {
    static const bool on = !getenv("STRIX_QSA_PACK_FUSE") || atoi(getenv("STRIX_QSA_PACK_FUSE")) != 0;
    if (!on || i + 3 >= g->n_nodes) {
        return false;
    }
    const ggml_tensor * c0 = g->nodes[i], * r = g->nodes[i + 1], * p = g->nodes[i + 2];
    ggml_tensor * o = g->nodes[i + 3];
    if (c0->op != GGML_OP_CONT || r->op != GGML_OP_RESHAPE || p->op != GGML_OP_PERMUTE || o->op != GGML_OP_CONT ||
            r->src[0] != c0 || p->src[0] != r || o->src[0] != p || c0->type != GGML_TYPE_F16 || o->type != GGML_TYPE_F16 ||
            !ggml_is_contiguous(c0) || !ggml_is_contiguous(o) || ggml_node_get_use_count(g, i) != 1 ||
            ggml_node_get_use_count(g, i + 1) != 1 || ggml_node_get_use_count(g, i + 2) != 1) {
        return false;
    }
    // the cache view: [256 dims, n_kv cells, heads], dims contiguous
    const ggml_tensor * s = c0->src[0];
    const int64_t n_kv = c0->ne[1], heads = c0->ne[2];
    if (s->type != GGML_TYPE_F16 || c0->ne[0] != 256 || c0->ne[3] != 1 || n_kv % 4 != 0 || n_kv < 4 || heads < 1 ||
            !ggml_are_same_shape(s, c0) || s->nb[0] != 2 || s->nb[1] % 16 != 0 || s->nb[2] % 16 != 0) {
        return false;
    }
    const int64_t N = n_kv / 4 * heads;
    const bool is_keys = o->ne[0] == 16 && o->ne[1] == 4 && o->ne[2] == 16 && o->ne[3] == N &&
        r->ne[0] == 16 && r->ne[1] == 16 && r->ne[2] == 4 && r->ne[3] == N &&
        ggml_get_op_params_i32(p, 0) == 0 && ggml_get_op_params_i32(p, 1) == 2 && ggml_get_op_params_i32(p, 2) == 1 &&
        ggml_get_op_params_i32(p, 3) == 3;
    const bool is_values = o->ne[0] == 4 && o->ne[1] == 256 && o->ne[2] == N && o->ne[3] == 1 &&
        r->ne[0] == 256 && r->ne[1] == 4 && r->ne[2] == N && r->ne[3] == 1 &&
        ggml_get_op_params_i32(p, 0) == 1 && ggml_get_op_params_i32(p, 1) == 0 && ggml_get_op_params_i32(p, 2) == 2 &&
        ggml_get_op_params_i32(p, 3) == 3;
    if (!is_keys && !is_values) {
        return false;
    }
    *src  = s;
    *out  = o;
    *keys = is_keys;
    return true;
}

bool ggml_cuda_qsa_pack_deps(const ggml_cgraph * g, const int i, ggml_tensor ** src_root, ggml_tensor ** out) {
    const ggml_tensor * s;
    bool keys;
    if (!qsa_pack_match(g, i, &s, out, &keys)) {
        return false;
    }
    *src_root = const_cast<ggml_tensor *>(s->view_src ? s->view_src : s);
    return true;
}

// returns the nodes consumed after nodes[i], or 0
int ggml_cuda_qsa_pack_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, const int i) {
    const ggml_tensor * s;
    ggml_tensor * out;
    bool keys;
    if (!qsa_pack_match(g, i, &s, &out, &keys) || ((uintptr_t) s->data) % 16 != 0 || ((uintptr_t) out->data) % 16 != 0) {
        return 0;
    }
    // the unfused path finished reading the source before the pack's output existed; the allocator may reuse the
    // source for it unless graph_optimize kept the source (ggml_cuda_qsa_pack_deps) - never read and write one memory
    if (qsa_overlap(s, out)) {
        return 0;
    }
    const ggml_tensor * c0 = g->nodes[i];
    const int64_t n_kv = c0->ne[1], heads = c0->ne[2];
    const int64_t nblk = n_kv / 4, N = nblk * heads;
    GGML_ASSERT(ggml_nelements(out) == 256 * n_kv * heads);
    if (keys) {
        const int64_t n_chunks = N * 64;   // 16 dims each, 1024 values a block
        qsa_pack_keys_kernel<<<(unsigned) ((n_chunks + 255) / 256), 256, 0, ctx.stream()>>>(
            (const char *) s->data, (uint4 *) out->data, nblk, (int64_t) s->nb[1], (int64_t) s->nb[2], n_chunks);
    } else {
        const int64_t n_pairs = N * 128;
        qsa_pack_values_kernel<<<(unsigned) ((n_pairs + 255) / 256), 256, 0, ctx.stream()>>>(
            (const char *) s->data, (uint4 *) out->data, nblk, (int64_t) s->nb[1], (int64_t) s->nb[2], n_pairs);
    }
    CUDA_CHECK(cudaGetLastError());
    return 3;
}

// strixllama: the attention output gate, CONT(view of Qcur_full) then SIGMOID, as one pass: the view's rows (the gate
// halves of the interleaved q|gate projection) read with float4s, the sigmoid applied, written contiguous. The same
// expression as op_sigmoid in unary.cu, so the values are the unfused ones; half the traffic (0.83 + 0.86 ms a layer
// at 2K tokens before).
static __global__ void strided_sigmoid_kernel(const char * __restrict__ src, float4 * __restrict__ dst, const int64_t ne0_4,
        const int64_t ne1, const int64_t ne2, const int64_t nb1, const int64_t nb2, const int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    const int64_t i0 = i % ne0_4, r = i / ne0_4, i1 = r % ne1, i2 = r / ne1;
    const float4 x = *(const float4 *) (src + i2 * nb2 + i1 * nb1 + i0 * 16);
    float4 y;
    y.x = 1.0f / (1.0f + expf(-x.x));
    y.y = 1.0f / (1.0f + expf(-x.y));
    y.z = 1.0f / (1.0f + expf(-x.z));
    y.w = 1.0f / (1.0f + expf(-x.w));
    dst[i] = y;
}

// CONT(strided F32 view) whose only reader is a SIGMOID; the graph's structure only
static bool cont_sigmoid_match(const ggml_cgraph * g, const int i, const ggml_tensor ** src, ggml_tensor ** out) {
    static const bool on = !getenv("STRIX_CONT_SIGMOID") || atoi(getenv("STRIX_CONT_SIGMOID")) != 0;
    if (!on || i + 1 >= g->n_nodes) {
        return false;
    }
    const ggml_tensor * c = g->nodes[i];
    ggml_tensor * sg = g->nodes[i + 1];
    if (c->op != GGML_OP_CONT || sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID ||
            sg->src[0] != c || ggml_node_get_use_count(g, i) != 1 || c->type != GGML_TYPE_F32 || sg->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(c) || !ggml_is_contiguous(sg) || ggml_nelements(c) != ggml_nelements(sg)) {
        return false;
    }
    const ggml_tensor * s = c->src[0];
    // a view of up to three dimensions, rows contiguous, in the CONT's (row-major) element order
    if (s->type != GGML_TYPE_F32 || s->ne[3] != 1 || s->nb[0] != 4 || s->ne[0] % 4 != 0 || s->nb[1] % 16 != 0 ||
            s->nb[2] % 16 != 0 || ggml_nelements(s) != ggml_nelements(c)) {
        return false;
    }
    *src = s;
    *out = sg;
    return true;
}

bool ggml_cuda_cont_sigmoid_deps(const ggml_cgraph * g, const int i, ggml_tensor ** src_root, ggml_tensor ** out) {
    const ggml_tensor * s;
    if (!cont_sigmoid_match(g, i, &s, out)) {
        return false;
    }
    *src_root = const_cast<ggml_tensor *>(s->view_src ? s->view_src : s);
    return true;
}

int ggml_cuda_cont_sigmoid_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, const int i) {
    const ggml_tensor * s;
    ggml_tensor * sg;
    if (!cont_sigmoid_match(g, i, &s, &sg) || ((uintptr_t) s->data) % 16 != 0 || ((uintptr_t) sg->data) % 16 != 0 ||
            qsa_overlap(s, sg)) {   // see ggml_cuda_qsa_pack_fuse
        return 0;
    }
    const int64_t n4 = ggml_nelements(sg) / 4;
    strided_sigmoid_kernel<<<(unsigned) ((n4 + 255) / 256), 256, 0, ctx.stream()>>>((const char *) s->data, (float4 *) sg->data,
        s->ne[0] / 4, s->ne[1], s->ne[2], (int64_t) s->nb[1], (int64_t) s->nb[2], n4);
    CUDA_CHECK(cudaGetLastError());
    return 1;
}
