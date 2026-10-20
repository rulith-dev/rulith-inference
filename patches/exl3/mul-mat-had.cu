// GGML_OP_MUL_MAT_HAD: y = rot_out * H(W^T H(rot_in * x)), see ggml_mul_mat_had()
// strixllama: from ranma.cpp (saturnsky), MIT license, without its expert cache and mapped-host aliases (our host
// experts are read through their own pointers)
// EXL3 trellis format and codebooks (mul1, mcg, 3inst): ExLlamaV3, Copyright (c) 2025 Turboderp, MIT license
//
// 1. prologue: one block per output row (token, or token x expert slot with ids) writes H(rot_in * x) once: F32 for
//    the GEMV, or for the WMMA GEMM in fp16, scaled by a power of 2 that keeps the values in the fp16 normal range.
// 2. GEMV (few rows per weight) or WMMA GEMM (RDNA4): a block owns one 128-row group of one weight matrix, the
//    H128 output block. Each 16x16 trellis tile is decoded once per block and used for all rows of the block.
// 3. epilogue in the same block (or in the last block of a k split): output Hadamard, rot_out and the row scale.
// Invalid expert ids are clamped to a valid expert (wrong rows, never a read outside the weight); there are no
// device-side checks, the op and its shapes are checked when the graph is built.
//
// Input precision, from the src1 precision of the op (ggml_prec_set_src(op, prec, 1)):
// - GGML_PREC_F16 or lower (the F16 mode): the WMMA GEMM takes one fp16 value of the scaled row, the precision class
//   of the official ExLlamaV3 kernels, which take fp16 activations.
// - GGML_PREC_F32: no GEMM; every row runs the GEMV with F32 input (after the routing pass in items of 8 rows), so
//   the product keeps the fp32 grade.
// - not set: MMH_PREC_DEFAULT; GGML_CUDA_MUL_MAT_HAD_PREC = f32 | f16 sets it for the process (to test the F32 path).
// The GEMV with EXL3 weights takes fp16 inputs in the F16 mode (v_dot2; for mul1 the codebook's affine map is applied
// once to the sums), except a dense op of one row; otherwise F32 inputs.
// The CPU backend always computes the fp32 grade.

#include "mul-mat-had.cuh"

#include <string>
#include <vector>
#include "unary.cuh"


#include <algorithm>
#include <cfloat>
#include <map>
#include <mutex>

// strixllama: a block barrier that waits for LDS traffic only (mmb.cu's mmb_lds_sync): __syncthreads() also waits for
// every outstanding global load (vmcnt(0)), so the next step's prefetched weights and rows had to land before the barrier
// that follows their issue - the prefetch hid nothing. The GEMM steps share data only through LDS.
static __device__ __forceinline__ void mmh_lds_sync() {
#if defined(__HIP_PLATFORM_AMD__)
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

#define MMH_NG          128 // output columns per block, one H128 output block
#define MMH_GEMV_TOKENS   8 // GEMV without a routing pass up to this many tokens
#define MMH_GEMV_LIST   512 // most rows one GEMV block collects without a routing pass
#define MMH_MAX_EXPERTS 4096 // weight matrices the routing pass counts in LDS
#define MMH_FUSED_LDS   49152 // LDS of a GEMV block that stages its own input (fused)
#define MMH_FUSED_MAX_BLOCKS 512 // fused GEMV up to this many blocks with work (each block transforms its own input)
#define MMH_GEMV_WAVES  640  // GEMV waves the k split aims at (strixllama: 1280 split too finely on gfx1151: decode +9%)
#define MMH_GEMV_WAVES_1ROW 320 // the same for a dense op of one row (single-stream decode): z / ssm_out -4..6%; at 2+ rows
                               // the smaller split is slower (ssm_out 76 -> 115 us at 2 rows)
#define MMH_GEMV_MIN_KSLICES 16 // fewest 16-value k slices per GEMV wave from the k split
#define MMH_GEMV_MAX_SPLIT 64 // largest k split
// rows of one expert per GEMV pass with ids and several tokens: 2 for 2 tokens, 4 for 3 or more (the 4-row instance decodes a
// tile once for up to 4 rows; for 2 tokens it measured 3-4% slower in VRAM, from 3 tokens it is 6-9% faster with the verify
// batch sharing of GLM-5.3 and 0.5-5% faster without sharing, in VRAM and host memory; 8 rows are slower in VRAM)
#define MMH_GEMV_IDS_ROWS_2TOK 2
#define MMH_GEMV_IDS_ROWS      4
// strixllama: the wide GEMM of ops with ids (mmh_gemm_wide): rows per block and groups per block
#define MMH_WIDE_MT 64
#define MMH_WIDE_G  5
#define MMH_WIDE_COAL true
// strixllama: the tiled GEMM of dense ops (mmh_gemm_t) from this many rows on, for weights of up to this many fp16 bytes
#define MMH_T_MIN_ROWS 256
#define MMH_T_MAX_WBYTES (64ll << 20)

// src1 precision of an op that does not set one (GGML_PREC_F32 = F32 GEMV only, GGML_PREC_F16 = fp16 GEMM)
#define MMH_PREC_DEFAULT GGML_PREC_F16

struct mmh_params {
    int64_t k;       // weight ne0
    int64_t n_exp;   // weight ne2
    int64_t nb01;
    int64_t nb02;

    int64_t n_rows;  // ne1*ne2*ne3 of dst, < 2^30
    int     ne1;
    int     ne2;
    int64_t nb1;
    int64_t nb2;
    int64_t nb3;

    int     ne11;
    int     bcast;   // without ids: rows of b per weight matrix along dim 2
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    const int32_t * ids;
    int64_t ids_nb0;
    int64_t ids_nb1;

    const float * x;
    const char  * rot_in;
    int64_t rin_nb1;
    const char  * rot_out;
    int64_t rout_nb0;
    int64_t rout_nb1;
    float * dst;

    int rin_f16;
    int rout_f16;
    int had;
    float hscale;    // 1/sqrt(had)

    // strixllama: GLU epilogue (ggml_cuda_mul_mat_had_glu): dst is the GLU's output and glu_other the output of the
    // other product, in the same layout; glu 1: this product is the gate (dst = silu(y)*other), 2: the up (silu(other)*y)
    const float * glu_other;
    int glu;
    // strixllama: routing-weight epilogue (ggml_cuda_mul_mat_had_rowmul): dst is the MUL's output and row_w the weights
    // [1, ne1, ne2] (bytes per slot and per token), each stored value multiplied by its row's weight
    const float * row_w;
    int64_t row_w_nb1;
    int64_t row_w_nb2;

    // strixllama: the staged rows of mmh_gemm_et: row r at sorted position pos[r] in panels of 16 rows (mmh_xh_panel)
    const int * pos;
    int panel_rows;  // the prologue writes row r at panel position r (no pos)
    unsigned long long * trace; // strixllama (dev): STRIX_MMH_TRACE, the GEMV's block timeline (start, end, cu | dup << 16)
    int tok_rows;    // mmh_gemm_et: one staged row per token for the tok_rows rows (expert slots) of the token

    // strixllama: per-expert widths (ggml_mul_mat_had_classes): expert e is in class cmap[e] >> 24 at index
    // cmap[e] & 0xFFFFFF; one launch per class, with that class's weight (nb01, nb02, type) and cls; ids, rotations and
    // n_exp stay global
    const int32_t * cmap;
    int cls;
};

static __device__ __forceinline__ int mmh_expert(const mmh_params & p, const int r) {
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    int e;
    if (p.ids) {
        e = *(const int32_t *) ((const char *) p.ids + i1*p.ids_nb0 + i2*p.ids_nb1);
    } else {
        e = i2 / p.bcast;
    }
    // invalid ids give wrong rows, never an access outside the weight
    return e < 0 ? 0 : (e >= p.n_exp ? (int) p.n_exp - 1 : e);
}

static __device__ __forceinline__ const float * mmh_src_row(const mmh_params & p, const int r) {
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    const int i3 = r / (p.ne1*p.ne2);
    if (p.ids) {
        return (const float *) ((const char *) p.x + (i1 % p.ne11)*p.nb11 + i2*p.nb12);
    }
    return (const float *) ((const char *) p.x + i1*p.nb11 + i2*p.nb12 + i3*p.nb13);
}

static __device__ __forceinline__ float * mmh_dst_row(const mmh_params & p, const int r) {
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    const int i3 = r / (p.ne1*p.ne2);
    return (float *) ((char *) p.dst + i1*p.nb1 + i2*p.nb2 + i3*p.nb3);
}

// strixllama: the routing weight of row r (ggml_cuda_mul_mat_had_rowmul), 1 without one
static __device__ __forceinline__ float mmh_row_w(const mmh_params & p, const int r) {
    if (p.row_w == nullptr) {
        return 1.0f;
    }
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    return *(const float *) ((const char *) p.row_w + i1*p.row_w_nb1 + i2*p.row_w_nb2);
}

// output value c of the row at out: y, or with the GLU epilogue silu(gate)*up as unary_gated_op_kernel computes it
static __device__ __forceinline__ void mmh_store(const mmh_params & p, float * out, const int c, const float y) {
    if (p.glu == 0) {
        out[c] = y;
        return;
    }
    const float o = ((const float *) ((const char *) p.glu_other + ((const char *) out - (const char *) p.dst)))[c];
    out[c] = p.glu == 1 ? ggml_cuda_op_silu_single(y)*o : ggml_cuda_op_silu_single(o)*y;
}

// index of value c of the staged row at sorted position q in the panel layout of mmh_gemm_et: panels of 16 rows,
// [q/16][k/128][16 rows][128 k], so that a prologue wave writes its 128 values of a row as one 256-byte segment
#ifndef MMH_ET_CW
#define MMH_ET_CW 128
#endif
static __device__ __forceinline__ int64_t mmh_xh_panel(const mmh_params & p, const int q, const int64_t c) {
    return ((int64_t) (q/16)*(p.k/MMH_ET_CW) + c/MMH_ET_CW)*(16*MMH_ET_CW) + (q % 16)*MMH_ET_CW + c % MMH_ET_CW;
}

// strixllama: the 4 rotation values of a lane, elements st*(c0 + 32*j + lane), j = 0..3: one branch on the type, then
// four independent loads (mmh_rot per value branches per value, and clang then waits for each load before the next)
static __device__ __forceinline__ void mmh_rot4(const char * base, const int f16, const int64_t c0, const int lane,
        const int64_t st, float r[4]) {
    if (f16) {
        const half * h = (const half *) base;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r[j] = __half2float(h[(c0 + 32*j + lane)*st]);
        }
    } else {
        const float * f = (const float *) base;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r[j] = f[(c0 + 32*j + lane)*st];
        }
    }
}

// max over i = i0, i0 + step, .. < k of |src[i]*rot[i]| (the branch on the rotation's type outside the loop)
static __device__ __forceinline__ float mmh_amax_rot(const float * src, const char * rot, const int f16, const int64_t i0,
        const int64_t k, const int step) {
    float amax = 0.0f;
    if (f16) {
        const half * h = (const half *) rot;
        for (int64_t i = i0; i < k; i += step) {
            amax = fmaxf(amax, fabsf(src[i]*__half2float(h[i])));
        }
    } else {
        const float * f = (const float *) rot;
        for (int64_t i = i0; i < k; i += step) {
            amax = fmaxf(amax, fabsf(src[i]*f[i]));
        }
    }
    return amax;
}

static __device__ __forceinline__ float mmh_rot(const char * base, const int f16, const int64_t i) {
    return f16 ? __half2float(((const half *) base)[i]) : ((const float *) base)[i];
}

// a wave-uniform value in a scalar register, so that addresses built from it stay scalar
static __device__ __forceinline__ int mmh_uniform(const int v) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_readfirstlane(v);
#else
    return v;
#endif // defined(GGML_USE_HIP)
}

// strixllama: per-expert widths: expert e is not in this launch's class (its blocks have nothing to do)
static __device__ __forceinline__ bool mmh_other_class(const mmh_params & p, const int e) {
    return p.cmap && (p.cmap[e] >> 24) != p.cls;
}

// the weight matrix of expert e (wave-uniform); with classes the matrix at e's index in its class
static __device__ __forceinline__ const char * mmh_weight_matrix(const mmh_params & p, const char * w, const int e) {
    const int le = p.cmap ? (p.cmap[e] & 0xFFFFFF) : e;
    const uint64_t a = (uint64_t) (uintptr_t) w + (uint64_t) le*p.nb02;
#if defined(GGML_USE_HIP)
    return (const char *) (uintptr_t) (((uint64_t) __builtin_amdgcn_readfirstlane((uint32_t) (a >> 32)) << 32) |
        (uint32_t) __builtin_amdgcn_readfirstlane((uint32_t) a));
#else
    return (const char *) (uintptr_t) a;
#endif // defined(GGML_USE_HIP)
}

// the rot_in / rot_out row of expert e (wave-uniform)
static __device__ __forceinline__ const char * mmh_rot_row(const mmh_params & p, const char * rot, const int64_t nb1, const int e) {
    GGML_UNUSED(p);
    return rot + e*nb1;
}

static __device__ __forceinline__ half2 mmh_as_half2(const uint32_t v) {
    half2 h;
    memcpy(&h, &v, sizeof(h));
    return h;
}

static __device__ __forceinline__ uint32_t mmh_as_u32(const half2 h) {
    uint32_t v;
    memcpy(&v, &h, sizeof(v));
    return v;
}

// c + w.x*x0 + w.y*x1 for a half2 w, fp32 FMAs (v_fma_mix_f32)
static __device__ __forceinline__ float mmh_fma2(const uint32_t w, const float x0, const float x1, const float c) {
    const float2 f = __half22float2(mmh_as_half2(w));
    return fmaf(f.y, x1, fmaf(f.x, x0, c));
}

// c + w.x*x.x + w.y*x.y for half2 w and x, fp32 sum (v_dot2_f32_f16)
static __device__ __forceinline__ float mmh_dot2(const uint32_t w, const uint32_t x, const float c) {
#if defined(V_DOT2_F32_F16_AVAILABLE)
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    h2_t a;
    h2_t b;
    memcpy(&a, &w, sizeof(a));
    memcpy(&b, &x, sizeof(b));
    return __builtin_amdgcn_fdot2(a, b, c, false);
#else
    const float2 f = __half22float2(mmh_as_half2(x));
    return mmh_fma2(w, f.x, f.y, c);
#endif // defined(V_DOT2_F32_F16_AVAILABLE)
}

// in-place normalized Hadamard of 128 values, value c = 32*j + lane; blocks of `had` (32, 64 or 128)
// strixllama: value v of lane ^ h, h a power of 2 below 32: DPP (quad_perm, row_xmask) and permlanex16 on RDNA, VALU
// instead of ds_bpermute through the LDS unit (5 per value and transform)
template <int h>
static __device__ __forceinline__ float mmh_lane_xor(const float v) {
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
    if constexpr (h == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xFEDCBA98u, false, false));
    } else {
        constexpr int ctrl = h == 1 ? 0xB1 : h == 2 ? 0x4E : 0x160 + h;
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xF, 0xF, false));
    }
#else
    return __shfl_xor_sync(0xFFFFFFFF, v, h, 32);
#endif
}

template <int h>
static __device__ __forceinline__ void mmh_fwht_stage(float v[4], const int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float o = mmh_lane_xor<h>(v[j]);
        v[j] = (lane & h) ? o - v[j] : v[j] + o;
    }
}

static __device__ __forceinline__ void mmh_fwht(float v[4], const int lane, const int had, const float scale) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] *= scale;
    }
    mmh_fwht_stage<1>(v, lane);
    mmh_fwht_stage<2>(v, lane);
    mmh_fwht_stage<4>(v, lane);
    mmh_fwht_stage<8>(v, lane);
    mmh_fwht_stage<16>(v, lane);
    if (had >= 64) {
        float a = v[0], b = v[1];
        v[0] = a + b;
        v[1] = a - b;
        a = v[2]; b = v[3];
        v[2] = a + b;
        v[3] = a - b;
    }
    if (had >= 128) {
        float a = v[0], b = v[2];
        v[0] = a + b;
        v[2] = a - b;
        a = v[1]; b = v[3];
        v[1] = a + b;
        v[3] = a - b;
    }
}

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
#define MMH_DPP_SUMS
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))

#if defined(MMH_DPP_SUMS)
// v + v of lane ^ m inside each row of 16 lanes (DPP quad_perm for m = 1, 2, row_xmask for 4, 8)
template <int m>
static __device__ __forceinline__ float mmh_add_xor(const float v) {
    constexpr int ctrl = m == 1 ? 0xB1 : m == 2 ? 0x4E : 0x160 + m;
    return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xF, 0xF, false));
}
#endif // defined(MMH_DPP_SUMS)

// the sum over the 16 lanes of each half of the wave (xor butterfly, the same value in every lane)
static __device__ __forceinline__ float mmh_sum16(float v) {
#if defined(MMH_DPP_SUMS)
    v = mmh_add_xor<1>(v);
    v = mmh_add_xor<2>(v);
    v = mmh_add_xor<4>(v);
    return mmh_add_xor<8>(v);
#else
#pragma unroll
    for (int m = 1; m < 16; m *= 2) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, m, 32);
    }
    return v;
#endif // defined(MMH_DPP_SUMS)
}

// the sum over the 32 lanes of the wave, the same value in every lane
static __device__ __forceinline__ float mmh_sum32(float v) {
    v = mmh_sum16(v);
#if defined(MMH_DPP_SUMS)
    return v + __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xFEDCBA98u, false, false));
#else
    return v + __shfl_xor_sync(0xFFFFFFFF, v, 16, 32);
#endif // defined(MMH_DPP_SUMS)
}

// fp16 GEMV input: one 128-value block v (value c = 32*j + lane) rounded to fp16 (saturated to the finite range, NaN
// kept) into the frag layout of halves at xd, and with sums the F32 sums of the rounded values of its 8 slices of 16
// (k = 16*i .. 16*i + 15) at xs[i]
template <bool sums>
static __device__ __forceinline__ void mmh_store_h16(const float v[4], const int lane, half * xd, float * xs) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = 32*j + lane;
        const half h = __float2half_rn(fabsf(v[j]) > 65504.0f ? copysignf(65504.0f, v[j]) : v[j]);
        xd[16*(c/16) + 4*((c % 8)/2) + 2*((c % 16)/8) + c % 2] = h;
        if constexpr (sums) {
            const float s = mmh_sum16(__half2float(h));
            if (lane % 16 == 0) {
                xs[2*j + lane/16] = s;
            }
        }
    }
}

// max over the threads of the block, any block size up to 1024
static __device__ __forceinline__ float mmh_block_max(float v, float * s_red) {
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, h, 32));
    }
    const int wave = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        s_red[wave] = v;
    }
    __syncthreads();
    float m = 0.0f;
    for (int i = 0; i < (int) blockDim.x/32; ++i) {
        m = fmaxf(m, s_red[i]);
    }
    return m;
}

// layouts of H(rot_in * x):
// - frag (GEMV): F32, per row and 16 values of k, for g = 0..3: k = 2g, 2g + 1, 2g + 8, 2g + 9; with f16 the same
//   order in fp16 (mmh_store_h16, not scaled), and with xsum also the F32 sums of the rounded values of each row over
//   the n_ranges k ranges of kper 16-value slices of the GEMV waves (xsum[n_ranges*row + range], from the slice sums in
//   dynamic LDS)
// - plain (GEMM, f16): fp16 values of the row scaled by a power of 2, K values per row; rscale holds the inverse scale
//   (tiled: the same values in the layout of mmh_gemm_t)
// The prologue also clears the k-split tickets of the GEMV that follows it (zero, n_zero).
template <bool frag, bool f16, bool tiled = false>
static __global__ void __launch_bounds__(256) mmh_prologue(const mmh_params p, half * __restrict__ xh, float * __restrict__ rscale,
        int * __restrict__ zero, const int n_zero, float * __restrict__ xsum, const int n_ranges, const int kper) {
    __shared__ float s_red[8];
    extern __shared__ float s_slice[]; // frag f16 with xsum: the sum of each 16 values of k

    const int r    = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;

    for (int i = r*blockDim.x + threadIdx.x; i < n_zero; i += gridDim.x*blockDim.x) {
        zero[i] = 0;
    }

    const int e = mmh_expert(p, r);
    const float * src = mmh_src_row(p, r);
    const char  * rin = mmh_rot_row(p, p.rot_in, p.rin_nb1, e);

    float s = 1.0f;
    if constexpr (!frag) {
        float amax = mmh_amax_rot(src, rin, p.rin_f16, threadIdx.x, p.k, blockDim.x);
        amax = mmh_block_max(amax, s_red);

        // |H v| <= sqrt(had) * max|v|: with the scale, every value stays below 2^14
        const float bound = amax*sqrtf((float) p.had);
        if (bound > 0.0f && bound <= FLT_MAX) {
            int ex;
            frexpf(bound, &ex);
            s = ldexpf(1.0f, max(-100, min(100, 14 - ex)));
        }
    }
    if (threadIdx.x == 0) {
        rscale[r] = 1.0f/s;
    }

    for (int64_t c0 = 128*wave; c0 < p.k; c0 += 128*(blockDim.x/32)) {
        float v[4];
        float rv[4];
        mmh_rot4(rin, p.rin_f16, c0, lane, 1, rv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t c = c0 + 32*j + lane;
            v[j] = src[c]*rv[j]*s;
        }
        mmh_fwht(v, lane, p.had, p.hscale);
        if constexpr (frag && f16) {
            if (xsum) {
                mmh_store_h16<true>(v, lane, xh + p.k*r + c0, s_slice + c0/16);
            } else {
                mmh_store_h16<false>(v, lane, xh + p.k*r + c0, nullptr);
            }
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t c  = c0 + 32*j + lane;
            if constexpr (frag) {
                ((float *) xh)[p.k*r + 16*(c/16) + 4*((c % 8)/2) + 2*((c % 16)/8) + c % 2] = v[j];
            } else if constexpr (tiled) {
                // mmh_gemm_t: [row block of 128][k/16][128][16]
                xh[((int64_t) (r/128)*(p.k/16) + c/16)*2048 + (r % 128)*16 + c % 16] = __float2half_rn(v[j]);
            } else {
                xh[p.pos ? mmh_xh_panel(p, p.pos[r], c) : p.k*r + c] = __float2half_rn(v[j]);
            }
        }
    }

    if constexpr (frag && f16) {
        if (xsum) {
            __syncthreads();
            const int kslices = p.k/16;
            for (int t = wave; t < n_ranges; t += blockDim.x/32) {
                float sum = 0.0f;
                for (int i = t*kper + lane; i < min(kslices, (t + 1)*kper); i += 32) {
                    sum += s_slice[i];
                }
                sum = mmh_sum32(sum);
                if (lane == 0) {
                    xsum[(int64_t) n_ranges*r + t] = sum;
                }
            }
        }
    }
}

// strixllama: the plain (GEMM) prologue with one wave per row and 8 rows per block: the row's maximum is a wave
// reduction, where mmh_prologue (a block per row) waits at two block barriers per row and leaves waves idle for short
// rows (640 values: 5 of 8) and mmh_prologue_tok at two per expert row. The same values, scales and layouts
// (tiled: mmh_gemm_t, p.pos: the panels of mmh_gemm_et).
template <bool tiled>
static __global__ void __launch_bounds__(256) mmh_prologue_wave(const mmh_params p, half * __restrict__ xh,
        float * __restrict__ rscale) {
    const int lane = threadIdx.x % 32;
    const int r    = blockIdx.x*8 + threadIdx.x/32;
    if (r >= p.n_rows) {
        return;
    }
    const int e = mmh_expert(p, r);
    const float * src = mmh_src_row(p, r);
    const char  * rin = mmh_rot_row(p, p.rot_in, p.rin_nb1, e);

    float amax = mmh_amax_rot(src, rin, p.rin_f16, lane, p.k, 32);
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, h, 32));
    }
    // |H v| <= sqrt(had) * max|v|: with the scale, every value stays below 2^14 (as mmh_prologue)
    float sc = 1.0f;
    const float bound = amax*sqrtf((float) p.had);
    if (bound > 0.0f && bound <= FLT_MAX) {
        int ex;
        frexpf(bound, &ex);
        sc = ldexpf(1.0f, max(-100, min(100, 14 - ex)));
    }
    if (lane == 0) {
        rscale[r] = 1.0f/sc;
    }
    const int q = p.pos ? p.pos[r] : r;
    for (int64_t c0 = 0; c0 < p.k; c0 += 128) {
        float v[4];
        float rv[4];
        mmh_rot4(rin, p.rin_f16, c0, lane, 1, rv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t c = c0 + 32*j + lane;
            v[j] = src[c]*rv[j]*sc;
        }
        mmh_fwht(v, lane, p.had, p.hscale);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t c = c0 + 32*j + lane;
            const int64_t o = (p.pos || p.panel_rows) ? mmh_xh_panel(p, q, c) :
                tiled ? ((int64_t) (r/128)*(p.k/16) + c/16)*2048 + (r % 128)*16 + c % 16 : p.k*r + c;
            xh[o] = __float2half_rn(v[j]);
        }
    }
}

// strixllama: the normalized Hadamard of 128 values held four consecutive per lane (value c = 4*lane + j): the
// butterflies of mmh_fwht in the same order of strides (1 and 2 in registers, then 4 .. 64 across lanes) on the same
// pairs, so the same results bit for bit; blocks of `had` (32, 64 or 128)
static __device__ __forceinline__ void mmh_fwht_v4(float v[4], const int lane, const int had, const float scale) {
#pragma clang fp contract(off)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] *= scale;
    }
    float a = v[0], b = v[1];
    v[0] = a + b;
    v[1] = a - b;
    a = v[2]; b = v[3];
    v[2] = a + b;
    v[3] = a - b;
    a = v[0]; b = v[2];
    v[0] = a + b;
    v[2] = a - b;
    a = v[1]; b = v[3];
    v[1] = a + b;
    v[3] = a - b;
    mmh_fwht_stage<1>(v, lane);
    mmh_fwht_stage<2>(v, lane);
    mmh_fwht_stage<4>(v, lane);
    if (had >= 64) {
        mmh_fwht_stage<8>(v, lane);
    }
    if (had >= 128) {
        mmh_fwht_stage<16>(v, lane);
    }
}

// the rotation values c .. c + 3 (c a multiple of 4) in one load
static __device__ __forceinline__ void mmh_rot4c(const char * base, const int f16, const int64_t c, float r[4]) {
    if (f16) {
        const uint2 w = *(const uint2 *) ((const half *) base + c);
        const float2 a = __half22float2(mmh_as_half2(w.x));
        const float2 b = __half22float2(mmh_as_half2(w.y));
        r[0] = a.x; r[1] = a.y; r[2] = b.x; r[3] = b.y;
    } else {
        const float4 f = *(const float4 *) ((const float *) base + c);
        r[0] = f.x; r[1] = f.y; r[2] = f.z; r[3] = f.w;
    }
}

// strixllama: mmh_prologue_wave with four consecutive values per lane: the row and rotation in 16- / 8-byte loads
// and each lane's four fp16 values in one 8-byte store (mmh_prologue_wave: four 2-byte stores per lane and 128-value
// chunk; staging the per-expert rows of the converted checkpoints is store-bound). Same values, scales and layouts
// (mmh_fwht_v4); needs 16-byte aligned rows and 8-byte aligned rotation rows (mmh_pwave4_ok).
template <bool tiled>
static __global__ void __launch_bounds__(256) mmh_prologue_wave4(const mmh_params p, half * __restrict__ xh,
        float * __restrict__ rscale) {
    const int lane = threadIdx.x % 32;
    const int r    = blockIdx.x*8 + threadIdx.x/32;
    if (r >= p.n_rows) {
        return;
    }
    const int e = mmh_expert(p, r);
    const float * src = mmh_src_row(p, r);
    const char  * rin = mmh_rot_row(p, p.rot_in, p.rin_nb1, e);

    float amax = 0.0f;
    for (int64_t c0 = 0; c0 < p.k; c0 += 128) {
        const int64_t c = c0 + 4*lane;
        const float4 x4 = *(const float4 *) (src + c);
        float rv[4];
        mmh_rot4c(rin, p.rin_f16, c, rv);
        amax = fmaxf(amax, fabsf(x4.x*rv[0]));
        amax = fmaxf(amax, fabsf(x4.y*rv[1]));
        amax = fmaxf(amax, fabsf(x4.z*rv[2]));
        amax = fmaxf(amax, fabsf(x4.w*rv[3]));
    }
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, h, 32));
    }
    float sc = 1.0f;
    const float bound = amax*sqrtf((float) p.had);
    if (bound > 0.0f && bound <= FLT_MAX) {
        int ex;
        frexpf(bound, &ex);
        sc = ldexpf(1.0f, max(-100, min(100, 14 - ex)));
    }
    if (lane == 0) {
        rscale[r] = 1.0f/sc;
    }
    const int q = p.pos ? p.pos[r] : r;
    for (int64_t c0 = 0; c0 < p.k; c0 += 128) {
        const int64_t c = c0 + 4*lane;
        const float4 x4 = *(const float4 *) (src + c);
        float rv[4];
        mmh_rot4c(rin, p.rin_f16, c, rv);
        float v[4] = { x4.x*rv[0]*sc, x4.y*rv[1]*sc, x4.z*rv[2]*sc, x4.w*rv[3]*sc };
        mmh_fwht_v4(v, lane, p.had, p.hscale);
        const int64_t o = (p.pos || p.panel_rows) ? mmh_xh_panel(p, q, c) :
            tiled ? ((int64_t) (r/128)*(p.k/16) + c/16)*2048 + (r % 128)*16 + c % 16 : p.k*r + c;
        const half2 h01 = __halves2half2(__float2half_rn(v[0]), __float2half_rn(v[1]));
        const half2 h23 = __halves2half2(__float2half_rn(v[2]), __float2half_rn(v[3]));
        *(uint2 *) (xh + o) = make_uint2(mmh_as_u32(h01), mmh_as_u32(h23));
    }
}

// strixllama: mmh_prologue_wave4 applies: 16-byte aligned input rows and 8-byte (fp16) / 16-byte (f32) aligned
// rotation rows; STRIX_MMH_PWAVE4=0 keeps mmh_prologue_wave (bitwise A/B)
static bool mmh_pwave4_ok(const mmh_params & p) {
    static const bool env = !getenv("STRIX_MMH_PWAVE4") || atoi(getenv("STRIX_MMH_PWAVE4")) != 0;
    const uintptr_t ra = p.rin_f16 ? 8 : 16;
    return env && p.k % 128 == 0 && (uintptr_t) p.x % 16 == 0 && p.nb11 % 16 == 0 && p.nb12 % 16 == 0 &&
        p.nb13 % 16 == 0 && (uintptr_t) p.rot_in % ra == 0 && (uintptr_t) p.rin_nb1 % ra == 0;
}

// strixllama: the plain (GEMM) prologue of an op with ids whose input has one row per token (ne11 == 1: the experts'
// gate and up): a block per token reads the token's row once and writes H(rot_in * x) of each of its ne1 expert rows,
// where mmh_prologue reads the row again for every expert (and twice: for the scale, then for the values). The values,
// scales and layout are the same as mmh_prologue<false, true> writes.
static __global__ void __launch_bounds__(256) mmh_prologue_tok(const mmh_params p, half * __restrict__ xh,
        float * __restrict__ rscale) {
    constexpr int MAXC = 4;               // 128-value chunks per wave: k <= 8*MAXC*128 = 4096
    constexpr int MAXS = 16;              // expert slots per token
    __shared__ float s_max[8][MAXS];
    const int t    = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int nchunks = (int) (p.k/128);
    const int ns = p.ne1;

    const float * src = (const float *) ((const char *) p.x + (int64_t) t*p.nb12);
    float xv[MAXC][4];
#pragma unroll
    for (int ci = 0; ci < MAXC; ++ci) {
        const int c0 = 128*(wave + 8*ci);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xv[ci][j] = wave + 8*ci < nchunks ? src[c0 + 32*j + lane] : 0.0f;
        }
    }

    // every slot's max |x * rot_in| first, one barrier for all of them
    for (int s = 0; s < ns; ++s) {
        const int r = s + ns*t;
        const char * rin = mmh_rot_row(p, p.rot_in, p.rin_nb1, mmh_expert(p, r));
        float amax = 0.0f;
#pragma unroll
        for (int ci = 0; ci < MAXC; ++ci) {
            const int c0 = 128*(wave + 8*ci);
            if (wave + 8*ci < nchunks) {
                float rv[4];
                mmh_rot4(rin, p.rin_f16, c0, lane, 1, rv);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    amax = fmaxf(amax, fabsf(xv[ci][j]*rv[j]));
                }
            }
        }
#pragma unroll
        for (int h = 16; h > 0; h >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, h, 32));
        }
        if (lane == 0) {
            s_max[wave][s] = amax;
        }
    }
    __syncthreads();

    for (int s = 0; s < ns; ++s) {
        const int r = s + ns*t;
        const char * rin = mmh_rot_row(p, p.rot_in, p.rin_nb1, mmh_expert(p, r));
        float amax = 0.0f;
        for (int w = 0; w < 8; ++w) {
            amax = fmaxf(amax, s_max[w][s]);
        }
        float sc = 1.0f;
        const float bound = amax*sqrtf((float) p.had);
        if (bound > 0.0f && bound <= FLT_MAX) {
            int ex;
            frexpf(bound, &ex);
            sc = ldexpf(1.0f, max(-100, min(100, 14 - ex)));
        }
        if (threadIdx.x == 0) {
            rscale[r] = 1.0f/sc;
        }
#pragma unroll
        for (int ci = 0; ci < MAXC; ++ci) {
            const int c0 = 128*(wave + 8*ci);
            if (wave + 8*ci < nchunks) {
                float v[4];
                float rv[4];
                mmh_rot4(rin, p.rin_f16, c0, lane, 1, rv);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[j] = xv[ci][j]*rv[j]*sc;
                }
                mmh_fwht(v, lane, p.had, p.hscale);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int64_t c = c0 + 32*j + lane;
                    xh[p.pos ? mmh_xh_panel(p, p.pos[r], c) : p.k*r + c] = __float2half_rn(v[j]);
                }
            }
        }
    }
}

// rows grouped by weight matrix: rows_sorted, and items (expert, first row, row count <= mt)
// one block, counts in LDS by atomics, so the order inside one expert is arbitrary; each row is computed alone
static __global__ void __launch_bounds__(1024) mmh_route(const mmh_params p, const int mt, int * __restrict__ eid,
        int * __restrict__ rank, int * __restrict__ rows_sorted, int * __restrict__ items, int * __restrict__ n_items) {
    __shared__ int s_count[MMH_MAX_EXPERTS];
    __shared__ int s_offs[MMH_MAX_EXPERTS];
    __shared__ int s_rows[1024];
    __shared__ int s_items[1024];

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    for (int64_t e = tid; e < p.n_exp; e += nth) {
        s_count[e] = 0;
    }
    __syncthreads();

    // eid and rank of row r are written and read by the same thread
    for (int r = tid; r < p.n_rows; r += nth) {
        const int e = mmh_expert(p, r);
        eid[r]  = e;
        rank[r] = atomicAdd(&s_count[e], 1);
    }
    __syncthreads();

    // each thread scans a contiguous range of experts
    const int64_t per = (p.n_exp + nth - 1)/nth;
    const int64_t e0  = std::min<int64_t>(p.n_exp, tid*per);
    const int64_t e1  = std::min<int64_t>(p.n_exp, e0 + per);
    int sum_rows  = 0;
    int sum_items = 0;
    for (int64_t e = e0; e < e1; ++e) {
        const int c = s_count[e];
        sum_rows  += c;
        sum_items += (c + mt - 1)/mt;
    }
    s_rows[tid]  = sum_rows;
    s_items[tid] = sum_items;
    __syncthreads();
    for (int h = 1; h < nth; h *= 2) {
        const int a = tid >= h ? s_rows[tid - h]  : 0;
        const int b = tid >= h ? s_items[tid - h] : 0;
        __syncthreads();
        s_rows[tid]  += a;
        s_items[tid] += b;
        __syncthreads();
    }
    int row  = s_rows[tid]  - sum_rows;
    int item = s_items[tid] - sum_items;
    for (int64_t e = e0; e < e1; ++e) {
        const int c = s_count[e];
        s_offs[e] = row;
        for (int i = 0; i < c; i += mt) {
            items[3*item + 0] = (int) e;
            items[3*item + 1] = row + i;
            items[3*item + 2] = min(mt, c - i);
            ++item;
        }
        row += c;
    }
    if (tid == nth - 1) {
        *n_items = s_items[tid];
    }
    __syncthreads();

    // strixllama: rank[r] becomes the sorted position of row r (the panel layout of mmh_gemm_et)
    for (int r = tid; r < p.n_rows; r += nth) {
        const int q = s_offs[eid[r]] + rank[r];
        rows_sorted[q] = r;
        rank[r] = q;
    }
}

// EXL3: 16x16 tiles of 256 positions, position p of a tile ends at bit ((p + 1)*b2)/2 of a ring of 32-bit words read
// MSB first, its state is the 16-bit window that ends there (b2 = 2*bits). Lane l decodes positions 8l .. 8l + 7:
// column c0 = l/4 and c0 + 8, k = 2*(l%4) + {0, 1, 8, 9}, so its 8 windows sit in 3 consecutive words of the ring.
template <int b2>
struct mmh_exl3 {
    static constexpr int words      = 4*b2;
    static constexpr int tile_bytes = 16*b2;

    int i0, i1, i2; // word index of the 3 words of this lane
    int o;          // bit offset of the first window in word i0

    __device__ __forceinline__ explicit mmh_exl3(const int lane) {
        const int s0 = (((8*lane + 1)*b2) >> 1) - 16 + 128*b2;
        const int wb = s0/32;
        i0 = wb % words;
        i1 = (wb + 1) % words;
        i2 = (wb + 2) % words;
        o  = s0 % 32;
    }

    // end of window j relative to the start of window 0 of the lane
    static constexpr int end(const int j) {
        return (((j + 1)*b2) >> 1) - (b2 >> 1) + 16;
    }

    // whether the 8 windows of every lane end within its first 2 words (then the third word is not read)
    static constexpr bool fits_two_words() {
        for (int lane = 0; lane < 32; ++lane) {
            if (((((8*lane + 1)*b2) >> 1) - 16 + 128*b2) % 32 + end(7) > 64) {
                return false;
            }
        }
        return true;
    }
    static constexpr bool two_words = fits_two_words();
};

// the low 32 bits of (hi:lo) >> s, 0 <= s < 32
static __device__ __forceinline__ uint32_t mmh_funnel(const uint32_t hi, const uint32_t lo, const int s) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_alignbit(hi, lo, s);
#else
    return __funnelshift_r(lo, hi, s);
#endif // defined(GGML_USE_HIP)
}

#if defined(GGML_USE_HIP)
typedef unsigned short mmh_u16x2 __attribute__((ext_vector_type(2)));
#endif // defined(GGML_USE_HIP)

// the trellis hash x*c + a mod 2^32 of the 16-bit state x (the low 16 bits of s, the bits above are ignored).
// HIP: x*(c mod 2^16) + (a mod 2^16) + ((x*(c >> 16) + (a >> 16)) mod 2^16) << 16, where the high half is one packed
// 16-bit multiply (add) whose low lane is zero (or a mod 2^16), and the rest one 24-bit multiply-add: two full-rate
// instructions instead of v_mul_lo_u32 (a quarter of the rate) or three 24-bit ones. Plain C, so that the compiler
// knows their latencies.
template <uint32_t c, uint32_t a>
static __device__ __forceinline__ uint32_t mmh_trellis_hash(const uint32_t s) {
#if defined(GGML_USE_HIP)
    const mmh_u16x2 xv = { (unsigned short) s, (unsigned short) s };
    const mmh_u16x2 cv = { 0, (unsigned short) (c >> 16) };
    const mmh_u16x2 av = { (unsigned short) (a & 0xFFFFu), (unsigned short) (a >> 16) };
    const mmh_u16x2 tv = xv*cv + av;
    uint32_t t;
    memcpy(&t, &tv, sizeof(t));
    return (s & 0xFFFFu)*(c & 0xFFFFu) + t;
#else
    return (s & 0xFFFFu)*c + a;
#endif // defined(GGML_USE_HIP)
}

// mul1 codebook: value = fp16(u*k_inv + k_bias) of u = byte_sum(state*0x83DCD12D) + 1024, an integer in [1024, 2044]
#define MMH_MUL1_KINV 0.00676727294921875f // fp16 0x1EEE
#define MMH_MUL1_BIAS (-10.3828125f)       // fp16 0xC931

// two values u of the mul1 codebook as half2 (exact: 1024 + byte sum, the fp16 bit pattern 0x6400 + byte sum)
static __device__ __forceinline__ uint32_t mmh_mul1_u_pair(const uint32_t s0, const uint32_t s1) {
    const uint32_t p0 = mmh_trellis_hash<0x83DCD12Du, 0>(s0);
    const uint32_t p1 = mmh_trellis_hash<0x83DCD12Du, 0>(s1);
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_sad_hi_u8(p1, 0u, __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u));
#else
    return __dp4a(p0, 0x01010101u, 0x6400u) | (__dp4a(p1, 0x01010101u, 0x6400u) << 16);
#endif
}

// two mul1 codebook values as half2: fp16(u*k_inv + k_bias), one fp16 rounding
static __device__ __forceinline__ uint32_t mmh_mul1_pair(const uint32_t s0, const uint32_t s1) {
    return mmh_as_u32(__hfma2(mmh_as_half2(mmh_mul1_u_pair(s0, s1)), mmh_as_half2(0x1EEE1EEEu), mmh_as_half2(0xC931C931u)));
}

// two mcg or 3inst codebook values as half2: each hash keeps its sign and low bits as two fp16 halves of exponent
// 12..15, which are added in fp16
static __device__ __forceinline__ uint32_t mmh_half_sum_pair(uint32_t p0, uint32_t p1) {
    p0 = (p0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    p1 = (p1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
#if defined(GGML_USE_HIP)
    const uint32_t lo = __builtin_amdgcn_perm(p1, p0, 0x05040100u);
    const uint32_t hi = __builtin_amdgcn_perm(p1, p0, 0x07060302u);
#else
    const uint32_t lo = __byte_perm(p0, p1, 0x5410);
    const uint32_t hi = __byte_perm(p0, p1, 0x7632);
#endif // defined(GGML_USE_HIP)
    return mmh_as_u32(__hadd2(mmh_as_half2(lo), mmh_as_half2(hi)));
}

// EXL3 types: twice the bits (0 for other types) and the codebook (0 = mul1, 1 = mcg, 2 = 3inst)
static constexpr bool mmh_is_exl3(const ggml_type type) {
    return (type >= GGML_TYPE_EXL3_M1 && type <= GGML_TYPE_EXL3_M3H) || (type >= GGML_TYPE_EXL3_G1 && type <= GGML_TYPE_EXL3_G8) ||
           (type >= GGML_TYPE_EXL3_T1 && type <= GGML_TYPE_EXL3_T8);
}

static constexpr int mmh_exl3_b2(const ggml_type type) {
    return !mmh_is_exl3(type) ? 0 : (type - GGML_TYPE_EXL3_M1) % 16 < 8 ? 2*((type - GGML_TYPE_EXL3_M1) % 16 + 1) :
        2*((type - GGML_TYPE_EXL3_M1) % 16 - 7) + 1;
}

static constexpr int mmh_exl3_codebook(const ggml_type type) {
    return mmh_is_exl3(type) ? (type - GGML_TYPE_EXL3_M1)/16 : -1;
}

// one lane's share of the 16x16 tile (kt, nt) of a 128-row group, as half2 over k, k + 1:
// out[0] = (c0, r0), out[1] = (c0, r0 + 8), out[2] = (c0 + 8, r0), out[3] = (c0 + 8, r0 + 8), r0 = 2*(lane%4)
// F32 weights give k in out and k + 1 in lo, as F32. fetch() only loads, so the loads can be issued ahead.
// coal (EXL3): the wave loads each tile once, one word per lane, and decode() takes the 3 words of a lane from their
// lanes (ds_bpermute) instead of 3 loads per lane of the whole tile: fewer memory instructions, more LDS-unit work.
// The words and so the results are the same.
static __device__ __forceinline__ uint32_t mmh_lane_word(const uint32_t v, const int src) {
#if defined(GGML_USE_HIP)
    return (uint32_t) __builtin_amdgcn_ds_bpermute(4*src, (int) v);
#else
    return __shfl_sync(0xFFFFFFFF, v, src, 32);
#endif // defined(GGML_USE_HIP)
}

// EXL3 tiles are read through a buffer resource of the group (scalar base, scalar tile offset, the lane's word offset in
// a vector register), so that a tile load needs no vector address arithmetic
#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_make_buffer_rsrc) &&     __has_builtin(__builtin_amdgcn_raw_buffer_load_b32) && defined(RDNA4)
#define MMH_BUFFER_LOADS
#endif // ...

// strixllama: weight reads through a global address-space pointer. The weight base is rebuilt from readfirstlane'd
// integers (mmh_weight_matrix), so clang read it with flat loads, which count in lgkmcnt as well: every wait for an
// LDS result (ds_bpermute of the coalesced decode, staged rows) then also waited for the weight loads in flight
#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
#define MMH_GLOBAL __attribute__((address_space(1)))
#else
#define MMH_GLOBAL
#endif // defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
template <typename T>
static __device__ __forceinline__ T mmh_ld(const char * p) {
    return *(const MMH_GLOBAL T *) p;
}

template <ggml_type type, bool coal = false>
struct mmh_weight {
    static constexpr int  b2        = mmh_exl3_b2(type);
    static constexpr int  codebook  = mmh_exl3_codebook(type);
    static constexpr bool is_exl3   = b2 > 0;
    static constexpr int  words     = is_exl3 ? 4*b2 : 0; // 32-bit words of a tile
    static constexpr bool coalesced = is_exl3 && coal;
    static constexpr bool two      = is_exl3 && mmh_exl3<is_exl3 ? b2 : 2>::two_words; // the lane reads 2 words of 3
    static constexpr int  n_raw     = coalesced ? (words > 32 ? 2 : 1) : is_exl3 ? (two ? 2 : 3) : type == GGML_TYPE_F16 ? 4 : 8;

    struct raw {
        uint32_t v[n_raw];
    };

    const char * group; // wave-uniform base of the 128-row group
#if defined(MMH_BUFFER_LOADS)
    __amdgpu_buffer_rsrc_t rsrc; // EXL3: the group as a raw buffer (0x31004000: the RDNA4 raw 32-bit buffer config)
#endif // defined(MMH_BUFFER_LOADS)
    int64_t  nb01;
    uint32_t off[3];    // EXL3: byte offsets of the 3 words of this lane in a tile (coal: word indices); F16/F32: offset of (c0, r0)
    int      o;         // EXL3: bit offset of the first window in the first word
    int      lane;

    __device__ __forceinline__ mmh_weight(const char * group, const int64_t nb01, const int lane) : group(group), nb01(nb01), lane(lane) {
#if defined(MMH_BUFFER_LOADS)
        rsrc = __builtin_amdgcn_make_buffer_rsrc((void *) group, 0, 0x7FFFFFFF, 0x31004000);
#endif // defined(MMH_BUFFER_LOADS)
        if constexpr (coalesced) {
            const mmh_exl3<b2> c(lane);
            off[0] = c.i0; off[1] = c.i1; off[2] = c.i2; o = c.o;
        } else if constexpr (is_exl3) {
            const mmh_exl3<b2> c(lane);
            off[0] = 4*c.i0; off[1] = 4*c.i1; off[2] = 4*c.i2; o = c.o;
        } else {
            off[0] = (lane/4)*(uint32_t) nb01 + 2*(lane % 4)*(type == GGML_TYPE_F32 ? 4 : 2);
            off[1] = off[2] = 0;
            o = 0;
        }
    }

    __device__ __forceinline__ raw fetch(const int kt, const int nt) const {
        raw r;
        if constexpr (coalesced) {
#if defined(MMH_BUFFER_LOADS)
            const int so = (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
            r.v[0] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, 4*(words >= 32 ? lane : lane % words), so, 0);
            if constexpr (words > 32) {
                r.v[1] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, 4*(32 + lane % (words - 32)), so, 0);
            }
#else
            const char * t = group + (int64_t) (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
            r.v[0] = mmh_ld<uint32_t>(t + 4*(words >= 32 ? lane : lane % words));
            if constexpr (words > 32) {
                r.v[1] = mmh_ld<uint32_t>(t + 4*(32 + lane % (words - 32)));
            }
#endif // defined(MMH_BUFFER_LOADS)
        } else if constexpr (is_exl3) {
#if defined(MMH_BUFFER_LOADS)
            const int so = (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
#pragma unroll
            for (int i = 0; i < n_raw; ++i) {
                r.v[i] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, off[i], so, 0);
            }
#else
            const char * t = group + (int64_t) (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
#pragma unroll
            for (int i = 0; i < n_raw; ++i) {
                r.v[i] = mmh_ld<uint32_t>(t + off[i]);
            }
#endif // defined(MMH_BUFFER_LOADS)
        } else if constexpr (type == GGML_TYPE_F16) {
            const char * a = group + 16*nt*nb01 + 32*kt + off[0];
            const char * b = a + 8*nb01;
            r.v[0] = mmh_ld<uint32_t>(a);
            r.v[1] = mmh_ld<uint32_t>(a + 16);
            r.v[2] = mmh_ld<uint32_t>(b);
            r.v[3] = mmh_ld<uint32_t>(b + 16);
        } else {
            static_assert(type == GGML_TYPE_F32, "unsupported weight type");
            const char * a = group + 16*nt*nb01 + 64*kt + off[0];
            const char * b = a + 8*nb01;
            const float2 f[4] = { make_float2(mmh_ld<float>(a), mmh_ld<float>(a + 4)), make_float2(mmh_ld<float>(a + 32), mmh_ld<float>(a + 36)),
                                  make_float2(mmh_ld<float>(b), mmh_ld<float>(b + 4)), make_float2(mmh_ld<float>(b + 32), mmh_ld<float>(b + 36)) };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                r.v[2*i + 0] = __float_as_uint(f[i].x);
                r.v[2*i + 1] = __float_as_uint(f[i].y);
            }
        }
        return r;
    }

    // coal: word i of the tile
    __device__ __forceinline__ uint32_t word(const raw & r, const int i) const {
        if constexpr (words > 32) {
            const uint32_t a = mmh_lane_word(r.v[0], i & 31);
            const uint32_t b = mmh_lane_word(r.v[1 % n_raw], (i - 32) & 31);
            return i < 32 ? a : b;
        } else {
            return mmh_lane_word(r.v[0], i);
        }
    }

    // u: mul1 gives the values u of the codebook (the affine map is applied by the caller), other types their values
    template <bool u = false>
    __device__ __forceinline__ void decode(const raw & r, uint32_t out[4], uint32_t lo[4]) const {
        if constexpr (is_exl3) {
            uint32_t w0, w1, w2;
            if constexpr (coalesced) {
                w0 = word(r, off[0]);
                w1 = word(r, off[1]);
                w2 = two ? 0 : word(r, off[2]);
            } else {
                w0 = r.v[0];
                w1 = r.v[1 % n_raw];
                w2 = two ? 0 : r.v[2 % n_raw];
            }
            // the 96 bits of the 3 words from the first window of the lane on; windows only need their low 16 bits right.
            // o is never 0 for the EXL3 bit widths, so 32 - o is a valid alignbit shift (no 64-bit shifts)
            const uint32_t v0 = mmh_funnel(w0, w1, 32 - o);
            const uint32_t v1 = mmh_funnel(w1, w2, 32 - o);
            const uint32_t v2 = w2 << o;
            uint32_t st[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int e = mmh_exl3<b2>::end(j);
                st[j] = e <= 32 ? v0 >> (32 - e) : e < 64 ? mmh_funnel(v0, v1, 64 - e) : e == 64 ? v1 :
                    e < 96 ? mmh_funnel(v1, v2, 96 - e) : v2;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (codebook == 0) {
                    out[i] = u ? mmh_mul1_u_pair(st[2*i], st[2*i + 1]) : mmh_mul1_pair(st[2*i], st[2*i + 1]);
                } else if constexpr (codebook == 1) {
                    // mcg: x*0xCBAC1FED
                    out[i] = mmh_half_sum_pair(mmh_trellis_hash<0xCBAC1FEDu, 0>(st[2*i]), mmh_trellis_hash<0xCBAC1FEDu, 0>(st[2*i + 1]));
                } else {
                    // 3inst: x*89226354 + 64248484
                    out[i] = mmh_half_sum_pair(mmh_trellis_hash<89226354u, 64248484u>(st[2*i]),
                        mmh_trellis_hash<89226354u, 64248484u>(st[2*i + 1]));
                }
            }
            GGML_UNUSED(lo);
        } else if constexpr (type == GGML_TYPE_F16) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = r.v[i];
            }
            GGML_UNUSED(lo);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = r.v[2*i + 0];
                lo[i]  = r.v[2*i + 1];
            }
        }
    }
};

// one pass of the GEMV over the k range [kb, ke) of a wave for the rows rowp[0 .. nc - 1] (nc <= rows): stages the input
// (fused), decodes each tile once for all rows and leaves the row sums of the wave's 16 columns in s_part[ksp].
// h16: fp16 input (8 bytes per lane and 16 k) and v_dot2 products; mul1 then multiplies the codebook values u and
// applies the affine map once to the sums: k_inv*sum(u*x) + k_bias*sum(x), sum(x) over [kb, ke) from the staged slice
// sums (fused) or the range sums of the prologue (xsum: this wave's range of row 0, rows n_ranges apart).
template <ggml_type type, int rows, bool fused, bool coal, int srows, bool h16>
static __device__ __forceinline__ void mmh_gemv_pass(const mmh_params & p, const mmh_weight<type, coal> & wl,
        const void * __restrict__ xh, const float * __restrict__ xsum, const int n_ranges, float4 * s_xf,
        float (*s_part)[srows][MMH_NG], const int * rowp, const int nc, const int e, const int kb, const int ke,
        const int kslices, const int koff, const int nks, const int lane, const int wave, const int nt, const int ksp) {
    typedef mmh_weight<type, coal> wtype;
    typedef typename std::conditional<h16, uint2, float4>::type xtype;
    constexpr bool defer = h16 && wtype::codebook == 0;
    constexpr int pf = rows <= 2 ? 4 : 2;
    // the first weight loads go out before the input (the fused staging does not wait for them). Tile indices past
    // the range are clamped to its last tile, so that the k loop below loads without conditions (its ring of pf
    // loads stays in fixed registers and each step waits only for its own load)
    typename wtype::raw wr[pf];
    if (kb < ke) {
#pragma unroll
        for (int d = 0; d < pf; ++d) {
            wr[d] = wl.fetch(min(kb + d, ke - 1), nt);
        }
    }

    // input row bases are wave-uniform, the lane adds its 16-byte (h16: 8-byte) group of each 16 k
    const xtype * xr[rows];
    float sx[rows]; // defer, prologue: the range sum of row i
    if constexpr (fused) {
        // as mmh_prologue<true, h16>: one 128-value block of one row per wave and pass; the waves of the block
        // read the previous rows only before the barrier after their k loop. The slice sums follow the rows.
        float * s_xs = (float *) ((char *) s_xf + (size_t) srows*nks*4*sizeof(xtype));
        const char * rin = mmh_rot_row(p, p.rot_in, p.rin_nb1, e);
        const int nch = nks/8;
        for (int t = wave; t < nc*nch; t += blockDim.x/32) {
            const int i  = t/nch;
            const int ch = t % nch;
            const float * src = mmh_src_row(p, rowp[i]) + 16*koff + 128*ch;
            const char  * rc  = rin + (16*koff + 128*ch)*(p.rin_f16 ? 2 : 4);
            float v[4];
            float rv[4];
            mmh_rot4(rc, p.rin_f16, 0, lane, 1, rv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                v[j] = src[c]*rv[j];
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            if constexpr (h16) {
                mmh_store_h16<defer>(v, lane, (half *) s_xf + (16*nks*i + 128*ch), s_xs + (nks*i + 8*ch));
            } else {
                float * xd = (float *) s_xf + (16*nks*i + 128*ch);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = 32*j + lane;
                    xd[16*(c/16) + 4*((c % 8)/2) + 2*((c % 16)/8) + c % 2] = v[j];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            xr[i] = (const xtype *) s_xf + 4*nks*(i < nc ? i : 0) + lane % 4;
        }
    } else {
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int64_t r = mmh_uniform(rowp[i < nc ? i : 0]);
            xr[i] = (const xtype *) xh + r*(4*kslices) + lane % 4;
            if constexpr (defer) {
                sx[i] = xsum[r*n_ranges];
            }
        }
    }

    float acc[rows][2];
#pragma unroll
    for (int i = 0; i < rows; ++i) {
        acc[i][0] = 0.0f;
        acc[i][1] = 0.0f;
    }

    // the input of the next xpf steps (slot d % xpf holds step kt + d): staged in LDS (fused) for several rows only one
    // step ahead, from global memory (prologue) two steps ahead; fewer registers than the ring of pf weight loads
    constexpr int xpf = fused ? (rows > 1 ? 1 : pf) : 2;
    xtype xv[xpf][rows];
    if (kb < ke) {
#pragma unroll
        for (int d = 0; d < xpf; ++d) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (rows == 1 || i < nc) {
                    xv[d][i] = xr[i][4*(min(kb + d, ke - 1) - koff)];
                }
            }
        }
    }

    // one k step: decode the tile and add its products to the rows
    auto step = [&](const typename wtype::raw & cw, const xtype (&cx)[rows]) {
        uint32_t wf[4];
        uint32_t wlo[4];
        wl.template decode<defer>(cw, wf, wlo);
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            if (rows == 1 || i < nc) {
                const xtype x = cx[i];
                float a0 = acc[i][0];
                float a1 = acc[i][1];
                if constexpr (h16) {
                    a0 = mmh_dot2(wf[0], x.x, a0);
                    a0 = mmh_dot2(wf[1], x.y, a0);
                    a1 = mmh_dot2(wf[2], x.x, a1);
                    a1 = mmh_dot2(wf[3], x.y, a1);
                } else if constexpr (type == GGML_TYPE_F32) {
                    a0 = fmaf(__uint_as_float(wf[0]),  x.x, a0);
                    a0 = fmaf(__uint_as_float(wlo[0]), x.y, a0);
                    a0 = fmaf(__uint_as_float(wf[1]),  x.z, a0);
                    a0 = fmaf(__uint_as_float(wlo[1]), x.w, a0);
                    a1 = fmaf(__uint_as_float(wf[2]),  x.x, a1);
                    a1 = fmaf(__uint_as_float(wlo[2]), x.y, a1);
                    a1 = fmaf(__uint_as_float(wf[3]),  x.z, a1);
                    a1 = fmaf(__uint_as_float(wlo[3]), x.w, a1);
                } else {
                    a0 = mmh_fma2(wf[0], x.x, x.y, a0);
                    a0 = mmh_fma2(wf[1], x.z, x.w, a0);
                    a1 = mmh_fma2(wf[2], x.x, x.y, a1);
                    a1 = mmh_fma2(wf[3], x.z, x.w, a1);
                }
                acc[i][0] = a0;
                acc[i][1] = a1;
            }
        }
    };

    // whole groups of pf steps: slot d holds step kt + d and is refilled with step kt + d + pf (clamped)
    int kt = kb;
    for (; kt + pf <= ke; kt += pf) {
#pragma unroll
        for (int d = 0; d < pf; ++d) {
            const typename wtype::raw cw = wr[d];
            xtype cx[rows];
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                cx[i] = xv[d % xpf][i];
            }
            const int kn = min(kt + d + pf, ke - 1);
            wr[d] = wl.fetch(kn, nt);
            const int kx = min(kt + d + xpf, ke - 1);
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (rows == 1 || i < nc) {
                    xv[d % xpf][i] = xr[i][4*(kx - koff)];
                }
            }
            step(cw, cx);
        }
    }
    // the last ke - kt < pf steps, already loaded in slots 0 ..
#pragma unroll
    for (int d = 0; d < pf - 1; ++d) {
        if (kt + d < ke) {
            xtype cx[rows];
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (rows == 1 || i < nc) {
                    cx[i] = d < xpf ? xv[d % xpf][i] : xr[i][4*(kt + d - koff)];
                }
            }
            step(wr[d], cx);
        }
    }

#pragma unroll
    for (int i = 0; i < rows; ++i) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[i][c] += __shfl_xor_sync(0xFFFFFFFF, acc[i][c], 1, 32);
            acc[i][c] += __shfl_xor_sync(0xFFFFFFFF, acc[i][c], 2, 32);
        }
        if constexpr (defer) {
            if (rows == 1 || i < nc) {
                if constexpr (fused) {
                    // the staged slice sums of row i (after the staged rows)
                    const float * xs = (const float *) ((const char *) s_xf + (size_t) srows*nks*4*sizeof(xtype)) + nks*i - koff;
                    float sum = 0.0f;
                    for (int t = kb + lane; t < ke; t += 32) {
                        sum += xs[t];
                    }
                    sx[i] = mmh_sum32(sum);
                }
                const float b = MMH_MUL1_BIAS*sx[i];
                acc[i][0] = fmaf(MMH_MUL1_KINV, acc[i][0], b);
                acc[i][1] = fmaf(MMH_MUL1_KINV, acc[i][1], b);
            }
        }
        if (i < nc && lane % 4 == 0) {
            s_part[ksp][i][16*nt + lane/4]     = acc[i][0];
            s_part[ksp][i][16*nt + lane/4 + 8] = acc[i][1];
        }
    }
}

// GEMV: a block of 8*ksplit waves owns one 128-row group g (blockIdx.x) of one expert, for the k range blockIdx.z of
// gridDim.z; wave = column tile nt + 8*k split. Its rows come from the routing pass (items) or are collected here: with
// ids, block y is row y and runs only if y is the first row of its expert, then for every row of that expert. The
// weight and input loads of the next pf k steps are in flight while a step is decoded. With a k split over blocks,
// each block stores its partial sums and the last block of the group adds them in k order, then runs the epilogue;
// it also resets its ticket, so that tickets that start at zero stay zero between ops.
// fused: no prologue, the block writes H(rot_in * x) of its rows for the 128-value blocks of its k range to LDS (the
// same values as the prologue, row scale 1), and its tickets must be zero when it starts. coal: see mmh_weight.
// h16: fp16 input, see mmh_gemv_pass (xsum: the range sums of the prologue, mul1).
template <ggml_type type, int rows, bool fused, bool coal, bool h16 = false>
static __global__ void __launch_bounds__(1024) mmh_gemv(const mmh_params p, const char * __restrict__ w,
        const void * __restrict__ xh, const float * __restrict__ xsum, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted, const int ksplit,
        float * __restrict__ part, int * __restrict__ tickets, const int max_chunks) {
    typedef mmh_weight<type, coal> wtype;
    extern __shared__ float4 s_xf[]; // fused: rows x staged k slices x 4 float4 (h16: 4 uint2, then the slice sums)

    __shared__ int   s_rows[MMH_GEMV_LIST];
    __shared__ int   s_eid[MMH_GEMV_LIST];
    __shared__ int   s_wcount[32];
    __shared__ int   s_last;
    __shared__ float s_part[4][rows][MMH_NG];
    const unsigned long long trace_t0 = p.trace ? wall_clock64() : 0;
    auto trace_end = [&](const int dup) {
        if (p.trace && threadIdx.x == 0) {
            const size_t bl = blockIdx.x + gridDim.x*(blockIdx.y + (size_t) gridDim.y*blockIdx.z);
            p.trace[3*bl + 0] = trace_t0;
            p.trace[3*bl + 1] = wall_clock64();
            p.trace[3*bl + 2] = (unsigned long long) __smid() | ((unsigned long long) dup << 16);
        }
    };

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int nt   = wave % 8;
    const int ksp  = wave / 8;
    const int g    = blockIdx.x;

    int e;
    int nr;
    const int * rowp;
    if (items) {
        if ((int) blockIdx.y >= *n_items) {
            return;
        }
        e    = items[3*blockIdx.y + 0];
        rowp = rows_sorted + items[3*blockIdx.y + 1];
        nr   = items[3*blockIdx.y + 2];
        if (mmh_other_class(p, e)) {
            return;
        }
    } else {
        // n_rows <= MMH_GEMV_LIST here: the expert of every row, once
        for (int r = tid; r < p.n_rows; r += blockDim.x) {
            s_eid[r] = mmh_expert(p, r);
        }
        __syncthreads();
        const int y = blockIdx.y;
        e = p.ids ? s_eid[y] : y;
        if (p.ids) {
            bool dup = false;
            for (int r = tid; r < y; r += blockDim.x) {
                dup = dup || s_eid[r] == e;
            }
            if (__syncthreads_or(dup)) {
                trace_end(1);
                return;
            }
        }
        if (mmh_other_class(p, e)) {                                  // block-uniform: e is the block's expert
            return;
        }
        // ordered compaction of the rows of expert e
        int n = 0;
        for (int r0 = p.ids ? y : 0; r0 < p.n_rows; r0 += blockDim.x) {
            const int r = r0 + tid;
            const bool hit = r < p.n_rows && s_eid[r] == e;
#if defined(GGML_USE_HIP)
            const uint64_t mask = __ballot(hit);
            const int before = __popcll(mask & ((1ull << lane) - 1));
            const int count  = __popcll(mask);
#else
            const uint32_t mask = __ballot_sync(0xFFFFFFFF, hit);
            const int before = __popc(mask & ((1u << lane) - 1));
            const int count  = __popc(mask);
#endif
            if (lane == 0) {
                s_wcount[wave] = count;
            }
            __syncthreads();
            int offs = n;
            for (int i = 0; i < wave; ++i) {
                offs += s_wcount[i];
            }
            if (hit && offs + before < MMH_GEMV_LIST) {
                s_rows[offs + before] = r;
            }
            for (int i = 0; i < (int) blockDim.x/32; ++i) {
                n += s_wcount[i];
            }
            __syncthreads();
        }
        nr   = min(n, MMH_GEMV_LIST);
        rowp = s_rows;
    }
    e = mmh_uniform(e);

    const int kz      = blockIdx.z;
    const int kblocks = gridDim.z;
    const int kslices = p.k/16;
    const int per     = (kslices + kblocks*ksplit - 1)/(kblocks*ksplit);
    const int kb      = (kz*ksplit + ksp)*per;
    const int ke      = min(kslices, kb + per);

    // fused: the k slices of the 128-value blocks that hold the k range of the block, from koff
    const int kb_blk = min(kslices, kz*ksplit*per);
    const int koff   = fused ? kb_blk/8*8 : 0;
    const int nks    = fused ? (min(kslices, kb_blk + ksplit*per) + 7)/8*8 - koff : 0;

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    // h16 with the prologue: the range sums of this wave (range kz*ksplit + ksp of kblocks*ksplit)
    int           n_ranges = 0;
    const float * xsr      = nullptr;
    if constexpr (h16 && !fused && wtype::codebook == 0) {
        n_ranges = kblocks*ksplit;
        xsr      = xsum + (kz*ksplit + ksp);
    }
    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;
    // strixllama: the epilogue's rot_out values, loaded before the k loop (not one memory latency after it)
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);

    for (int i0 = 0; i0 < nr; i0 += rows) {
        const int nc = min(rows, nr - i0);

        // a pass decodes each tile once for all its rows; a single row (an expert of one token) takes the one-row pass,
        // whose registers allow the deeper prefetch
        if (rows > 1 && nc == 1) {
            mmh_gemv_pass<type, 1, fused, coal, rows, h16>(p, wl, xh, xsr, n_ranges, s_xf, s_part, rowp + i0, nc, e, kb, ke,
                kslices, koff, nks, lane, wave, nt, ksp);
        } else {
            mmh_gemv_pass<type, rows, fused, coal, rows, h16>(p, wl, xh, xsr, n_ranges, s_xf, s_part, rowp + i0, nc, e, kb, ke,
                kslices, koff, nks, lane, wave, nt, ksp);
        }
        __syncthreads();

        if (kblocks > 1) {
            for (int i = wave; i < nc; i += blockDim.x/32) {
                float * dp = part + (((int64_t) rowp[i0 + i]*gridDim.x + g)*kblocks + kz)*MMH_NG;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.0f;
                    for (int s = 0; s < ksplit; ++s) {
                        sum += s_part[s][i][32*j + lane];
                    }
                    dp[32*j + lane] = sum;
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                const int64_t t = ((int64_t) blockIdx.y*gridDim.x + g)*max_chunks + i0/rows;
                const bool last = atomicAdd(&tickets[t], 1) == kblocks - 1;
                if (last) {
                    tickets[t] = 0;
                }
                s_last = last;
            }
            __syncthreads();
            if (!s_last) {
                continue;
            }
            __threadfence();
        }

        for (int i = wave; i < nc; i += blockDim.x/32) {
            const int r = rowp[i0 + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sum = 0.0f;
                if (kblocks > 1) {
                    const float * dp = part + ((int64_t) r*gridDim.x + g)*kblocks*MMH_NG + 32*j + lane;
                    for (int z = 0; z < kblocks; ++z) {
                        sum += dp[z*MMH_NG];
                    }
                } else {
                    for (int s = 0; s < ksplit; ++s) {
                        sum += s_part[s][i][32*j + lane];
                    }
                }
                v[j] = sum;
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rw = mmh_row_w(p, r);
            const float rs = fused ? 1.0f : rscale[r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                mmh_store(p, out, c, v[j]*rs*ro[j]*rw);
            }
        }
        __syncthreads();
    }
    trace_end(0);
}

// WMMA GEMM (RDNA4), fp16 input: 8 waves, wave w owns column tile w of the 128-row group; mt rows per block. Per step
// of 32 k: the block stages the fp16 rows in LDS (two buffers, one barrier per step), each wave decodes its 2 tiles
// into a wave-private LDS area (the decode lane order is not the WMMA lane order), then one WMMA per 16 k and m tile,
// accumulating in the WMMA accumulator (its rounding is far below the fp16 input rounding). The loads of the next step
// are issued before the decode. The accumulators go to the epilogue through LDS in passes of up to 32 rows. A k split
// over blocks (gridDim.z) works as in the GEMV.
template <ggml_type type, int mt>
static __global__ void __launch_bounds__(256, 2) mmh_gemm(const mmh_params p, const char * __restrict__ w,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted, float * __restrict__ part,
        int * __restrict__ tickets) {
#if defined(AMD_WMMA_AVAILABLE) && (defined(RDNA4) || defined(RDNA3))
    // strixllama: RDNA3 (gfx11) WMMA takes 16 k values of a row (A) or column (B) per lane, the same in both half-waves,
    // and returns row 2*l + lane/16 in accumulator l; RDNA4 takes 8 per lane (k = 8*(lane/16) + 0..7) and returns row
    // 8*(lane/16) + l. The products and sums are the same.
#if defined(RDNA4)
    typedef _Float16 h8_t __attribute__((ext_vector_type(8)));
#else
    typedef _Float16 h8_t __attribute__((ext_vector_type(16)));
#endif // defined(RDNA4)
    typedef float    f8_t __attribute__((ext_vector_type(8)));
    typedef mmh_weight<type, MMH_WIDE_COAL> wtype;   // strixllama: coalesced tile loads (see mmh_gemm_wide)

    constexpr int ks  = 32;
    constexpr int xst = ks + 8;      // halves per staged row, with padding
    constexpr int wst = ks + 8;      // halves per decoded weight row
    constexpr int yst = MMH_NG + 4;  // floats per output row
    constexpr int nm  = mt/16;
    constexpr int ec  = mt > 32 ? 32 : mt; // output rows per pass through LDS
    constexpr int main_bytes = (2*mt*xst + 8*16*wst)*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    __shared__ int s_last;

    if ((int) blockIdx.y >= *n_items) {
        return;
    }
    const int e     = mmh_uniform(items[3*blockIdx.y + 0]);
    const int start = items[3*blockIdx.y + 1];
    const int cnt   = items[3*blockIdx.y + 2];
    if (mmh_other_class(p, e)) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int g    = blockIdx.x;

    half  * xs = (half *) smem;
    half  * ws = xs + 2*mt*xst + wave*16*wst;
    float * ys = (float *) smem;

    // the 16-byte chunks of the staged rows this thread loads: mt rows x 4
    constexpr int cpr    = 4;
    constexpr int nchunk = cpr*mt;
    constexpr int cpt    = (nchunk + 255)/256;
    const half * xsrc[cpt];
    int  xdst[cpt];
    bool xval[cpt];
#pragma unroll
    for (int t = 0; t < cpt; ++t) {
        const int c    = tid + 256*t;
        const int i    = c/cpr;
        const int part = c % cpr;
        xval[t] = c < nchunk && i < cnt;
        const int64_t r = xval[t] ? rows_sorted[start + i] : 0;
        xsrc[t] = xh + r*p.k + 8*part;
        xdst[t] = i*xst + 8*part;
    }

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const int c0 = lane/4;
    const int r0 = 2*(lane % 4);

    f8_t acc[nm];
#pragma unroll
    for (int m = 0; m < nm; ++m) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[m][l] = 0.0f;
        }
    }

    auto fetch_x = [&](const int k0, uint4 * xv) {
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            xv[t] = make_uint4(0, 0, 0, 0);
            if (xval[t]) {
                xv[t] = *(const uint4 *) (xsrc[t] + k0);
            }
        }
    };

    const int kz      = blockIdx.z;
    const int kblocks = gridDim.z;
    const int per     = (p.k/ks + kblocks - 1)/kblocks;
    const int s0      = kz*per;
    const int nsteps  = min(p.k/ks, s0 + per);

    uint4 xv[cpt];
    typename wtype::raw wr[2];
    if (s0 < nsteps) {
        fetch_x(ks*s0, xv);
        wr[0] = wl.fetch(2*s0 + 0, wave);
        wr[1] = wl.fetch(2*s0 + 1, wave);
    }

    for (int step = s0; step < nsteps; ++step) {
        half * xb = xs + (step % 2)*mt*xst;
        uint4 cxv[cpt];
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            cxv[t] = xv[t];
        }
        const typename wtype::raw cw0 = wr[0];
        const typename wtype::raw cw1 = wr[1];
        if (step + 1 < nsteps) {
            fetch_x(ks*(step + 1), xv);
            wr[0] = wl.fetch(2*(step + 1) + 0, wave);
            wr[1] = wl.fetch(2*(step + 1) + 1, wave);
        }
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            uint32_t wf[4];
            uint32_t wlo[4];
            wl.decode(kt == 0 ? cw0 : cw1, wf, wlo);
            *(uint32_t *) (ws + c0*wst       + 16*kt + r0)     = wf[0];
            *(uint32_t *) (ws + c0*wst       + 16*kt + r0 + 8) = wf[1];
            *(uint32_t *) (ws + (c0 + 8)*wst + 16*kt + r0)     = wf[2];
            *(uint32_t *) (ws + (c0 + 8)*wst + 16*kt + r0 + 8) = wf[3];
        }
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            if (tid + 256*t < nchunk) {
                *(uint4 *) (xb + xdst[t]) = cxv[t];
            }
        }
        mmh_lds_sync();
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#if defined(RDNA4)
            const h8_t b = *(const h8_t *) (ws + (lane % 16)*wst + 16*kk + 8*(lane/16));
#else
            const h8_t b = *(const h8_t *) (ws + (lane % 16)*wst + 16*kk);
#endif // defined(RDNA4)
#pragma unroll
            for (int m = 0; m < nm; ++m) {
#if defined(RDNA4)
                const half * xrow = xb + (16*m + lane % 16)*xst + 16*kk + 8*(lane/16);
                const h8_t a = *(const h8_t *) xrow;
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[m]);
#else
                const half * xrow = xb + (16*m + lane % 16)*xst + 16*kk;
                const h8_t a = *(const h8_t *) xrow;
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[m]);
#endif // defined(RDNA4)
            }
        }
    }

    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
    // output Hadamard, rot_out and the row scale of row i of the item
    auto finish = [&](const int i, float v[4]) {
        const int r = rows_sorted[start + i];
        mmh_fwht(v, lane, p.had, p.hscale);
        float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
        const float rw = mmh_row_w(p, r);
        const float rs = rscale[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = 32*j + lane;
            mmh_store(p, out, c, v[j]*rs*ro[j]*rw);
        }
    };

    // the accumulators go through LDS ec rows at a time: to the epilogue, or with a k split to the partial sums
#pragma unroll
    for (int e0 = 0; e0 < mt; e0 += ec) {
        __syncthreads();
#pragma unroll
        for (int m = e0/16; m < (e0 + ec)/16; ++m) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
#if defined(RDNA4)
                ys[(16*m - e0 + 8*(lane/16) + l)*yst + 16*wave + lane % 16] = acc[m][l];
#else
                ys[(16*m - e0 + 2*l + lane/16)*yst + 16*wave + lane % 16] = acc[m][l];
#endif // defined(RDNA4)
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < min(cnt, e0 + ec); i += 8) {
            if (kblocks > 1) {
                float * dp = part + (((int64_t) rows_sorted[start + i]*gridDim.x + g)*kblocks + kz)*MMH_NG;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dp[32*j + lane] = ys[(i - e0)*yst + 32*j + lane];
                }
            } else {
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[j] = ys[(i - e0)*yst + 32*j + lane];
                }
                finish(i, v);
            }
        }
    }

    if (kblocks > 1) {
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            s_last = atomicAdd(&tickets[(int64_t) blockIdx.y*gridDim.x + g], 1) == kblocks - 1;
        }
        __syncthreads();
        if (!s_last) {
            return;
        }
        __threadfence();
        for (int i = wave; i < cnt; i += 8) {
            const int r = rows_sorted[start + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float * dp = part + ((int64_t) r*gridDim.x + g)*kblocks*MMH_NG + 32*j + lane;
                float sum = 0.0f;
                for (int z = 0; z < kblocks; ++z) {
                    sum += dp[z*MMH_NG];
                }
                v[j] = sum;
            }
            finish(i, v);
        }
    }
#else
    GGML_UNUSED_VARS(p, w, xh, rscale, items, n_items, rows_sorted, part, tickets);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && (defined(RDNA4) || defined(RDNA3))
}

// strixllama: the WMMA GEMM over G 128-row groups of the weight at once (blockIdx.x = a chunk of G groups). The GEMM
// above stages its rows once per group; with ids those rows are the per-expert H(rot_in * x) of the prologue, so an op
// of n columns read them n/128 times (5 for the experts' gate and up, 20 for down) - the most of its memory traffic.
// Here wave w owns n-tile w of every group of the chunk, the staged rows serve all of them, and there is no k split.
template <ggml_type type, int mt, int G>
static __global__ void __launch_bounds__(256, 1) mmh_gemm_wide(const mmh_params p, const char * __restrict__ w,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted, const int n_groups) {
#if defined(AMD_WMMA_AVAILABLE) && (defined(RDNA4) || defined(RDNA3))
#if defined(RDNA4)
    typedef _Float16 h8_t __attribute__((ext_vector_type(8)));
#else
    typedef _Float16 h8_t __attribute__((ext_vector_type(16)));
#endif // defined(RDNA4)
    typedef float    f8_t __attribute__((ext_vector_type(8)));
    // coalesced tile loads: one 32-bit word per lane, shared by ds_bpermute (3x fewer memory instructions)
    typedef mmh_weight<type, MMH_WIDE_COAL> wtype;

    constexpr int ks  = 32;
    constexpr int xst = ks + 8;      // halves per staged row, with padding
    constexpr int wst = ks + 8;      // halves per decoded weight row
    constexpr int yst = MMH_NG + 4;  // floats per output row
    constexpr int nm  = mt/16;
    constexpr int ec  = mt > 32 ? 32 : mt; // output rows per pass through LDS
    constexpr int main_bytes = (2*mt*xst + G*8*16*wst)*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];

    if ((int) blockIdx.y >= *n_items) {
        return;
    }
    const int e     = mmh_uniform(items[3*blockIdx.y + 0]);
    const int start = items[3*blockIdx.y + 1];
    const int cnt   = items[3*blockIdx.y + 2];
    if (mmh_other_class(p, e)) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int gc0  = blockIdx.x*G;
    const int ng   = min(G, n_groups - gc0);

    half  * xs  = (half *) smem;
    half  * wsb = xs + 2*mt*xst;     // [G][wave][16][wst]
    float * ys  = (float *) smem;

    constexpr int cpr    = 4;
    constexpr int nchunk = cpr*mt;
    constexpr int cpt    = (nchunk + 255)/256;
    const half * xsrc[cpt];
    int  xdst[cpt];
    bool xval[cpt];
#pragma unroll
    for (int t = 0; t < cpt; ++t) {
        const int c    = tid + 256*t;
        const int i    = c/cpr;
        const int part = c % cpr;
        xval[t] = c < nchunk && i < cnt;
        const int64_t r = xval[t] ? rows_sorted[start + i] : 0;
        xsrc[t] = xh + r*p.k + 8*part;
        xdst[t] = i*xst + 8*part;
    }

    const char * wm = mmh_weight_matrix(p, w, e);
    // group g of the chunk: the same lane offsets, another base
    const int64_t gstride = (int64_t) MMH_NG*p.nb01;
    const wtype wl(wm + (int64_t) gc0*gstride, p.nb01, lane);
    const int c0 = lane/4;
    const int r0 = 2*(lane % 4);

    f8_t acc[G][nm];
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int m = 0; m < nm; ++m) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[g][m][l] = 0.0f;
            }
        }
    }

    auto fetch_x = [&](const int k0, uint4 * xv) {
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            xv[t] = make_uint4(0, 0, 0, 0);
            if (xval[t]) {
                xv[t] = *(const uint4 *) (xsrc[t] + k0);
            }
        }
    };
    auto fetch_w = [&](const int step, typename wtype::raw (*wr)[2]) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const wtype wg(wm + (int64_t) (gc0 + (g < ng ? g : 0))*gstride, p.nb01, lane);
            wr[g][0] = wg.fetch(2*step + 0, wave);
            wr[g][1] = wg.fetch(2*step + 1, wave);
        }
    };

    const int nsteps = p.k/ks;

    uint4 xv[cpt];
    typename wtype::raw wr[G][2];
    fetch_x(0, xv);
    fetch_w(0, wr);

    for (int step = 0; step < nsteps; ++step) {
        half * xb = xs + (step % 2)*mt*xst;
        uint4 cxv[cpt];
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            cxv[t] = xv[t];
        }
        typename wtype::raw cwr[G][2];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            cwr[g][0] = wr[g][0];
            cwr[g][1] = wr[g][1];
        }
        if (step + 1 < nsteps) {
            fetch_x(ks*(step + 1), xv);
            fetch_w(step + 1, wr);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            half * ws = wsb + (g*8 + wave)*16*wst;
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                uint32_t wf[4];
                uint32_t wlo[4];
                wl.decode(cwr[g][kt], wf, wlo);
                *(uint32_t *) (ws + c0*wst       + 16*kt + r0)     = wf[0];
                *(uint32_t *) (ws + c0*wst       + 16*kt + r0 + 8) = wf[1];
                *(uint32_t *) (ws + (c0 + 8)*wst + 16*kt + r0)     = wf[2];
                *(uint32_t *) (ws + (c0 + 8)*wst + 16*kt + r0 + 8) = wf[3];
            }
        }
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            if (tid + 256*t < nchunk) {
                *(uint4 *) (xb + xdst[t]) = cxv[t];
            }
        }
        mmh_lds_sync();
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            h8_t a[nm];
#pragma unroll
            for (int m = 0; m < nm; ++m) {
#if defined(RDNA4)
                a[m] = *(const h8_t *) (xb + (16*m + lane % 16)*xst + 16*kk + 8*(lane/16));
#else
                a[m] = *(const h8_t *) (xb + (16*m + lane % 16)*xst + 16*kk);
#endif // defined(RDNA4)
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const half * ws = wsb + (g*8 + wave)*16*wst;
#if defined(RDNA4)
                const h8_t b = *(const h8_t *) (ws + (lane % 16)*wst + 16*kk + 8*(lane/16));
#else
                const h8_t b = *(const h8_t *) (ws + (lane % 16)*wst + 16*kk);
#endif // defined(RDNA4)
#pragma unroll
                for (int m = 0; m < nm; ++m) {
#if defined(RDNA4)
                    acc[g][m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[m], b, acc[g][m]);
#else
                    acc[g][m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[m], b, acc[g][m]);
#endif // defined(RDNA4)
                }
            }
        }
    }

    // output Hadamard, rot_out and the row scale of row i of the item, for group gg of the weight
    auto finish = [&](const int i, const int gg, float v[4]) {
        const int r = rows_sorted[start + i];
        const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, e) + (int64_t) gg*MMH_NG*p.rout_nb0;
        float ro[4];
        mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
        mmh_fwht(v, lane, p.had, p.hscale);
        float * out = mmh_dst_row(p, r) + (int64_t) gg*MMH_NG;
        const float rw = mmh_row_w(p, r);
        const float rs = rscale[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = 32*j + lane;
            mmh_store(p, out, c, v[j]*rs*ro[j]*rw);
        }
    };

#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (g >= ng) {
            break;
        }
#pragma unroll
        for (int e0 = 0; e0 < mt; e0 += ec) {
            __syncthreads();
#pragma unroll
            for (int m = e0/16; m < (e0 + ec)/16; ++m) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
#if defined(RDNA4)
                    ys[(16*m - e0 + 8*(lane/16) + l)*yst + 16*wave + lane % 16] = acc[g][m][l];
#else
                    ys[(16*m - e0 + 2*l + lane/16)*yst + 16*wave + lane % 16] = acc[g][m][l];
#endif // defined(RDNA4)
                }
            }
            __syncthreads();
            for (int i = e0 + wave; i < min(cnt, e0 + ec); i += 8) {
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[j] = ys[(i - e0)*yst + 32*j + lane];
                }
                finish(i, gc0 + g, v);
            }
        }
    }
#else
    GGML_UNUSED_VARS(p, w, xh, rscale, items, n_items, rows_sorted, n_groups);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && (defined(RDNA4) || defined(RDNA3))
}

// strixllama: the GEMM of dense ops of many rows over fp16 tiles. mmh_gemm decodes every weight tile again in each
// block of rows (40 times for 5000 rows) and reads its staged rows 64 bytes per row and step, kilobytes apart; here the
// weight is decoded once (mmh_decode_t) and both operands are laid out per 16 k step and block densely:
//   x: [row block of 128][k/16][128 rows][16 k] (mmh_prologue<false, true, true>), w: [group][k/16][128 columns][16 k].
// The products and their order per output (16 k per WMMA, k ascending) are those of mmh_gemm: the same results.

// a 128-column group's tiles of 8 steps (blockIdx.y) to fp16, through LDS so that the stores are whole 16-byte chunks
template <ggml_type type>
static __global__ void __launch_bounds__(256) mmh_decode_t(const mmh_params p, const char * __restrict__ w, half * __restrict__ wd) {
    typedef mmh_weight<type, MMH_WIDE_COAL> wtype;
    constexpr int wst = 32 + 8;
    __shared__ __align__(16) half ws[2][MMH_NG*wst];

    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int g    = blockIdx.x;
    const int nst  = (int) (p.k/32);
    const wtype wl(mmh_weight_matrix(p, w, 0) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const int c0 = lane/4;
    const int r0 = 2*(lane % 4);

    for (int i = 0; i < 8; ++i) {
        const int step = 8*blockIdx.y + i;
        if (step >= nst) {
            break;
        }
        half * wb = ws[i % 2] + 16*wave*wst;
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            uint32_t wf[4];
            uint32_t wlo[4];
            wl.decode(wl.fetch(2*step + kt, wave), wf, wlo);
            *(uint32_t *) (wb + c0*wst       + 16*kt + r0)     = wf[0];
            *(uint32_t *) (wb + c0*wst       + 16*kt + r0 + 8) = wf[1];
            *(uint32_t *) (wb + (c0 + 8)*wst + 16*kt + r0)     = wf[2];
            *(uint32_t *) (wb + (c0 + 8)*wst + 16*kt + r0 + 8) = wf[3];
        }
        __syncthreads();
        // the two 16-k steps 2*step, 2*step + 1: [128 columns][16 k] each
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int c    = threadIdx.x + 256*t;
            const int kt   = c/256;
            const int col  = (c % 256)/2;
            const int part = c % 2;
            half * dst = wd + ((int64_t) g*2*nst + 2*step + kt)*(MMH_NG*16);
            *(uint4 *) (dst + col*16 + 8*part) = *(const uint4 *) (ws[i % 2] + col*wst + 16*kt + 8*part);
        }
    }
}

#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
typedef int v8i_t __attribute__((ext_vector_type(8)));
// one gfx11 f16 WMMA (f32 accumulator) on operands of 8 dwords: with the builtin's 16 x f16 operands clang copies every
// fragment loaded from LDS half by half (v_bfi_b32, 4 per WMMA here); without them the GEMM runs 5% faster
static __device__ __forceinline__ void mmh_wmma_f16(float __attribute__((ext_vector_type(8))) & acc, const v8i_t a, const v8i_t b) {
    asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
}
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)

// block: 128 rows x one 128-column group, 8 waves as 2 (m) x 4 (n) of 64 x 32, steps of 16 k through double-buffered
// LDS (one barrier per step), two blocks per CU: the small steps keep the LDS of two blocks, whose loads and barriers
// then hide behind each other's WMMAs (+24% over 32 k steps and one block). Blocks run in panels of 8 row blocks (row
// block fastest, then the groups), so that the blocks in flight share their rows and columns in L2.
static __global__ void __launch_bounds__(256, 2) mmh_gemm_t(const mmh_params p, const half * __restrict__ wd,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int n_rb, const int n_groups) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    typedef float    f8_t  __attribute__((ext_vector_type(8)));
    constexpr int BM  = 128;
    constexpr int ST  = 16 + 8;       // halves per staged row
    constexpr int yst = MMH_NG + 4;   // floats per output row
    constexpr int ec  = 32;           // output rows per pass through LDS
    constexpr int MTW = 4;            // m-tiles per wave
    constexpr int NTW = 2;            // n-tiles per wave
    constexpr int main_bytes = 2*2*BM*ST*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    half  * as = (half *) smem;       // [2][BM*ST]
    half  * bs = as + 2*BM*ST;        // [2][128*ST]
    float * ys = (float *) smem;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int wn   = wave % 4;
    const int wm   = wave / 4;

    constexpr int P = 8;
    const int bid   = blockIdx.x;
    const int panel = bid/(P*n_groups);
    const int pr    = min(P, n_rb - panel*P);
    const int in    = bid - panel*P*n_groups;
    const int rb    = panel*P + in % pr;
    const int g     = in/pr;

    const int nst = (int) (p.k/16);
    const uint4 * ag = (const uint4 *) (xh + (int64_t) rb*BM*p.k) + tid;
    const uint4 * bg = (const uint4 *) (wd + (int64_t) g*MMH_NG*p.k) + tid;
    // chunk tid of a step: row tid/2, 8 k at 8*(tid % 2)
    const int d0 = (tid/2)*ST + 8*(tid % 2);

    f8_t acc[MTW][NTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[i][j][l] = 0.0f;
            }
        }
    }

    uint4 ra = ag[0], rbw = bg[0];
    *(uint4 *) (as + d0) = ra;
    *(uint4 *) (bs + d0) = rbw;
    mmh_lds_sync();

    for (int step = 0; step < nst; ++step) {
        const int cur = step % 2;
        if (step + 1 < nst) {
            ra  = ag[(step + 1)*(BM*16/8)];
            rbw = bg[(step + 1)*(BM*16/8)];
        }
        const half * xa = as + cur*BM*ST;
        const half * xb = bs + cur*BM*ST;
        v8i_t a[MTW];
#pragma unroll
        for (int i = 0; i < MTW; ++i) {
            a[i] = *(const v8i_t *) (xa + (16*(MTW*wm + i) + lane % 16)*ST);
        }
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
            const v8i_t b = *(const v8i_t *) (xb + (16*(NTW*wn + j) + lane % 16)*ST);
#pragma unroll
            for (int i = 0; i < MTW; ++i) {
                mmh_wmma_f16(acc[i][j], a[i], b);
            }
        }
        if (step + 1 < nst) {
            *(uint4 *) (as + (cur ^ 1)*BM*ST + d0) = ra;
            *(uint4 *) (bs + (cur ^ 1)*BM*ST + d0) = rbw;
        }
        mmh_lds_sync();
    }

    const int row0 = rb*BM;
    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, 0) + (int64_t) g*MMH_NG*p.rout_nb0;
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
    // accumulator l of a tile holds row 2*l + lane/16, column lane % 16
#pragma unroll
    for (int e0 = 0; e0 < BM; e0 += ec) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < MTW; ++i) {
            const int m = MTW*wm + i;
            if (m < e0/16 || m >= (e0 + ec)/16) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    ys[(16*m - e0 + 2*l + lane/16)*yst + 16*(NTW*wn + j) + lane % 16] = acc[i][j][l];
                }
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < e0 + ec; i += 8) {
            const int r = row0 + i;
            if (r >= p.n_rows) {
                break;
            }
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = ys[(i - e0)*yst + 32*j + lane];
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rw = mmh_row_w(p, r);
            const float rs = rscale[r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                mmh_store(p, out, c, v[j]*rs*ro[j]*rw);
            }
        }
    }
#else
    GGML_UNUSED_VARS(p, wd, xh, rscale, n_rb, n_groups);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

// strixllama: the expert GEMM over a panel layout of the staged rows. mmh_gemm_wide gathers its rows 64 bytes per row
// and step from all over the staged rows; here the routing pass runs first and the prologue writes each row at its
// sorted position q (p.pos) in panels of 16 rows (mmh_xh_panel), so that the rows of an item are a few dense chunks. A block owns one item (up to 128 rows of one expert) and one 128-column group: 8 waves as
// 2 (m) x 4 (n), steps of 16 k through double-buffered LDS (two blocks per CU), wave w decodes n-tile w of the step
// into the block's weight tile. The WMMAs of 16-row tiles past the item's rows are skipped; the two wave rows own the
// even and the odd 16-row tiles, so that a short item keeps both busy. The products and their
// order per output are those of mmh_gemm: the same results.
template <ggml_type type, int BM, int WM>
static __global__ void __launch_bounds__(256, 2) mmh_gemm_et(const mmh_params p, const char * __restrict__ w,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    typedef float f8_t __attribute__((ext_vector_type(8)));
    typedef mmh_weight<type, MMH_WIDE_COAL> wtype;
    constexpr int ST  = 16 + 8;       // halves per staged row
    constexpr int yst = MMH_NG + 4;   // floats per output row
    constexpr int ec  = 32;           // output rows per pass through LDS
    constexpr int MTW = BM/(16*WM);   // m-tiles per wave
    constexpr int NTW = WM;           // n-tiles per wave: the 8 n-tiles over 8/WM waves
    static_assert(NTW*(8/WM) == 8 && MTW >= 1, "wave layout");
    constexpr int main_bytes = 2*(BM + MMH_NG)*ST*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    half  * as = (half *) smem;       // [2][BM*ST]
    half  * bs = as + 2*BM*ST;        // [2][128*ST]
    float * ys = (float *) smem;

    if ((int) blockIdx.y >= *n_items) {
        return;
    }
    const int e     = mmh_uniform(items[3*blockIdx.y + 0]);
    const int start = items[3*blockIdx.y + 1];
    const int cnt   = mmh_uniform(items[3*blockIdx.y + 2]);
    if (mmh_other_class(p, e)) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int wn   = wave % (8/WM);
    const int wm   = wave / (8/WM);
    const int g    = blockIdx.x;

    const int nst = (int) (p.k/16);
    // chunk tid of a step: item row tid/2 (sorted position start + tid/2), 8 k at 8*(tid % 2); rows past the item
    // read the next rows (the staged rows have 128 rows of padding), their products are not stored
    const bool aload = tid < 2*BM;   // BM 64: half the threads load the rows
    int qa = start + (aload ? tid/2 : 0);
    if (p.tok_rows) {
        // one staged row per token: the token of the item's row
        qa = rows_sorted[start + (aload && tid/2 < cnt ? tid/2 : 0)]/p.tok_rows;
    }
    constexpr int spc = MMH_ET_CW/16;   // steps per chunk
    const half * ag = xh + (int64_t) (qa/16)*(nst/spc)*(16*MMH_ET_CW) + (qa % 16)*MMH_ET_CW + 8*(tid % 2);
    const int d0 = (tid/2)*ST + 8*(tid % 2);

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const int c0 = lane/4;
    const int r0 = 2*(lane % 4);
    auto put_w = [&](half * wb, const typename wtype::raw & wr) {
        uint32_t wf[4];
        uint32_t wlo[4];
        wl.decode(wr, wf, wlo);
        *(uint32_t *) (wb + c0*ST       + r0)     = wf[0];
        *(uint32_t *) (wb + c0*ST       + r0 + 8) = wf[1];
        *(uint32_t *) (wb + (c0 + 8)*ST + r0)     = wf[2];
        *(uint32_t *) (wb + (c0 + 8)*ST + r0 + 8) = wf[3];
    };

    f8_t acc[MTW][NTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[i][j][l] = 0.0f;
            }
        }
    }

    uint4 ra = aload ? *(const uint4 *) ag : make_uint4(0, 0, 0, 0);
    typename wtype::raw wr = wl.fetch(0, wave);
    if (aload) {
        *(uint4 *) (as + d0) = ra;
    }
    put_w(bs + 16*wave*ST, wr);
    mmh_lds_sync();

    for (int step = 0; step < nst; ++step) {
        const int cur = step % 2;
        if (step + 1 < nst) {
            if (aload) {
                ra = *(const uint4 *) (ag + ((step + 1)/spc)*(16*MMH_ET_CW) + ((step + 1) % spc)*16);
            }
            wr = wl.fetch(step + 1, wave);
        }
        const half * xa = as + cur*BM*ST;
        const half * xb = bs + cur*MMH_NG*ST;
        v8i_t a[MTW];
#pragma unroll
        for (int i = 0; i < MTW; ++i) {
            if (16*(WM*i + wm) < cnt) {
                a[i] = *(const v8i_t *) (xa + (16*(WM*i + wm) + lane % 16)*ST);
            }
        }
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
            const v8i_t b = *(const v8i_t *) (xb + (16*(NTW*wn + j) + lane % 16)*ST);
#pragma unroll
            for (int i = 0; i < MTW; ++i) {
                if (16*(WM*i + wm) < cnt) {
                    mmh_wmma_f16(acc[i][j], a[i], b);
                }
            }
        }
        if (step + 1 < nst) {
            if (aload) {
                *(uint4 *) (as + (cur ^ 1)*BM*ST + d0) = ra;
            }
            put_w(bs + (cur ^ 1)*MMH_NG*ST + 16*wave*ST, wr);
        }
        mmh_lds_sync();
    }

    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
    // accumulator l of a tile holds row 2*l + lane/16, column lane % 16
#pragma unroll
    for (int e0 = 0; e0 < BM; e0 += ec) {
        if (e0 >= cnt) {
            break;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < MTW; ++i) {
            const int m = WM*i + wm;
            if (m < e0/16 || m >= (e0 + ec)/16) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    ys[(16*m - e0 + 2*l + lane/16)*yst + 16*(NTW*wn + j) + lane % 16] = acc[i][j][l];
                }
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < min(cnt, e0 + ec); i += 8) {
            const int r = rows_sorted[start + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = ys[(i - e0)*yst + 32*j + lane];
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rw = mmh_row_w(p, r);
            const float rs = rscale[p.tok_rows ? r/p.tok_rows : r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                mmh_store(p, out, c, v[j]*rs*ro[j]*rw);
            }
        }
    }
#else
    GGML_UNUSED_VARS(p, w, xh, rscale, items, n_items, rows_sorted);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

// strixllama: the expert GEMM, wave layout 4 (m) x 2 (n) over 32-k chunks. A block owns one item (up to 128 rows of one
// expert) and one 128-column group; wave (wm, wn) owns m-tiles wm and wm + 4 and n-tiles 4*wn .. 4*wn + 3 (2 + 4 fragment
// reads per 8 WMMAs). The weight tiles of a chunk (2 steps x 8 n-tiles) are decoded by the 8 waves during the chunk
// before (wave w: n-tile w) into a block-shared buffer; the staged rows come in 32-k chunks two ahead: one barrier per
// 2 steps. Weights are fetched PFW steps ahead of their decode. The step loop is instantiated per wave by its m-tile
// count (0, 1 or 2) with the last chunks peeled, so that the steady state has no branches and the fragment loads
// issue ahead of their WMMAs. The products and their k order per output are those of mmh_gemm_et: the same results.
template <ggml_type type>
static __global__ void __launch_bounds__(256, 8) mmh_gemm_et3(const mmh_params p, const char * __restrict__ w,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    typedef float f8_t __attribute__((ext_vector_type(8)));
    typedef mmh_weight<type, false> wtype;
    constexpr int BM  = 128;
    constexpr int CK  = 32;           // k per LDS chunk: 2 steps per barrier
    constexpr int yst = MMH_NG + 4;   // floats per output row
    constexpr int ec  = 32;           // output rows per pass through LDS
    constexpr int PFW = 8;            // weight tiles fetched ahead of their decode (divides the 8 steps of 4 chunks)
    constexpr int TB  = 16*16;        // halves per weight tile in LDS: 16 n rows of 16 k, the two 16-byte pieces swizzled
    constexpr int a_halves   = 2*BM*CK;             // [2][BM][CK], the 16-byte pieces of a row swizzled
    constexpr int b_halves   = 2*2*8*TB;            // [2 chunks][2 steps][8 n-tiles][TB]
    constexpr int main_bytes = (a_halves + b_halves)*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    half  * as = (half *) smem;
    half  * bs = as + a_halves;
    float * ys = (float *) smem;

    if ((int) blockIdx.y >= *n_items) {
        return;
    }
    const int e     = mmh_uniform(items[3*blockIdx.y + 0]);
    const int start = items[3*blockIdx.y + 1];
    const int cnt   = mmh_uniform(items[3*blockIdx.y + 2]);
    if (mmh_other_class(p, e)) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int wm   = wave % 4;        // wave row: m-tiles wm and wm + 4
    const int wn   = wave / 4;        // wave column: n-tiles 4*wn .. 4*wn + 3
    const int g    = blockIdx.x;

    const int nst = (int) (p.k/16);   // a multiple of 8 (k % 128 == 0)
    const int nch = nst/2;            // chunks, a multiple of 4
    // staged rows: thread tid loads piece tid % 4 (8 halves) of item rows tid/4 and tid/4 + 64 of each chunk; rows past
    // the item read the next rows (the staged rows have 128 rows of padding), their products are not stored
    const int ar0 = tid/4;
    const int ar1 = ar0 + 64;
    const int apc = tid % 4;
    int q0 = start + ar0;
    int q1 = start + ar1;
    if (p.tok_rows) {
        // one staged row per token: the token of the item's row
        q0 = rows_sorted[start + (ar0 < cnt ? ar0 : 0)]/p.tok_rows;
        q1 = rows_sorted[start + (ar1 < cnt ? ar1 : 0)]/p.tok_rows;
    }
    const int kpan = (int) (p.k/MMH_ET_CW);   // 128-k panels per row
    const half * ag0 = xh + (int64_t) (q0/16)*kpan*(16*MMH_ET_CW) + (q0 % 16)*MMH_ET_CW + 8*apc;
    const half * ag1 = xh + (int64_t) (q1/16)*kpan*(16*MMH_ET_CW) + (q1 % 16)*MMH_ET_CW + 8*apc;
    // LDS: row r at r*CK, piece j at 8*(j ^ ((r/2) % 4)): 8 consecutive rows read one piece from 8 different 16-byte
    // bank groups
    const int ad0 = ar0*CK + 8*(apc ^ ((ar0/2) % 4));
    const int ad1 = ar1*CK + 8*(apc ^ ((ar1/2) % 4));
    struct a_regs { uint4 v[2]; };
    auto load_a = [&](const int c) {
        const int o = (c/4)*(16*MMH_ET_CW) + (c % 4)*CK;
        a_regs r;
        r.v[0] = *(const uint4 *) (ag0 + o);
        r.v[1] = *(const uint4 *) (ag1 + o);
        return r;
    };
    auto store_a = [&](const int buf, const a_regs r) {
        *(uint4 *) (as + buf*BM*CK + ad0) = r.v[0];
        *(uint4 *) (as + buf*BM*CK + ad1) = r.v[1];
    };

    // weights: wave w fetches and decodes tile (step, n-tile w). In LDS a tile is 16 n rows of 16 k, piece k/8 of row n
    // at 8*((k/8) ^ ((n/4) % 2)): 8 consecutive rows read one piece from 8 different bank groups
    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const int c0  = lane/4;
    const int r0  = 2*(lane % 4);
    const int bsw = (c0/4) % 2;       // the same for c0 + 8
    auto put_w = [&](half * wb, const typename wtype::raw wr) {
        uint32_t wf[4];
        uint32_t wlo[4];
        wl.decode(wr, wf, wlo);
        *(uint32_t *) (wb + c0*16       + 8*(0 ^ bsw) + r0) = wf[0];
        *(uint32_t *) (wb + c0*16       + 8*(1 ^ bsw) + r0) = wf[1];
        *(uint32_t *) (wb + (c0 + 8)*16 + 8*(0 ^ bsw) + r0) = wf[2];
        *(uint32_t *) (wb + (c0 + 8)*16 + 8*(1 ^ bsw) + r0) = wf[3];
    };
    auto btile = [&](const int buf, const int kk, const int nt) {
        return bs + ((buf*2 + kk)*8 + nt)*TB;
    };
    // fragments: A of m-tile mt at step kk of the chunk in buffer buf (row 16*mt + lane % 16, pieces 2*kk, 2*kk + 1),
    // B of n-tile nt (row lane % 16 of the tile, its two pieces)
    const int arow = lane % 16;
    const int asw  = (arow/2) % 4;
    const int bsr  = (arow/4) % 2;
    auto get_a = [&](const int buf, const int kk, const int mt) {
        const half * xr = as + buf*BM*CK + (16*mt + arow)*CK;
        const uint4 lo = *(const uint4 *) (xr + 8*((2*kk) ^ asw));
        const uint4 hi = *(const uint4 *) (xr + 8*((2*kk + 1) ^ asw));
        v8i_t a;
        a[0] = (int) lo.x; a[1] = (int) lo.y; a[2] = (int) lo.z; a[3] = (int) lo.w;
        a[4] = (int) hi.x; a[5] = (int) hi.y; a[6] = (int) hi.z; a[7] = (int) hi.w;
        return a;
    };
    auto get_b = [&](const int buf, const int kk, const int nt) {
        const half * xr = btile(buf, kk, nt) + arow*16;
        const uint4 lo = *(const uint4 *) (xr + 8*(0 ^ bsr));
        const uint4 hi = *(const uint4 *) (xr + 8*(1 ^ bsr));
        v8i_t b;
        b[0] = (int) lo.x; b[1] = (int) lo.y; b[2] = (int) lo.z; b[3] = (int) lo.w;
        b[4] = (int) hi.x; b[5] = (int) hi.y; b[6] = (int) hi.z; b[7] = (int) hi.w;
        return b;
    };
    const int mts = (cnt + 15)/16;                    // m-tiles of the item
    const int ntw = (wm < mts ? 1 : 0) + (wm + 4 < mts ? 1 : 0);   // m-tiles of this wave

    f8_t acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[i][j][l] = 0.0f;
            }
        }
    }

    // tiles 0 and 1 go straight into chunk 0; tile t >= 2 waits in wr[t % PFW] from its fetch at step t - 2 - PFW
    // (the prologue fetches 2 .. PFW + 1) to its decode at step t - 2
    put_w(btile(0, 0, wave), wl.fetch(0, wave));
    put_w(btile(0, 1, wave), wl.fetch(1, wave));
    typename wtype::raw wr[PFW];
#pragma unroll
    for (int t = 2; t < PFW + 2; ++t) {
        wr[t % PFW] = wl.fetch(min(t, nst - 1), wave);
    }
    a_regs ra[2];
    ra[0] = load_a(0);
    ra[1] = load_a(1);
    store_a(0, ra[0]);
    mmh_lds_sync();

    // one chunk c = cb + u (cb a multiple of 4): A and B buffers u % 2, ra[u % 2] held the chunk and is free. TAIL: the
    // last chunks, every prefetch bounds-checked; otherwise all on
    auto chunk = [&](auto nt_c, auto tail_c, const int c, const int u) {
        constexpr int  NT   = decltype(nt_c)::value;
        constexpr bool TAIL = decltype(tail_c)::value;
        if (!TAIL || c + 2 < nch) {
            ra[u % 2] = load_a(c + 2);
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int s = 2*c + kk;
            // the tile of step s + 2 (chunk c + 1, step kk) decoded; its ring slot refilled with tile s + 2 + PFW
            if (!TAIL || s + 2 < nst) {
                put_w(btile((u + 1) % 2, kk, wave), wr[(2*u + kk + 2) % PFW]);
            }
            if (!TAIL || s + 2 + PFW < nst) {
                wr[(2*u + kk + 2) % PFW] = wl.fetch(s + 2 + PFW, wave);
            }
            if constexpr (NT > 0) {
                v8i_t a[2];
                v8i_t b[4];
                a[0] = get_a(u % 2, kk, wm);
                if constexpr (NT > 1) {
                    a[1] = get_a(u % 2, kk, wm + 4);
                }
                b[0] = get_b(u % 2, kk, 4*wn);
                b[1] = get_b(u % 2, kk, 4*wn + 1);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j + 2 < 4) {
                        b[j + 2] = get_b(u % 2, kk, 4*wn + j + 2);
                    }
                    mmh_wmma_f16(acc[0][j], a[0], b[j]);
                    if constexpr (NT > 1) {
                        mmh_wmma_f16(acc[1][j], a[1], b[j]);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
        }
        if (!TAIL || c + 1 < nch) {
            store_a((u + 1) % 2, ra[(u + 1) % 2]);   // chunk c + 1, loaded two chunks ago
        }
        mmh_lds_sync();
    };
    auto run = [&](auto nt_c) {
        int cb = 0;
        for (; cb + 8 < nch; cb += 4) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                chunk(nt_c, mmh_bc<false>{}, cb + u, u);
            }
        }
        for (; cb < nch; cb += 4) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                chunk(nt_c, mmh_bc<true>{}, cb + u, u);
            }
        }
    };
    if (ntw == 0) {
        run(mmh_ic<0>{});
    } else if (ntw == 1) {
        run(mmh_ic<1>{});
    } else {
        run(mmh_ic<2>{});
    }

    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
    // accumulator l of a tile holds row 2*l + lane/16, column lane % 16
#pragma unroll
    for (int e0 = 0; e0 < BM; e0 += ec) {
        if (e0 >= cnt) {
            break;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int m = 4*i + wm;
            if (m < e0/16 || m >= (e0 + ec)/16) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    ys[(16*m - e0 + 2*l + lane/16)*yst + 16*(4*wn + j) + lane % 16] = acc[i][j][l];
                }
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < min(cnt, e0 + ec); i += 8) {
            const int r = rows_sorted[start + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = ys[(i - e0)*yst + 32*j + lane];
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rs = rscale[p.tok_rows ? r/p.tok_rows : r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                mmh_store(p, out, c, v[j]*rs*ro[j]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(p, w, xh, rscale, items, n_items, rows_sorted);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

// strixllama: the expert GEMM with wave-private weight strips. A block owns one item (up to 128 rows of one expert) and
// one 128-column group; wave w owns n-tile w (16 columns) and all 8 m-tiles, decodes its own weight tile of each step
// into a private LDS strip (no block barrier for the weights) and reads the staged rows from a 32-k chunk the block
// shares (one barrier per 2 steps, half of mmh_gemm_et's). Weights are fetched PFW steps ahead, rows two chunks ahead.
// The step loop is instantiated per item size (2, 4, 6 or 8 m-tiles) and the last 4 chunks are peeled, so that the
// steady state has no branches: the fragment loads of a step issue together ahead of its WMMAs.
// The products and their k order per output are those of mmh_gemm_et: the same results.
#ifndef MMH_ET2_PFW
#define MMH_ET2_PFW 4
#endif
template <int N> struct mmh_ic { static constexpr int  value = N; };
template <bool V> struct mmh_bc { static constexpr bool value = V; };

template <ggml_type type>
static __global__ void __launch_bounds__(256, 8) mmh_gemm_et2(const mmh_params p, const char * __restrict__ w,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    typedef float f8_t __attribute__((ext_vector_type(8)));
    typedef mmh_weight<type, false> wtype;
    constexpr int BM  = 128;
    constexpr int CK  = 32;           // k per LDS chunk of staged rows: 2 steps per barrier
    constexpr int ST  = 16 + 8;       // halves per row of a private weight strip
    constexpr int yst = MMH_NG + 4;   // floats per output row
    constexpr int ec  = 32;           // output rows per pass through LDS
    constexpr int PFW = MMH_ET2_PFW;  // weight tiles in flight per wave (divides the 8 steps of 4 chunks)
    constexpr int a_halves   = 2*BM*CK;             // [2][BM][CK], the 16-byte pieces of a row swizzled
    constexpr int b_halves   = 8*2*16*ST;           // [8 waves][2][16 n][ST]
    constexpr int main_bytes = (a_halves + b_halves)*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    half  * as = (half *) smem;
    half  * bs = as + a_halves;
    float * ys = (float *) smem;

    if ((int) blockIdx.y >= *n_items) {
        return;
    }
    const int e     = mmh_uniform(items[3*blockIdx.y + 0]);
    const int start = items[3*blockIdx.y + 1];
    const int cnt   = mmh_uniform(items[3*blockIdx.y + 2]);
    if (mmh_other_class(p, e)) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int g    = blockIdx.x;

    const int nst = (int) (p.k/16);   // a multiple of 8 (k % 128 == 0)
    const int nch = nst/2;            // chunks, a multiple of 4
    // staged rows: thread tid loads piece tid % 4 (8 halves) of item rows tid/4 and tid/4 + 64 of each chunk; rows past
    // the item read the next rows (the staged rows have 128 rows of padding), their products are not stored
    const int ar0 = tid/4;
    const int ar1 = ar0 + 64;
    const int apc = tid % 4;
    int q0 = start + ar0;
    int q1 = start + ar1;
    if (p.tok_rows) {
        // one staged row per token: the token of the item's row
        q0 = rows_sorted[start + (ar0 < cnt ? ar0 : 0)]/p.tok_rows;
        q1 = rows_sorted[start + (ar1 < cnt ? ar1 : 0)]/p.tok_rows;
    }
    const int kpan = (int) (p.k/MMH_ET_CW);   // 128-k panels per row
    const half * ag0 = xh + (int64_t) (q0/16)*kpan*(16*MMH_ET_CW) + (q0 % 16)*MMH_ET_CW + 8*apc;
    const half * ag1 = xh + (int64_t) (q1/16)*kpan*(16*MMH_ET_CW) + (q1 % 16)*MMH_ET_CW + 8*apc;
    // LDS: row r at r*CK, piece j at 8*(j ^ ((r/2) % 4)): 8 consecutive rows read one piece from 8 different 16-byte
    // bank groups
    const int ad0 = ar0*CK + 8*(apc ^ ((ar0/2) % 4));
    const int ad1 = ar1*CK + 8*(apc ^ ((ar1/2) % 4));
    struct a_regs { uint4 v[2]; };
    auto load_a = [&](const int c) {
        const int o = (c/4)*(16*MMH_ET_CW) + (c % 4)*CK;
        a_regs r;
        r.v[0] = *(const uint4 *) (ag0 + o);
        r.v[1] = *(const uint4 *) (ag1 + o);
        return r;
    };
    auto store_a = [&](const int buf, const a_regs r) {
        *(uint4 *) (as + buf*BM*CK + ad0) = r.v[0];
        *(uint4 *) (as + buf*BM*CK + ad1) = r.v[1];
    };

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const int c0 = lane/4;
    const int r0 = 2*(lane % 4);
    half * bw = bs + 2*wave*16*ST;    // this wave's two strips
    auto put_w = [&](half * wb, const typename wtype::raw wr) {
        uint32_t wf[4];
        uint32_t wlo[4];
        wl.decode(wr, wf, wlo);
        *(uint32_t *) (wb + c0*ST       + r0)     = wf[0];
        *(uint32_t *) (wb + c0*ST       + r0 + 8) = wf[1];
        *(uint32_t *) (wb + (c0 + 8)*ST + r0)     = wf[2];
        *(uint32_t *) (wb + (c0 + 8)*ST + r0 + 8) = wf[3];
    };
    // the A fragment of m-tile mt at step kk of the chunk in buffer buf: row 16*mt + lane % 16, pieces 2*kk, 2*kk + 1
    const int arow = lane % 16;
    const int asw  = (arow/2) % 4;
    auto get_a = [&](const int buf, const int kk, const int mt) {
        const half * xr = as + buf*BM*CK + (16*mt + arow)*CK;
        const uint4 lo = *(const uint4 *) (xr + 8*((2*kk) ^ asw));
        const uint4 hi = *(const uint4 *) (xr + 8*((2*kk + 1) ^ asw));
        v8i_t a;
        a[0] = (int) lo.x; a[1] = (int) lo.y; a[2] = (int) lo.z; a[3] = (int) lo.w;
        a[4] = (int) hi.x; a[5] = (int) hi.y; a[6] = (int) hi.z; a[7] = (int) hi.w;
        return a;
    };
    const int mts = (cnt + 15)/16;    // m-tiles of the item

    f8_t acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[i][l] = 0.0f;
        }
    }

    typename wtype::raw wr[PFW];
#pragma unroll
    for (int i = 0; i < PFW; ++i) {
        wr[i] = wl.fetch(i < nst ? i : 0, wave);
    }
    a_regs ra[2];
    ra[0] = load_a(0);
    ra[1] = load_a(nch > 1 ? 1 : 0);
    store_a(0, ra[0]);
    put_w(bw, wr[0]);
    mmh_lds_sync();

    // one chunk c = cb + u (cb a multiple of 4): A buffer u % 2, ra[u % 2] held it and is free; its steps' tiles are
    // wr[2*u + kk]. TAIL: the last 4 chunks, every prefetch bounds-checked; otherwise all on
    auto chunk = [&](auto mt_c, auto tail_c, const int c, const int u) {
        constexpr int  MT   = decltype(mt_c)::value;
        constexpr bool TAIL = decltype(tail_c)::value;
        if (!TAIL || c + 2 < nch) {
            ra[u % 2] = load_a(c + 2);
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int s = 2*c + kk;       // step: its strip is s % 2 = kk
            if (!TAIL || s + 1 < nst) {
                put_w(bw + (1 - kk)*16*ST, wr[(2*u + kk + 1) % PFW]);
            }
            if (!TAIL || s + PFW < nst) {
                wr[(2*u + kk) % PFW] = wl.fetch(s + PFW, wave);
            }
            const v8i_t b = *(const v8i_t *) (bw + kk*16*ST + arow*ST);
            // the A fragments two tiles ahead of their WMMA (3 in flight: 24 registers). The scheduling barriers keep
            // the loads there: left alone, the scheduler sinks each load next to its WMMA and waits for it (lgkmcnt(0))
            v8i_t a[MT];
            a[0] = get_a(u % 2, kk, 0);
            if constexpr (MT > 1) {
                a[1] = get_a(u % 2, kk, 1);
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mt + 2 < MT) {
                    a[mt + 2] = get_a(u % 2, kk, mt + 2);
                }
                mmh_wmma_f16(acc[mt], a[mt], b);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        if (!TAIL || c + 1 < nch) {
            store_a((u + 1) % 2, ra[(u + 1) % 2]);   // chunk c + 1, loaded two chunks ago
        }
        mmh_lds_sync();
    };
    auto run = [&](auto mt_c) {
        int cb = 0;
        for (; cb + 4 < nch; cb += 4) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                chunk(mt_c, mmh_bc<false>{}, cb + u, u);
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            chunk(mt_c, mmh_bc<true>{}, cb + u, u);
        }
    };
    if (mts <= 2) {
        run(mmh_ic<2>{});
    } else if (mts <= 4) {
        run(mmh_ic<4>{});
    } else if (mts <= 6) {
        run(mmh_ic<6>{});
    } else {
        run(mmh_ic<8>{});
    }

    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
    // accumulator l of a tile holds row 2*l + lane/16, column lane % 16
#pragma unroll
    for (int e0 = 0; e0 < BM; e0 += ec) {
        if (e0 >= cnt) {
            break;
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < e0/16 || m >= (e0 + ec)/16) {
                continue;
            }
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                ys[(16*m - e0 + 2*l + lane/16)*yst + 16*wave + lane % 16] = acc[m][l];
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < min(cnt, e0 + ec); i += 8) {
            const int r = rows_sorted[start + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = ys[(i - e0)*yst + 32*j + lane];
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rs = rscale[p.tok_rows ? r/p.tok_rows : r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                mmh_store(p, out, c, v[j]*rs*ro[j]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(p, w, xh, rscale, items, n_items, rows_sorted);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

// strixllama: the tiled dense GEMM over 32-k chunks: mmh_gemm_t's block (128 rows x one 128-column group, 8 waves as
// 2 (m) x 4 (n) of 64 x 32, two blocks per CU, panel block swizzle) with the LDS holding 32 k per chunk (one barrier
// per 2 steps instead of 1), the 16-byte pieces of a row swizzled instead of padded (16 KB per chunk, 32 KB double-
// buffered), the global loads two chunks ahead and the last chunks peeled, so that the steady state has no branches.
// The same products in the same k order per output as mmh_gemm_t: the same results.
static __global__ void __launch_bounds__(256, 8) mmh_gemm_t2(const mmh_params p, const half * __restrict__ wd,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int n_rb, const int n_groups) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    typedef float    f8_t  __attribute__((ext_vector_type(8)));
    constexpr int BM  = 128;
    constexpr int CK  = 32;           // k per LDS chunk: 2 steps per barrier
    constexpr int yst = MMH_NG + 4;   // floats per output row
    constexpr int ec  = 32;           // output rows per pass through LDS
    constexpr int MTW = 4;            // m-tiles per wave
    constexpr int NTW = 2;            // n-tiles per wave
    constexpr int c_halves   = BM*CK;                           // one chunk of rows (or of columns)
    constexpr int main_bytes = 4*c_halves*(int) sizeof(half);   // [2 buffers][rows, columns]
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    half  * as = (half *) smem;       // [2][BM*CK]
    half  * bs = as + 2*c_halves;     // [2][128*CK]
    float * ys = (float *) smem;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int wn   = wave % 4;
    const int wm   = wave / 4;

    constexpr int P = 8;
    const int bid   = blockIdx.x;
    const int panel = bid/(P*n_groups);
    const int pr    = min(P, n_rb - panel*P);
    const int in    = bid - panel*P*n_groups;
    const int rb    = panel*P + in % pr;
    const int g     = in/pr;

    const int nst = (int) (p.k/16);   // a multiple of 8 (k % 128 == 0)
    const int nch = nst/2;            // chunks, a multiple of 4
    // the step tiles of a row block / column group: [step][128 rows][16 k]; thread tid takes uint4 tid of each tile
    // (row tid/2, 8 k at 8*(tid % 2)); a chunk is two consecutive tiles
    constexpr int TU = BM*16/8;       // uint4 per step tile
    const uint4 * ag = (const uint4 *) (xh + (int64_t) rb*BM*p.k) + tid;
    const uint4 * bg = (const uint4 *) (wd + (int64_t) g*MMH_NG*p.k) + tid;
    // LDS: row r at r*CK, piece j at 8*(j ^ ((r/2) % 4)): 8 consecutive rows read one piece from 8 different 16-byte
    // bank groups
    const int lr  = tid/2;
    const int lp  = tid % 2;
    const int ld0 = lr*CK + 8*(lp ^ ((lr/2) % 4));
    const int ld1 = lr*CK + 8*((lp + 2) ^ ((lr/2) % 4));
    struct regs { uint4 a[2]; uint4 b[2]; };
    auto load = [&](const int c) {
        regs r;
        r.a[0] = ag[(2*c)*TU];
        r.a[1] = ag[(2*c + 1)*TU];
        r.b[0] = bg[(2*c)*TU];
        r.b[1] = bg[(2*c + 1)*TU];
        return r;
    };
    auto store = [&](const int buf, const regs r) {
        *(uint4 *) (as + buf*c_halves + ld0) = r.a[0];
        *(uint4 *) (as + buf*c_halves + ld1) = r.a[1];
        *(uint4 *) (bs + buf*c_halves + ld0) = r.b[0];
        *(uint4 *) (bs + buf*c_halves + ld1) = r.b[1];
    };
    // the fragment of tile t at step kk of a chunk buffer: row 16*t + lane % 16, pieces 2*kk, 2*kk + 1
    const int frow = lane % 16;
    const int fsw  = (frow/2) % 4;
    auto frag = [&](const half * base, const int kk, const int t) {
        const half * xr = base + (16*t + frow)*CK;
        const uint4 lo = *(const uint4 *) (xr + 8*((2*kk) ^ fsw));
        const uint4 hi = *(const uint4 *) (xr + 8*((2*kk + 1) ^ fsw));
        v8i_t v;
        v[0] = (int) lo.x; v[1] = (int) lo.y; v[2] = (int) lo.z; v[3] = (int) lo.w;
        v[4] = (int) hi.x; v[5] = (int) hi.y; v[6] = (int) hi.z; v[7] = (int) hi.w;
        return v;
    };

    f8_t acc[MTW][NTW];
#pragma unroll
    for (int i = 0; i < MTW; ++i) {
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                acc[i][j][l] = 0.0f;
            }
        }
    }

    regs rg[2];
    rg[0] = load(0);
    rg[1] = load(nch > 1 ? 1 : 0);
    store(0, rg[0]);
    mmh_lds_sync();

    // one chunk c = cb + u (cb a multiple of 4): buffer u % 2, rg[u % 2] held it and is free. TAIL: the last 4 chunks,
    // the loads and stores bounds-checked; otherwise all on
    auto chunk = [&](auto tail_c, const int c, const int u) {
        constexpr bool TAIL = decltype(tail_c)::value;
        if (!TAIL || c + 2 < nch) {
            rg[u % 2] = load(c + 2);
        }
        const half * xa = as + (u % 2)*c_halves;
        const half * xb = bs + (u % 2)*c_halves;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            v8i_t a[MTW];
            v8i_t b[NTW];
#pragma unroll
            for (int i = 0; i < MTW; ++i) {
                a[i] = frag(xa, kk, MTW*wm + i);
            }
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
                b[j] = frag(xb, kk, NTW*wn + j);
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
#pragma unroll
                for (int i = 0; i < MTW; ++i) {
                    mmh_wmma_f16(acc[i][j], a[i], b[j]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        if (!TAIL || c + 1 < nch) {
            store((u + 1) % 2, rg[(u + 1) % 2]);   // chunk c + 1, loaded two chunks ago
        }
        mmh_lds_sync();
    };
    int cb = 0;
    for (; cb + 4 < nch; cb += 4) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            chunk(mmh_bc<false>{}, cb + u, u);
        }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        chunk(mmh_bc<true>{}, cb + u, u);
    }

    const int row0 = rb*BM;
    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_nb1, 0) + (int64_t) g*MMH_NG*p.rout_nb0;
    float ro[4];
    mmh_rot4(rout, p.rout_f16, 0, lane, p.rout_nb0/(p.rout_f16 ? 2 : 4), ro);
    // accumulator l of a tile holds row 2*l + lane/16, column lane % 16
#pragma unroll
    for (int e0 = 0; e0 < BM; e0 += ec) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < MTW; ++i) {
            const int m = MTW*wm + i;
            if (m < e0/16 || m >= (e0 + ec)/16) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    ys[(16*m - e0 + 2*l + lane/16)*yst + 16*(NTW*wn + j) + lane % 16] = acc[i][j][l];
                }
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < e0 + ec; i += 8) {
            const int r = row0 + i;
            if (r >= p.n_rows) {
                break;
            }
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = ys[(i - e0)*yst + 32*j + lane];
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rs = rscale[r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                mmh_store(p, out, c, v[j]*rs*ro[j]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(p, wd, xh, rscale, n_rb, n_groups);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
}

// strixllama: mmh_gemm_et's wave layout, 4 m x 2 n for items of up to 128 rows (the staged rows read from LDS by 2
// waves instead of 4: -10% at 5050 tokens), 2 m x 4 n for the 64-row items of small batches (where 4 m-waves would
// leave half of them without a 16-row tile). STRIX_MMH_ET_WM=2 / 4 forces one (dev A/B; bitwise either way)
static int mmh_et_wm(const int bm) {
    static const int wm = getenv("STRIX_MMH_ET_WM") ? atoi(getenv("STRIX_MMH_ET_WM")) : 0;
    return wm ? wm : bm == 128 ? 4 : 2;
}

bool ggml_cuda_mul_mat_had_supported(const int device, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * b = op->src[1];
    const int had = ggml_get_op_params_i32(op, 0);

    if (ggml_cuda_info().devices[device].warp_size != 32) {
        return false;
    }
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && !ggml_is_exl3(w->type)) {
        return false;
    }
    if (b->type != GGML_TYPE_F32 || b->nb[0] != sizeof(float) || op->type != GGML_TYPE_F32) {
        return false;
    }
    if ((had != 32 && had != 64 && had != 128) || w->ne[0] % MMH_NG != 0 || w->ne[1] % MMH_NG != 0) {
        return false;
    }
    if (w->nb[1] % 8 != 0 || w->nb[2] % 8 != 0 || w->ne[2] > MMH_MAX_EXPERTS || ggml_nrows(op) > INT_MAX/2) {
        return false;
    }
    // strixllama: per-expert widths: every class as the first, the experts counted by the class map
    if (op->src[5]) {
        if (op->src[5]->ne[0] > MMH_MAX_EXPERTS || !op->src[2]) {
            return false;
        }
        for (int c = 6; c < 8; ++c) {
            const ggml_tensor * wc = op->src[c];
            if (wc && (!ggml_is_exl3(wc->type) || wc->nb[1] % 8 != 0 || wc->nb[2] % 8 != 0)) {
                return false;
            }
        }
    }
    return true;
}

// src1 precision of the op: true for the F16 mode (GGML_PREC_F16 or lower), false for GGML_PREC_F32
static bool mmh_input_f16(const ggml_tensor * dst) {
    static const ggml_prec env = [] {
        const char * v = getenv("GGML_CUDA_MUL_MAT_HAD_PREC");
        if (v == nullptr) {
            return GGML_PREC_UNDEFINED;
        }
        const ggml_prec prec = strcmp(v, "f16") == 0 ? GGML_PREC_F16 : strcmp(v, "f32") == 0 ? GGML_PREC_F32 : GGML_PREC_UNDEFINED;
        if (prec == GGML_PREC_UNDEFINED) {
            GGML_LOG_WARN("%s: unknown GGML_CUDA_MUL_MAT_HAD_PREC = '%s', available: f32, f16\n", __func__, v);
        } else {
            GGML_LOG_INFO("%s: GGML_CUDA_MUL_MAT_HAD_PREC = %s\n", __func__, v);
        }
        return prec;
    }();
    // strixllama (dev): STRIX_MMH_F32_MATCH=a,b,... runs the ops whose weight name contains one of the parts with F32
    // inputs (which ops' fp16 rounding moves the KLD)
    static const std::vector<std::string> f32_match = [] {
        std::vector<std::string> v;
        const char * e = getenv("STRIX_MMH_F32_MATCH");
        for (std::string s = e ? e : ""; !s.empty(); ) {
            const size_t c = s.find(',');
            if (c != 0) v.push_back(s.substr(0, c));
            s = c == std::string::npos ? "" : s.substr(c + 1);
        }
        return v;
    }();
    if (!f32_match.empty() && dst->src[0]) {
        for (const std::string & m : f32_match) {
            if (strstr(dst->src[0]->name, m.c_str())) {
                return false;
            }
        }
    }
    ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    if (prec == GGML_PREC_UNDEFINED) {
        prec = env != GGML_PREC_UNDEFINED ? env : MMH_PREC_DEFAULT;
    }
    return prec >= GGML_PREC_F16;
}

// with WMMA, whether a dense op of `tokens` tokens runs the GEMV instead of the GEMM: the GEMV decodes every weight
// once for all its rows, but its cost per weight grows with the rows, while the GEMM tile costs about the same up to
// 16 rows; with few 128-row groups the GEMM also has few blocks (measured crossovers)
static bool mmh_dense_gemv(const int64_t tokens, const int64_t groups, const int64_t weights) {
    if (tokens <= 1) {
        return true;
    }
    if (tokens <= 2) {
        return groups <= 64;
    }
    if (tokens <= 3) {
        return groups <= 48;
    }
    if (tokens <= 4) {
        return groups <= 32;
    }
    return groups <= 16 || weights <= (1 << 21);
}

// k-split tickets of the fused GEMV, one buffer per device and stream: all zero between ops, as the last block of each
// group resets its ticket. A new buffer is cleared on a stream of its own and waited for, so that it can also be made
// while the op's stream is being captured into a CUDA graph (nullptr if n is too large).
static int * mmh_fused_tickets(const int device, cudaStream_t stream, const int64_t n) {
    constexpr int64_t cap = 16384;
    if (n > cap) {
        return nullptr;
    }
    static std::mutex mutex;
    static std::map<std::pair<int, cudaStream_t>, int *> buffers;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = buffers.find({device, stream});
    if (it != buffers.end()) {
        return it->second;
    }
    int * t = nullptr;
    cudaStream_t clear;
    CUDA_CHECK(cudaMalloc((void **) &t, cap*sizeof(int)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&clear, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemsetAsync(t, 0, cap*sizeof(int), clear));
    CUDA_CHECK(cudaStreamSynchronize(clear));
    CUDA_CHECK(cudaStreamDestroy(clear));
    buffers[{device, stream}] = t;
    return t;
}

// the inline GEMV; coal only with one row per pass (expert ops), where it measured faster
template <ggml_type type, int rows, bool h16>
static void mmh_launch_gemv(const bool fused, const bool coal, const dim3 grid, const int ksplit, const size_t lds,
        cudaStream_t stream, const mmh_params & p, const char * w, const void * xh, const float * xsum, const float * rscale,
        float * part, int * tickets, const int max_chunks) {
    if (coal && fused) {
        mmh_gemv<type, rows, true, true, h16><<<grid, 256*ksplit, lds, stream>>>(p, w, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    } else if (coal) {
        mmh_gemv<type, rows, false, true, h16><<<grid, 256*ksplit, 0, stream>>>(p, w, xh, xsum, rscale, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    } else if (fused) {
        mmh_gemv<type, rows, true, false, h16><<<grid, 256*ksplit, lds, stream>>>(p, w, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    } else {
        mmh_gemv<type, rows, false, false, h16><<<grid, 256*ksplit, 0, stream>>>(p, w, xh, xsum, rscale, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    }
}

// the inline GEMV with F32 or (h16) fp16 input
template <ggml_type type, bool h16>
static void mmh_launch_gemv_rows(const int rows_t, const bool fused, const bool coal, const dim3 grid, const int ksplit,
        const size_t lds, cudaStream_t stream, const mmh_params & p, const char * w, const void * xh, const float * xsum,
        const float * rscale, float * part, int * tickets, const int max_chunks) {
    switch (rows_t) {
        case 1:  mmh_launch_gemv<type, 1, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
        case 2:  mmh_launch_gemv<type, 2, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
        case 4:  mmh_launch_gemv<type, 4, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
        default: mmh_launch_gemv<type, 8, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
    }
}

// strixllama (dev): STRIX_MMH_TIMING=1 (graphs off) times the prologue, the routing pass and the GEMM of the WMMA path
struct mmh_timing {
    bool on = getenv("STRIX_MMH_TIMING") != nullptr;
    cudaEvent_t ev[4] = {};
    double sum[3][3] = {};
    int calls[3] = {};
    void mark(cudaStream_t st, int i) {
        if (!on) return;
        if (!ev[0]) { for (auto & e : ev) CUDA_CHECK(cudaEventCreateWithFlags(&e, 0)); }
        CUDA_CHECK(cudaEventRecord(ev[i], st));
    }
    void done(int kind) {
        if (!on) return;
        CUDA_CHECK(cudaEventSynchronize(ev[3]));
        for (int i = 0; i < 3; ++i) { float ms = 0; CUDA_CHECK(hipEventElapsedTime(&ms, ev[i], ev[i + 1])); sum[kind][i] += ms; }
        if (++calls[kind] % 12 == 0) {
            fprintf(stderr, "MMH_TIMING %s: %d calls, prologue %.1f ms, route %.1f ms, gemm %.1f ms\n", kind == 2 ? "ids k<=1024" : kind ? "ids" : "dense",
                    calls[kind], sum[kind][0], sum[kind][1], sum[kind][2]);
        }
    }
};
static mmh_timing g_mmh_timing;

// strixllama: per-expert widths: the routes and staged rows of an op's first class, reused by its other classes on the
// expert GEMM path (they do not depend on the weight's type); owned by mmh_run, so its pool memory outlives the
// launches and is freed after theirs
struct mmh_et_shared {
    bool ready = false;
    ggml_cuda_pool_alloc<int>   route;
    ggml_cuda_pool_alloc<half>  xh;
    ggml_cuda_pool_alloc<float> rscale;
    explicit mmh_et_shared(ggml_cuda_pool & pool) : route(pool), xh(pool), rscale(pool) {}
};

template <ggml_type type, bool f16>
static void mmh_launch(ggml_backend_cuda_context & ctx, const mmh_params & p, const char * w, const int64_t n,
        const bool inline_rows, mmh_et_shared * sh = nullptr) {
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t n_groups = n/MMH_NG;

    // the WMMA GEMM takes fp16 input: in the F32 mode the rows run the F32 GEMV
    const bool wmma = !inline_rows && f16 && type != GGML_TYPE_F32 && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));

    ggml_cuda_pool_alloc<half>  xh(ctx.pool());
    ggml_cuda_pool_alloc<float> rscale(ctx.pool());

    // rows per weight matrix, for the tile size
    const int64_t used    = p.ids ? std::min(p.n_rows, p.n_exp) : p.n_exp;
    const int64_t per_exp = (p.n_rows + used - 1)/used;

    if (inline_rows) {
        const int64_t n_items = p.ids ? p.n_rows : p.n_exp;
        // with ids, an expert of several tokens is decoded once for up to MMH_GEMV_IDS_ROWS of its rows per pass (an
        // expert of one token takes the one-row pass of the same kernel)
        const int     rows    = p.ids && p.ne2 > 1 ? (p.ne2 <= 2 ? MMH_GEMV_IDS_ROWS_2TOK : MMH_GEMV_IDS_ROWS) :
                                (int) std::min<int64_t>(MMH_GEMV_TOKENS, per_exp);

        // k split S = ksplit*kblocks: the power of 2 nearest (in log scale) to the split that gives about
        // MMH_GEMV_WAVES waves, with at least MMH_GEMV_MIN_KSLICES k slices per wave; S >= 2 splits k over 2 waves of
        // a block and S/2 blocks (small blocks fill the CUs better than blocks of 4 k splits)
        const int64_t kslices = p.k/16;
        // strixllama (dev): STRIX_MMH_GEMV_WAVES / STRIX_MMH_GEMV_MINK override the k split's targets
        static const int gemv_waves_env = getenv("STRIX_MMH_GEMV_WAVES") ? atoi(getenv("STRIX_MMH_GEMV_WAVES")) : 0;
        // strixllama (dev): STRIX_MMH_GEMV_WAVES_1ROW overrides the one-row dense target alone
        static const int gemv_waves_1row_env = getenv("STRIX_MMH_GEMV_WAVES_1ROW") ? atoi(getenv("STRIX_MMH_GEMV_WAVES_1ROW")) : 0;
        const int gemv_waves = !p.ids && p.n_rows == 1 && gemv_waves_1row_env ? gemv_waves_1row_env :
                               gemv_waves_env ? gemv_waves_env : !p.ids && p.n_rows == 1 ? MMH_GEMV_WAVES_1ROW : MMH_GEMV_WAVES;
        static const int gemv_mink  = getenv("STRIX_MMH_GEMV_MINK") ? atoi(getenv("STRIX_MMH_GEMV_MINK")) : MMH_GEMV_MIN_KSLICES;
        const double  want    = (double) gemv_waves/(8.0*(double) (n_groups*used));
        int split = 1;
        while (split < MMH_GEMV_MAX_SPLIT && want >= 1.41421356*split && kslices >= gemv_mink*2*split) {
            split *= 2;
        }
        // with ids and more than one token an expert can have several rows, which a block runs one after the other:
        // two waves per k range keep that chain short
        if (p.ids && p.ne2 > 1 && split < 2 && kslices >= 4*MMH_GEMV_MIN_KSLICES) {
            split = 2;
        }
        static const int ksplit_max = getenv("STRIX_MMH_KSPLIT_MAX") ? atoi(getenv("STRIX_MMH_KSPLIT_MAX")) : 2;
        const int ksplit  = split >= 4 && ksplit_max >= 4 ? 4 : split >= 2 ? 2 : 1;
        const int kblocks = split/ksplit;

        ggml_cuda_pool_alloc<float> part(ctx.pool());
        ggml_cuda_pool_alloc<int>   tickets(ctx.pool());
        const int max_chunks = (int) ((std::min<int64_t>(p.n_rows, MMH_GEMV_LIST) + rows - 1)/rows);
        const int n_tickets  = kblocks > 1 ? (int) (n_items*n_groups*max_chunks) : 0;
        const int rows_t     = rows <= 1 ? 1 : rows <= 2 ? 2 : rows <= 4 ? 4 : 8;

        // fused: the k slices one block stages (at most one 128-value block more at each end of its range), LDS of
        // the staged rows beside the static LDS of the GEMV
        const int64_t per    = (kslices + kblocks*ksplit - 1)/(kblocks*ksplit);
        const int64_t nks    = std::min<int64_t>(kslices, (per*ksplit + 7)/8*8 + 8);
        const size_t  lds    = (size_t) rows_t*nks*16*sizeof(float);
        const size_t  lds_static = (size_t) (2*MMH_GEMV_LIST + 33)*sizeof(int) + (size_t) 4*rows_t*MMH_NG*sizeof(float);
        bool fused = lds + lds_static <= MMH_FUSED_LDS && n_groups*used*kblocks <= MMH_FUSED_MAX_BLOCKS;
        // the fp16 input: 2 bytes per value, and for mul1 the F32 sums of each 16 values (the fused choice is the same).
        // Not for a dense op of one row, where it measured slower for the larger types
        const bool    h16    = f16 && mmh_is_exl3(type) && (p.ids || per_exp > 1) && kslices*sizeof(float) <= MMH_FUSED_LDS;
        const bool    sums   = h16 && mmh_exl3_codebook(type) == 0;
        const size_t  lds16  = (size_t) rows_t*nks*(16*sizeof(half) + (sums ? sizeof(float) : 0));
        int * fused_tickets = nullptr;
        if (fused && kblocks > 1) {
            fused_tickets = mmh_fused_tickets(ctx.device, stream, n_tickets);
            fused = fused_tickets != nullptr;
        }

        if (kblocks > 1) {
            part.alloc(p.n_rows*n_groups*kblocks*MMH_NG);
        }
        ggml_cuda_pool_alloc<float> xsum(ctx.pool());
        if (!fused) {
            xh.alloc((h16 ? 1 : 2)*p.n_rows*p.k);
            rscale.alloc(p.n_rows);
            if (sums) {
                xsum.alloc(p.n_rows*split);
            }
            if (kblocks > 1) {
                tickets.alloc(n_tickets);
            }
            if (h16) {
                mmh_prologue<true, true><<<p.n_rows, 256, sums ? kslices*sizeof(float) : 0, stream>>>(p, xh.get(), rscale.get(),
                    tickets.ptr, n_tickets, xsum.ptr, split, (int) per);
            } else {
                mmh_prologue<true, false><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), tickets.ptr, n_tickets, nullptr, 0, 0);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        int * tk = fused ? fused_tickets : tickets.ptr;

        const bool coal = p.ids != nullptr;
        // strixllama (dev): STRIX_MMH_PATHLOG=n prints the first n inline GEMV launch choices
        {
            static const int plog = getenv("STRIX_MMH_PATHLOG") ? atoi(getenv("STRIX_MMH_PATHLOG")) : 0;
            static int nlog = 0;
            if (plog && nlog < plog) {
                ++nlog;
                fprintf(stderr, "MMH_PATH k %lld n %lld rows %lld ids %d ne11 %d: fused %d h16 %d coal %d rows_t %d ksplit %d kblocks %d grid %lldx%lld lds %zu/%zu\n",
                    (long long) p.k, (long long) n, (long long) p.n_rows, p.ids != nullptr, p.ne11, (int) fused, (int) h16, (int) coal,
                    rows_t, ksplit, kblocks, (long long) n_groups, (long long) n_items, h16 ? lds16 : lds, lds_static);
            }
        }
        const dim3 grid(n_groups, n_items, kblocks);
        // strixllama (dev): STRIX_MMH_TRACE=n prints the block timeline of the first n inline GEMVs
        static const int trace_env = getenv("STRIX_MMH_TRACE") ? atoi(getenv("STRIX_MMH_TRACE")) : 0;
        static const int trace_skip = getenv("STRIX_MMH_TRACE_SKIP") ? atoi(getenv("STRIX_MMH_TRACE_SKIP")) : 0;
        static int traced = 0;
        if (trace_env && traced++ >= trace_skip && traced <= trace_skip + trace_env) {
            const size_t nbl = (size_t) grid.x*grid.y*grid.z;
            unsigned long long * tr = nullptr;
            CUDA_CHECK(cudaMalloc((void **) &tr, 3*nbl*sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemsetAsync(tr, 0, 3*nbl*sizeof(unsigned long long), stream));
            mmh_params pt = p;
            pt.trace = tr;
            if (h16) {
                if constexpr (f16 && mmh_is_exl3(type)) {
                    mmh_launch_gemv_rows<type, true>(rows_t, fused, coal, grid, ksplit, lds16, stream, pt, w, xh.ptr, xsum.ptr,
                        rscale.ptr, part.ptr, tk, max_chunks);
                }
            } else {
                mmh_launch_gemv_rows<type, false>(rows_t, fused, coal, grid, ksplit, lds, stream, pt, w, xh.ptr, nullptr, rscale.ptr,
                    part.ptr, tk, max_chunks);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<unsigned long long> h(3*nbl);
            CUDA_CHECK(cudaMemcpy(h.data(), tr, h.size()*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(tr));
            int rate_khz = 100000;
            (void) cudaDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, ctx.device);
            const double us = 1000.0/rate_khz;
            unsigned long long t0 = ~0ull, t1 = 0, s1 = 0;
            double dsum = 0, dmax = 0;
            int nrun = 0;
            std::map<int, int> per_cu;
            std::vector<double> starts;
            for (size_t i = 0; i < nbl; ++i) {
                if (!h[3*i] || (h[3*i + 2] >> 16)) continue;
                ++nrun;
                t0 = std::min(t0, h[3*i]); t1 = std::max(t1, h[3*i + 1]); s1 = std::max(s1, h[3*i]);
                const double d = (h[3*i + 1] - h[3*i])*us;
                dsum += d; dmax = std::max(dmax, d);
                per_cu[(int) (h[3*i + 2] & 0xFFFF)]++;
                starts.push_back((double) h[3*i]);
            }
            int maxcu = 0;
            for (auto & kv : per_cu) maxcu = std::max(maxcu, kv.second);
            fprintf(stderr, "MMH_TRACE k %lld n %lld rows %lld grid %ux%ux%u x %d waves: %d blocks ran, span %.1f us, starts over %.1f us, "
                "block mean %.1f max %.1f us, %zu CU ids, max %d blocks on one\n", (long long) p.k, (long long) n,
                (long long) p.n_rows, grid.x, grid.y, grid.z, 8*ksplit, nrun, (t1 - t0)*us, (s1 - t0)*us, nrun ? dsum/nrun : 0.0,
                dmax, per_cu.size(), maxcu);
            return;
        }
        if constexpr (f16 && mmh_is_exl3(type)) {
            if (h16) {
                mmh_launch_gemv_rows<type, true>(rows_t, fused, coal, grid, ksplit, lds16, stream, p, w, xh.ptr, xsum.ptr, rscale.ptr,
                    part.ptr, tk, max_chunks);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        mmh_launch_gemv_rows<type, false>(rows_t, fused, coal, grid, ksplit, lds, stream, p, w, xh.ptr, nullptr, rscale.ptr, part.ptr,
            tk, max_chunks);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // strixllama (dev): STRIX_MMH_PWAVE=0 runs the old prologues on the tiled and panel paths
    static const bool pwave_env = !getenv("STRIX_MMH_PWAVE") || atoi(getenv("STRIX_MMH_PWAVE")) != 0;
    // strixllama: dense ops of many rows decode the weight once to fp16 and run the tiled GEMM (mmh_gemm_t)
    if constexpr (mmh_is_exl3(type)) {
        static const bool tiled_env = !getenv("STRIX_MMH_TILED") || atoi(getenv("STRIX_MMH_TILED")) != 0;
        // strixllama (dev): STRIX_MMH_T_NOSPLIT=1 leaves the ops whose GEMM below would split k to it (bitwise A/B)
        static const bool nosplit_env = getenv("STRIX_MMH_T_NOSPLIT") && atoi(getenv("STRIX_MMH_T_NOSPLIT")) != 0;
        const bool split_below = n_groups*((p.n_rows + 127)/128) < 256 && p.k/32 >= 32;
        if (wmma && GGML_CUDA_CC_IS_RDNA3(cc) && tiled_env && !p.ids && p.n_exp == 1 && p.n_rows >= MMH_T_MIN_ROWS &&
                n*p.k*(int64_t) sizeof(half) <= MMH_T_MAX_WBYTES && p.k % 128 == 0 && !(nosplit_env && split_below)) {
            const int64_t n_rb = (p.n_rows + 127)/128;
            ggml_cuda_pool_alloc<half> wd(ctx.pool(), n*p.k);
            xh.alloc(n_rb*128*p.k);
            rscale.alloc(p.n_rows);
            g_mmh_timing.mark(stream, 0);
            if (pwave_env) {
                if (mmh_pwave4_ok(p)) {
                    mmh_prologue_wave4<true><<<(unsigned) ((p.n_rows + 7)/8), 256, 0, stream>>>(p, xh.get(), rscale.get());
                } else {
                    mmh_prologue_wave<true><<<(unsigned) ((p.n_rows + 7)/8), 256, 0, stream>>>(p, xh.get(), rscale.get());
                }
            } else {
                mmh_prologue<false, true, true><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), nullptr, 0, nullptr, 0, 0);
            }
            CUDA_CHECK(cudaGetLastError());
            g_mmh_timing.mark(stream, 1);
            mmh_decode_t<type><<<dim3((unsigned) n_groups, (unsigned) ((p.k/32 + 7)/8)), 256, 0, stream>>>(p, w, wd.get());
            CUDA_CHECK(cudaGetLastError());
            g_mmh_timing.mark(stream, 2);
            // strixllama (dev): STRIX_MMH_GT2=1 runs the chunked tiled GEMM (A/B; bitwise)
            static const bool gt2_env = !getenv("STRIX_MMH_GT2") || atoi(getenv("STRIX_MMH_GT2")) != 0;   // default on
            if (gt2_env) {
                mmh_gemm_t2<<<(unsigned) (n_rb*n_groups), 256, 0, stream>>>(p, wd.get(), xh.get(), rscale.get(), (int) n_rb, (int) n_groups);
            } else {
                mmh_gemm_t<<<(unsigned) (n_rb*n_groups), 256, 0, stream>>>(p, wd.get(), xh.get(), rscale.get(), (int) n_rb, (int) n_groups);
            }
            CUDA_CHECK(cudaGetLastError());
            g_mmh_timing.mark(stream, 3);
            g_mmh_timing.done(0);
            return;
        }
    }

    // strixllama: ops with ids route first and stage their rows in sorted panels for mmh_gemm_et
    if constexpr (mmh_is_exl3(type)) {
        static const bool et_env = !getenv("STRIX_MMH_ET") || atoi(getenv("STRIX_MMH_ET")) != 0;
        // strixllama (dev): STRIX_MMH_ET2=1 runs mmh_gemm_et2 for the 128-row items (A/B; bitwise)
        static const int et2_env = getenv("STRIX_MMH_ET2") ? atoi(getenv("STRIX_MMH_ET2")) : 1;   // default: et2
        if (wmma && GGML_CUDA_CC_IS_RDNA3(cc) && et_env && p.ids && p.k % 128 == 0) {
            const int mt_et = per_exp > 64 ? 128 : 64;
            const int64_t max_items = used + p.n_rows/mt_et + 1;
            GGML_ASSERT(max_items <= 65535);
            // strixllama: per-expert widths: the first class routes and stages, the others reuse it (sh->ready)
            ggml_cuda_pool_alloc<int>   route_own(ctx.pool());
            ggml_cuda_pool_alloc<int>   & route = sh ? sh->route : route_own;
            ggml_cuda_pool_alloc<half>  & xhs   = sh ? sh->xh : xh;
            ggml_cuda_pool_alloc<float> & rss   = sh ? sh->rscale : rscale;
            const bool reuse = sh && sh->ready;
            if (!reuse) {
                route.alloc(3*p.n_rows + 3*max_items + 1);
            }
            int * eid         = route.get();
            int * pos         = eid + p.n_rows;
            int * rows_sorted = pos + p.n_rows;
            int * items       = rows_sorted + p.n_rows;
            int * n_items     = items + 3*max_items;
            g_mmh_timing.mark(stream, 0);
            if (!reuse) {
                mmh_route<<<1, 1024, 0, stream>>>(p, mt_et, eid, pos, rows_sorted, items, n_items);
                CUDA_CHECK(cudaGetLastError());
            }
            g_mmh_timing.mark(stream, 1);
            // strixllama (dev): STRIX_MMH_ROUTE_STATS=1 prints the rows per expert of the large ops with ids
            static const bool stats_env = getenv("STRIX_MMH_ROUTE_STATS") != nullptr;
            if (stats_env && !reuse && p.n_rows >= 4096 && p.k > 1024) {
                std::vector<int> h_items(3*max_items);
                int h_n = 0;
                CUDA_CHECK(cudaMemcpyAsync(&h_n, n_items, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h_items.data(), items, h_items.size()*sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::map<int, int> per_e;
                for (int i = 0; i < h_n; ++i) {
                    per_e[h_items[3*i]] += h_items[3*i + 2];
                }
                std::vector<int> c;
                for (auto & kv : per_e) {
                    c.push_back(kv.second);
                }
                std::sort(c.begin(), c.end());
                int64_t rows16 = 0;
                for (int i = 0; i < h_n; ++i) {
                    rows16 += (h_items[3*i + 2] + 15)/16*16;
                }
                fprintf(stderr, "MMH_ROUTE rows %lld experts %zu items %d (mt %d) rows16 %lld | min %d p10 %d p50 %d p90 %d p99 %d max %d\n",
                    (long long) p.n_rows, c.size(), h_n, mt_et, (long long) rows16, c.front(), c[c.size()/10], c[c.size()/2],
                    c[c.size()*9/10], c[c.size()*99/100], c.back());
            }

            mmh_params pp = p;
            pp.pos = pos;
            // strixllama (dev): STRIX_MMH_SHARED_ROT=1 also takes expert 0's rot_in for all (wrong values: the speed of a
            // file whose experts share their input rotation)
            // a rot_in shared by all experts (rin_nb1 0) with one input row per token: one staged row per token
            static const bool shared_env = getenv("STRIX_MMH_SHARED_ROT") && atoi(getenv("STRIX_MMH_SHARED_ROT")) != 0;
            if ((shared_env || p.rin_nb1 == 0) && p.ne11 == 1) {
                mmh_params pt = p;
                pt.ids = nullptr;
                pt.n_rows = p.ne2;
                pt.ne1 = 1;
                pt.bcast = 1 << 30;
                pt.panel_rows = 1;
                if (!reuse) {
                    xhs.alloc(((p.ne2 + 128)/16 + 1)*16*p.k);
                    rss.alloc(p.ne2);
                    if (mmh_pwave4_ok(pt)) {
                        mmh_prologue_wave4<false><<<(unsigned) ((p.ne2 + 7)/8), 256, 0, stream>>>(pt, xhs.get(), rss.get());
                    } else {
                        mmh_prologue_wave<false><<<(unsigned) ((p.ne2 + 7)/8), 256, 0, stream>>>(pt, xhs.get(), rss.get());
                    }
                    CUDA_CHECK(cudaGetLastError());
                }
                if (sh) {
                    sh->ready = true;
                }
                pp.pos = nullptr;
                pp.tok_rows = p.ne1;
                g_mmh_timing.mark(stream, 2);
                if (mt_et == 128 && et2_env == 2) {
                    mmh_gemm_et3<type><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                        rss.get(), items, n_items, rows_sorted);
                } else if (mt_et == 128 && et2_env) {
                    mmh_gemm_et2<type><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                        rss.get(), items, n_items, rows_sorted);
                } else if (mt_et == 128) {
                    if (mmh_et_wm(128) == 2) {
                        mmh_gemm_et<type, 128, 2><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                            rss.get(), items, n_items, rows_sorted);
                    } else {
                        mmh_gemm_et<type, 128, 4><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                            rss.get(), items, n_items, rows_sorted);
                    }
                } else {
                    if (mmh_et_wm(64) == 2) {
                        mmh_gemm_et<type, 64, 2><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                            rss.get(), items, n_items, rows_sorted);
                    } else {
                        mmh_gemm_et<type, 64, 4><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                            rss.get(), items, n_items, rows_sorted);
                    }
                }
                CUDA_CHECK(cudaGetLastError());
                g_mmh_timing.mark(stream, 3);
                g_mmh_timing.done(p.k <= 1024 ? 2 : 1);
                return;
            }
            // the panels of all rows and of the 128 rows a block of the last item reads past them
            if (!reuse) {
                xhs.alloc(((p.n_rows + 128)/16 + 1)*16*p.k);
                rss.alloc(p.n_rows);
                if (pwave_env) {
                    if (mmh_pwave4_ok(pp)) {
                        mmh_prologue_wave4<false><<<(unsigned) ((p.n_rows + 7)/8), 256, 0, stream>>>(pp, xhs.get(), rss.get());
                    } else {
                        mmh_prologue_wave<false><<<(unsigned) ((p.n_rows + 7)/8), 256, 0, stream>>>(pp, xhs.get(), rss.get());
                    }
                } else if (p.ne11 == 1 && p.ne1 <= 16 && p.k <= 4096) {
                    mmh_prologue_tok<<<p.ne2, 256, 0, stream>>>(pp, xhs.get(), rss.get());
                } else {
                    mmh_prologue<false, true><<<p.n_rows, 256, 0, stream>>>(pp, xhs.get(), rss.get(), nullptr, 0, nullptr, 0, 0);
                }
                CUDA_CHECK(cudaGetLastError());
            }
            if (sh) {
                sh->ready = true;
            }
            g_mmh_timing.mark(stream, 2);
            if (mt_et == 128 && et2_env == 2) {
                mmh_gemm_et3<type><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                    rss.get(), items, n_items, rows_sorted);
            } else if (mt_et == 128 && et2_env) {
                mmh_gemm_et2<type><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                    rss.get(), items, n_items, rows_sorted);
            } else if (mt_et == 128) {
                if (mmh_et_wm(128) == 2) {
                    mmh_gemm_et<type, 128, 2><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                        rss.get(), items, n_items, rows_sorted);
                } else {
                    mmh_gemm_et<type, 128, 4><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                        rss.get(), items, n_items, rows_sorted);
                }
            } else {
                if (mmh_et_wm(64) == 2) {
                    mmh_gemm_et<type, 64, 2><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                        rss.get(), items, n_items, rows_sorted);
                } else {
                    mmh_gemm_et<type, 64, 4><<<dim3((unsigned) n_groups, (unsigned) max_items), 256, 0, stream>>>(pp, w, xhs.get(),
                        rss.get(), items, n_items, rows_sorted);
                }
            }
            CUDA_CHECK(cudaGetLastError());
            g_mmh_timing.mark(stream, 3);
            g_mmh_timing.done(p.k <= 1024 ? 2 : 1);
            return;
        }
    }

    xh.alloc((wmma ? 1 : 2)*p.n_rows*p.k);
    rscale.alloc(p.n_rows);

    const int mt = !wmma ? 8 : per_exp <= 16 ? 16 : per_exp <= 32 ? 32 : per_exp <= 64 ? 64 : 128;
    // strixllama (dev): STRIX_MMH_MT forces the GEMM's rows per block for ops with ids
    static const int mt_env = getenv("STRIX_MMH_MT") ? atoi(getenv("STRIX_MMH_MT")) : 0;
    static const bool wide_env = !getenv("STRIX_MMH_WIDE") || atoi(getenv("STRIX_MMH_WIDE")) != 0;
    static const int wide_dense = getenv("STRIX_MMH_WIDE_DENSE") ? atoi(getenv("STRIX_MMH_WIDE_DENSE")) : 0;
    const bool wide = wmma && (p.ids || wide_dense) && wide_env && type != GGML_TYPE_F32 && mmh_exl3_codebook(type) == 0 && mt_env == 0;
    const int mt_forced = wide ? MMH_WIDE_MT : wmma && p.ids && (mt_env == 16 || mt_env == 32 || mt_env == 64 || mt_env == 128) ? mt_env : mt;
    const int64_t max_items = used + p.n_rows/mt_forced + 1;

    GGML_ASSERT(max_items <= 65535);

    // WMMA: blocks also split k while there are few blocks and each keeps >= 8 steps
    const int64_t items_est = p.ids ? used + p.n_rows/mt_forced : p.n_exp*((per_exp + mt_forced - 1)/mt_forced);
    int kblocks = 1;
    while (wmma && kblocks < 8 && n_groups*items_est*kblocks < 256 && p.k/32 >= 8*kblocks*2) {
        kblocks *= 2;
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    ggml_cuda_pool_alloc<int>   tickets(ctx.pool());
    const int n_tickets = kblocks > 1 ? (int) (max_items*n_groups) : 0;
    if (kblocks > 1) {
        part.alloc(p.n_rows*n_groups*kblocks*MMH_NG);
        tickets.alloc(n_tickets);
    }

    g_mmh_timing.mark(stream, 0);
    static const bool tok_env = !getenv("STRIX_MMH_PROLOGUE_TOK") || atoi(getenv("STRIX_MMH_PROLOGUE_TOK")) != 0;
    if (wmma && tok_env && n_tickets == 0 && p.ids && p.ne11 == 1 && p.ne1 <= 16 && p.k % 128 == 0 && p.k <= 4096) {
        mmh_prologue_tok<<<p.ne2, 256, 0, stream>>>(p, xh.get(), rscale.get());
    } else if (wmma) {
        mmh_prologue<false, true><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), tickets.ptr, n_tickets, nullptr, 0, 0);
    } else {
        mmh_prologue<true, false><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), nullptr, 0, nullptr, 0, 0);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int> route(ctx.pool(), 3*p.n_rows + 3*max_items + 1);
    int * eid         = route.get();
    int * rank        = eid + p.n_rows;
    int * rows_sorted = rank + p.n_rows;
    int * items       = rows_sorted + p.n_rows;
    int * n_items     = items + 3*max_items;

    g_mmh_timing.mark(stream, 1);
    mmh_route<<<1, 1024, 0, stream>>>(p, mt_forced, eid, rank, rows_sorted, items, n_items);
    CUDA_CHECK(cudaGetLastError());
    g_mmh_timing.mark(stream, 2);

    const dim3 grid(n_groups, max_items, kblocks);
    // strixllama: ops with ids read their staged rows once per chunk of MMH_WIDE_G groups (mmh_gemm_wide)
    if constexpr (type != GGML_TYPE_F32 && mmh_exl3_codebook(type) == 0) {
        if (wide && kblocks == 1) {
            const dim3 grid_w((unsigned) ((n_groups + MMH_WIDE_G - 1)/MMH_WIDE_G), (unsigned) max_items, 1);
            mmh_gemm_wide<type, MMH_WIDE_MT, MMH_WIDE_G><<<grid_w, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items,
                rows_sorted, (int) n_groups);
            CUDA_CHECK(cudaGetLastError());
            g_mmh_timing.mark(stream, 3);
            g_mmh_timing.done(p.ids ? 1 : 0);
            return;
        }
    }
    if (wmma) {
        if constexpr (type != GGML_TYPE_F32) {
            if (mt_forced == 16) {
                mmh_gemm<type, 16><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            } else if (mt_forced == 32) {
                mmh_gemm<type, 32><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            } else if (mt_forced == 64) {
                mmh_gemm<type, 64><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            } else {
                mmh_gemm<type, 128><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            }
        }
    } else {
        // the waves split k also for precision: a lane then sums at most a quarter of k
        const int ksplit = p.k/16 >= 64 ? 4 : p.k/16 >= 32 ? 2 : 1;
        mmh_gemv<type, 8, false, false><<<grid, 256*ksplit, 0, stream>>>(p, w, xh.get(), nullptr, rscale.get(), items, n_items, rows_sorted, ksplit, nullptr, nullptr, 1);
    }
    CUDA_CHECK(cudaGetLastError());
    g_mmh_timing.mark(stream, 3);
    g_mmh_timing.done(p.ids ? 1 : 0);
}

template <ggml_type type>
static void mmh_launch_prec(ggml_backend_cuda_context & ctx, const mmh_params & p, const char * w, const int64_t n,
        const bool inline_rows, const bool f16, mmh_et_shared * sh = nullptr) {
    if (f16) {
        mmh_launch<type, true>(ctx, p, w, n, inline_rows, sh);
    } else {
        mmh_launch<type, false>(ctx, p, w, n, inline_rows, sh);
    }
}

// the weight, rotation and scale tensors are read where they are (VRAM, or a pinned host buffer for host experts)
static const char * mmh_data(const ggml_tensor * t) {
    return (const char *) t->data;
}

// the launch for a weight type
static void mmh_dispatch(ggml_backend_cuda_context & ctx, const ggml_type wtype, const mmh_params & p, const char * wd,
        const int64_t n, const bool inline_rows, const bool f16, mmh_et_shared * sh = nullptr) {
    switch (wtype) {
        case GGML_TYPE_F32:      mmh_launch<GGML_TYPE_F32, false>(ctx, p, wd, n, inline_rows); break;
        case GGML_TYPE_F16:      mmh_launch_prec<GGML_TYPE_F16>     (ctx, p, wd, n, inline_rows, f16, sh); break;
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M1:  mmh_launch_prec<GGML_TYPE_EXL3_M1> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M2:  mmh_launch_prec<GGML_TYPE_EXL3_M2> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M3:  mmh_launch_prec<GGML_TYPE_EXL3_M3> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
        case GGML_TYPE_EXL3_M4:  mmh_launch_prec<GGML_TYPE_EXL3_M4> (ctx, p, wd, n, inline_rows, f16, sh); break;
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M5:  mmh_launch_prec<GGML_TYPE_EXL3_M5> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
        case GGML_TYPE_EXL3_M6:  mmh_launch_prec<GGML_TYPE_EXL3_M6> (ctx, p, wd, n, inline_rows, f16, sh); break;
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M7:  mmh_launch_prec<GGML_TYPE_EXL3_M7> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M8:  mmh_launch_prec<GGML_TYPE_EXL3_M8> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M1H: mmh_launch_prec<GGML_TYPE_EXL3_M1H>(ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M2H: mmh_launch_prec<GGML_TYPE_EXL3_M2H>(ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_M3H: mmh_launch_prec<GGML_TYPE_EXL3_M3H>(ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G1:  mmh_launch_prec<GGML_TYPE_EXL3_G1> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G2:  mmh_launch_prec<GGML_TYPE_EXL3_G2> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G3:  mmh_launch_prec<GGML_TYPE_EXL3_G3> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G4:  mmh_launch_prec<GGML_TYPE_EXL3_G4> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G5:  mmh_launch_prec<GGML_TYPE_EXL3_G5> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G6:  mmh_launch_prec<GGML_TYPE_EXL3_G6> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G7:  mmh_launch_prec<GGML_TYPE_EXL3_G7> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_G8:  mmh_launch_prec<GGML_TYPE_EXL3_G8> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T1:  mmh_launch_prec<GGML_TYPE_EXL3_T1> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T2:  mmh_launch_prec<GGML_TYPE_EXL3_T2> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T3:  mmh_launch_prec<GGML_TYPE_EXL3_T3> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T4:  mmh_launch_prec<GGML_TYPE_EXL3_T4> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T5:  mmh_launch_prec<GGML_TYPE_EXL3_T5> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T6:  mmh_launch_prec<GGML_TYPE_EXL3_T6> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T7:  mmh_launch_prec<GGML_TYPE_EXL3_T7> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
#if !defined(MMH_DEV_FAST)
        case GGML_TYPE_EXL3_T8:  mmh_launch_prec<GGML_TYPE_EXL3_T8> (ctx, p, wd, n, inline_rows, f16, sh); break;
#endif
        default:
            GGML_ABORT("unsupported MUL_MAT_HAD weight type %s", ggml_type_name(wtype));
    }
}

static void mmh_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * glu_dst, const ggml_tensor * glu_other,
        const int glu, const ggml_tensor * row_mul = nullptr) {
    const ggml_tensor * w       = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * rot_in  = dst->src[3];
    const ggml_tensor * rot_out = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float));
    GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);

    const int had = ggml_get_op_params_i32(dst, 0);

    mmh_params p;
    p.k        = w->ne[0];
    p.n_exp    = w->ne[2];
    p.nb01     = w->nb[1];
    p.nb02     = w->nb[2];
    p.n_rows   = ggml_nrows(dst);
    p.ne1      = (int) dst->ne[1];
    p.ne2      = (int) dst->ne[2];
    p.nb1      = dst->nb[1];
    p.nb2      = dst->nb[2];
    p.nb3      = dst->nb[3];
    p.ne11     = (int) x->ne[1];
    p.nb11     = x->nb[1];
    p.nb12     = x->nb[2];
    p.nb13     = x->nb[3];
    p.bcast    = ids ? 1 : (int) (x->ne[2]/w->ne[2]);
    p.ids      = ids ? (const int32_t *) ids->data : nullptr;
    p.ids_nb0  = ids ? ids->nb[0] : 0;
    p.ids_nb1  = ids ? ids->nb[1] : 0;
    p.x        = (const float *) x->data;
    p.rot_in   = mmh_data(rot_in);
    p.rin_nb1  = rot_in->ne[1] == 1 ? 0 : rot_in->nb[1];     // [K, 1]: shared by all matrices
    p.rot_out  = mmh_data(rot_out);
    p.rout_nb0 = rot_out->nb[0];
    p.rout_nb1 = rot_out->ne[1] == 1 ? 0 : rot_out->nb[1];
    p.dst      = (float *) dst->data;
    p.rin_f16  = rot_in->type == GGML_TYPE_F16;
    p.rout_f16 = rot_out->type == GGML_TYPE_F16;
    p.had      = had;
    p.hscale   = 1.0f/sqrtf((float) had);
    p.glu_other = glu_other ? (const float *) glu_other->data : nullptr;
    p.pos       = nullptr;
    p.panel_rows = 0;
    p.tok_rows   = 0;
    p.trace      = nullptr;
    p.cmap       = nullptr;
    p.cls        = 0;
    p.glu       = glu;
    if (glu) {
        // the GLU output in place of dst: the same shape and strides
        GGML_ASSERT(glu_dst->nb[1] == dst->nb[1] && glu_dst->nb[2] == dst->nb[2] && glu_dst->nb[3] == dst->nb[3]);
        GGML_ASSERT(glu_other->nb[1] == dst->nb[1] && glu_other->nb[2] == dst->nb[2] && glu_other->nb[3] == dst->nb[3]);
        p.dst = (float *) glu_dst->data;
    }
    p.row_w     = nullptr;
    p.row_w_nb1 = 0;
    p.row_w_nb2 = 0;
    if (row_mul) {
        // the MUL's output in place of dst (the same shape and strides), its weights [1, ne1, ne2]
        const ggml_tensor * wts = row_mul->src[1];
        GGML_ASSERT(row_mul->nb[1] == dst->nb[1] && row_mul->nb[2] == dst->nb[2] && row_mul->nb[3] == dst->nb[3]);
        GGML_ASSERT(wts->type == GGML_TYPE_F32 && wts->ne[0] == 1 && wts->ne[1] == dst->ne[1] && wts->ne[2] == dst->ne[2]);
        p.dst       = (float *) row_mul->data;
        p.row_w     = (const float *) wts->data;
        p.row_w_nb1 = wts->nb[1];
        p.row_w_nb2 = wts->nb[2];
    }

    if (p.n_rows == 0) {
        return;
    }

    const int64_t n = w->ne[1];

    // strixllama (dev): STRIX_MMH_SKIP=mask leaves out the decode-size (<= 16 tokens) products of these kinds - wrong
    // outputs; the generation speed with and without them is what they cost in the real step (graphs on):
    // 16 routed experts' gate/up, 32 their down, 2 shared expert, 4 output head, 8 the other dense weights
    {
        static const int skip = getenv("STRIX_MMH_SKIP") ? atoi(getenv("STRIX_MMH_SKIP")) : 0;
        if (skip) {
            const int64_t toks = ids ? dst->ne[2] : p.n_rows;
            const int kind = ids ? (p.k > 1024 ? 16 : 32) : (n == 640 || p.k == 640) ? 2 : n > 100000 ? 4 : 8;
            static const int skip_lo = getenv("STRIX_MMH_SKIP_LAYERS") ? atoi(getenv("STRIX_MMH_SKIP_LAYERS")) : 0;
            static const int skip_hi = getenv("STRIX_MMH_SKIP_LAYERS") && strchr(getenv("STRIX_MMH_SKIP_LAYERS"), '-') ?
                atoi(strchr(getenv("STRIX_MMH_SKIP_LAYERS"), '-') + 1) : 1 << 20;
            int layer = -1;
            sscanf(w->name, "blk.%d.", &layer);
            if ((skip & kind) && toks <= 16 && (layer < 0 || (layer >= skip_lo && layer <= skip_hi))) {
                return;
            }
        }
    }

    // without a routing pass: few tokens (ids) or few rows per weight matrix; with WMMA a dense matrix of more than
    // one row goes to the GEMM, which stages the rows once per block, unless the GEMV is faster (mmh_dense_gemv)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool wmma = w->type != GGML_TYPE_F32 && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
    const int64_t tokens = ids ? dst->ne[2] : (p.n_rows + p.n_exp - 1)/p.n_exp;
    const bool inline_rows = p.n_rows <= MMH_GEMV_LIST && tokens <= MMH_GEMV_TOKENS &&
        (ids || !wmma || mmh_dense_gemv(tokens, n/MMH_NG, p.k*n));

    // F32 weights run only the GEMV, which has no F16 mode
    const bool f16 = w->type != GGML_TYPE_F32 && mmh_input_f16(dst);

    // strixllama: per-expert widths: one launch per class with its weight
    const ggml_tensor * cmap = dst->src[5];
    if (cmap) {
        GGML_ASSERT(ids && cmap->type == GGML_TYPE_I32);
        p.cmap  = (const int32_t *) cmap->data;
        p.n_exp = cmap->ne[0];
        mmh_et_shared sh(ctx.pool());
        for (int c = 0; c < 3; ++c) {
            const ggml_tensor * wc = c == 0 ? w : dst->src[5 + c];
            if (wc == nullptr || wc->ne[2] == 0) {
                continue;
            }
            mmh_params pc = p;
            pc.cls  = c;
            pc.nb01 = wc->nb[1];
            pc.nb02 = wc->nb[2];
            mmh_dispatch(ctx, wc->type, pc, mmh_data(wc), n, inline_rows, f16, &sh);
        }
        return;
    }
    mmh_dispatch(ctx, w->type, p, mmh_data(w), n, inline_rows, f16);
}

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    mmh_run(ctx, dst, nullptr, nullptr, 0);
}

void ggml_cuda_mul_mat_had_glu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * other,
        const ggml_tensor * glu, const bool dst_is_gate) {
    mmh_run(ctx, dst, glu, other, dst_is_gate ? 1 : 2);
}

void ggml_cuda_mul_mat_had_rowmul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * row_mul) {
    mmh_run(ctx, dst, nullptr, nullptr, 0, row_mul);
}
