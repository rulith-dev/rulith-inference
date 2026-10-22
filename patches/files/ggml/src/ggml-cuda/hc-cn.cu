#include "hc-cn.cuh"
#include <cstdlib>

#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    return __fadd_rn(a, b);
}
#endif

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float hc_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

#define HC_CN_BLOCK 1024
#define HC_CN_MAX_EMB 3072

__device__ __forceinline__ float hc_bf2f32(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
__device__ __forceinline__ uint16_t hc_f2bf32(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_f32(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32, const int64_t xn_ld,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ float s_sum[32];

    const int c   = blockIdx.x;   // stream
    const int t   = blockIdx.y;   // token
    const int hc  = gridDim.x;
    const int tid = threadIdx.x;

    // identical expressions to scale_f32 / op_sigmoid / scale_f32
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;

    const int64_t row = (int64_t) t * hc + c;
    const float *    res   = residual + row * n_embd;
    const uint16_t * res16  = res_in_bf16  ? res_in_bf16  + row * n_embd : nullptr;
    float *          dst    = out_res      + row * n_embd;
    uint16_t *       dst16  = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    block_out += (int64_t) t * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (int64_t) t * n_embd : nullptr;
    float xs[3];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        xs[k] = 0.0f;
        if (col < n_embd) {
            const float m  = hc_mul_rn(blk16 ? hc_bf2f32(blk16[col]) : block_out[col], w);
            const float xi = hc_add_rn(res16 ? hc_bf2f32(res16[col]) : res[col], m);
            if (dst16) dst16[col] = hc_f2bf32(xi); else dst[col] = xi;
            xs[k]    = xi;
            tmp += xi * xi;
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK>(tmp, s_sum);

    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + (int64_t) t * xn_ld + (int64_t) c * n_embd : nullptr;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        if (col < n_embd) {
            const float v = scale * xs[k] * g[col];
            if (store_xn_f32) xn[col] = v;
            if (xh) { uint32_t u = __float_as_uint(v); u += 0x7fffu + ((u >> 16) & 1u); xh[col] = (uint16_t)(u >> 16); }
        }
    }
}

#define HC_CN_BLOCK2 256
__device__ __forceinline__ float hc_lo(const uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hc_hi(const uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t hc_pack2(const float a, const float b) {
    return (uint32_t) hc_f2bf32(a) | ((uint32_t) hc_f2bf32(b) << 16);
}

// two elements per thread per iteration, packed 32-bit accesses (Halogen's k_hc_scatter_norm layout)
// strixllama: INJ - also the next hc_inject's dot products (hc == 4) over this stream's slice of xn, written as
// inj_part[(t * hc + c) * 4 + m]; hc_inject_reduce sums the streams. Saves the inject's own pass over the F32 xn.
// strixllama: NF > 0 - n_embd is NF * 2 * HC_CN_BLOCK2 (2560: 5), so with F32 streams every iteration is whole and the
// loops run without per-iteration guards: all of a thread's loads go out before its first arithmetic. Guarded, each
// iteration compiled to a load, a wait and the arithmetic, and at one token (four workgroups on the whole GPU) the kernel
// paid two rounds of five memory round trips. The arithmetic and its order are the guarded loops' own
template <bool INJ, int NF = 0>
static __global__ void __launch_bounds__(HC_CN_BLOCK2, 4) hc_combine_norm_f32_b256(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32, const int64_t xn_ld,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps,
        const float * __restrict__ inj_w, float * __restrict__ inj_part, float * __restrict__ row_scale) {
    __shared__ float s_sum[32];
    const int c = blockIdx.x, t = blockIdx.y, hc = gridDim.x, tid = threadIdx.x;
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;
    const int64_t row = (int64_t) t * hc + c;
    const float *    res   = residual + row * n_embd;
    const uint16_t * res16 = res_in_bf16  ? res_in_bf16  + row * n_embd : nullptr;
    float *          dst   = out_res      + row * n_embd;
    uint16_t *       dst16 = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    const float *    blk   = block_out + (int64_t) t * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (int64_t) t * n_embd : nullptr;
    constexpr int KP = (HC_CN_MAX_EMB / 2 + HC_CN_BLOCK2 - 1) / HC_CN_BLOCK2;
    float xs[2 * KP];
    float tmp = 0.0f;
    const bool whole = NF > 0 && !res16 && !blk16 && !dst16;
    if constexpr (NF > 0) { if (whole) {
    float2 rv[NF], mv[NF];
#pragma unroll
    for (int k = 0; k < NF; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        rv[k] = *(const float2 *)(res + col);
        mv[k] = *(const float2 *)(blk + col);
    }
#pragma unroll
    for (int k = 0; k < NF; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        const float r0 = rv[k].x, r1 = rv[k].y, m0 = mv[k].x, m1 = mv[k].y;
        const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
        const float a1 = hc_add_rn(r1, hc_mul_rn(m1, w));
        *(float2 *)(dst + col) = make_float2(a0, a1);
        xs[2 * k] = a0; xs[2 * k + 1] = a1;
        tmp += a0 * a0 + a1 * a1;
    }
    } }
    if (!whole)
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        xs[2 * k] = 0.0f; xs[2 * k + 1] = 0.0f;
        if (col + 1 < n_embd) {
            float r0, r1, m0, m1;
            if (res16) { const uint32_t u = *(const uint32_t *)(res16 + col); r0 = hc_lo(u); r1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(res + col);     r0 = v.x;      r1 = v.y; }
            if (blk16) { const uint32_t u = *(const uint32_t *)(blk16 + col); m0 = hc_lo(u); m1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(blk + col);     m0 = v.x;      m1 = v.y; }
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            const float a1 = hc_add_rn(r1, hc_mul_rn(m1, w));
            if (dst16) *(uint32_t *)(dst16 + col) = hc_pack2(a0, a1);
            else       *(float2 *)(dst + col)     = make_float2(a0, a1);
            xs[2 * k] = a0; xs[2 * k + 1] = a1;
            tmp += a0 * a0 + a1 * a1;
        } else if (col < n_embd) {
            const float r0 = res16 ? hc_bf2f32(res16[col]) : res[col];
            const float m0 = blk16 ? hc_bf2f32(blk16[col]) : blk[col];
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            if (dst16) dst16[col] = hc_f2bf32(a0); else dst[col] = a0;
            xs[2 * k] = a0;
            tmp += a0 * a0;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK2>(tmp, s_sum);
    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);
    if (row_scale && tid == 0) {
        row_scale[row] = scale;
    }
    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + (int64_t) t * xn_ld + (int64_t) c * n_embd : nullptr;
    // the inject weights of this stream's slice: row m of w_inject [hc * n_embd, 4] is contiguous
    const int64_t wk = (int64_t) hc * n_embd;
    const float * wi = INJ ? inj_w + (int64_t) c * n_embd : nullptr;
    float pa[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (NF > 0) { if (whole) {
    float2 gvs[NF], wvs[INJ ? 4 : 1][NF];
#pragma unroll
    for (int k = 0; k < NF; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        gvs[k] = *(const float2 *)(g + col);
        if (INJ) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                wvs[INJ ? m : 0][k] = *(const float2 *)(wi + m * wk + col);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NF; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        const float2 gv = gvs[k];
        const float v0 = scale * xs[2 * k] * gv.x, v1 = scale * xs[2 * k + 1] * gv.y;
        if (store_xn_f32) *(float2 *)(xn + col) = make_float2(v0, v1);
        if (xh) *(uint32_t *)(xh + col) = hc_pack2(v0, v1);
        if (INJ) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float2 wv = wvs[INJ ? m : 0][k];
                pa[m] = fmaf(v1, wv.y, fmaf(v0, wv.x, pa[m]));
            }
        }
    }
    } }
    if (!whole)
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        if (col + 1 < n_embd) {
            const float2 gv = *(const float2 *)(g + col);
            const float v0 = scale * xs[2 * k] * gv.x, v1 = scale * xs[2 * k + 1] * gv.y;
            if (store_xn_f32) *(float2 *)(xn + col) = make_float2(v0, v1);
            if (xh) *(uint32_t *)(xh + col) = hc_pack2(v0, v1);
            if (INJ) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const float2 wv = *(const float2 *)(wi + m * wk + col);
                    pa[m] = fmaf(v1, wv.y, fmaf(v0, wv.x, pa[m]));
                }
            }
        } else if (col < n_embd) {
            const float v0 = scale * xs[2 * k] * g[col];
            if (store_xn_f32) xn[col] = v0;
            if (xh) xh[col] = hc_f2bf32(v0);
            if (INJ) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    pa[m] = fmaf(v0, wi[m * wk + col], pa[m]);
                }
            }
        }
    }
    if (INJ) {
        // fixed order throughout: a butterfly within each wave, then the waves in index order
        __shared__ float s_part[HC_CN_BLOCK2 / 32][4];
        const int lane = tid % 32, wave = tid / 32;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            pa[m] = warp_reduce_sum(pa[m]);
        }
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                s_part[wave][m] = pa[m];
            }
        }
        __syncthreads();
        if (tid < 4) {
            float sum = s_part[0][tid];
#pragma unroll
            for (int w = 1; w < HC_CN_BLOCK2 / 32; ++w) {
                sum += s_part[w][tid];
            }
            inj_part[row * 4 + tid] = sum;
        }
    }
}

// inject[t * hc + m] = sum over the streams c, in order, of the partials the combine wrote
static __global__ void hc_inject_reduce_f32(const float * __restrict__ part, float * __restrict__ dst, const int hc, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t t = i / hc;
    const int     m = (int) (i % hc);
    const float * p = part + t * hc * 4 + m;
    float sum = p[0];
    for (int c = 1; c < hc; ++c) {
        sum += p[c * 4];
    }
    dst[i] = sum;
}

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & a, const int warp_size) {
    const int64_t n_embd = a.out_res->ne[0];
    const int64_t hc     = a.out_res->ne[1];
    return warp_size == 32 && n_embd >= 1024 && n_embd <= HC_CN_MAX_EMB && hc >= 1 && hc <= 16 &&
        ggml_nelements(a.gamma) == n_embd * hc;
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & a) {
    const int64_t n_embd   = a.out_res->ne[0];
    const int64_t hc       = a.out_res->ne[1];
    const int64_t n_tokens = a.out_res->ne[2] * a.out_res->ne[3];

    GGML_ASSERT(a.out_res->ne[3] == 1);
    GGML_ASSERT(ggml_nelements(a.out_res) == ggml_nelements(a.out_xn) && ggml_are_same_shape(a.out_res, a.residual));
    GGML_ASSERT(ggml_nelements(a.block_out) == n_embd * n_tokens && ggml_nelements(a.gamma) == n_embd * hc &&
                ggml_nelements(a.inject) == hc * n_tokens);

    static const int shape = getenv("LLAMA_HC_CN_SHAPE") ? atoi(getenv("LLAMA_HC_CN_SHAPE")) : 0;
    GGML_ASSERT(!a.inject_w || (shape == 1 && a.inject_part && hc == 4 && n_embd % 2 == 0 && n_tokens <= HC_INJ_MAX_T));
    if (shape == 1) {
        const ggml_cuda_kernel_launch_params lp2(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK2, 0, ctx.stream());
        // strixllama: 2560-wide streams take the unguarded loops (STRIX_HC_CN_WHOLE=0: off)
        static const bool whole = !getenv("STRIX_HC_CN_WHOLE") || atoi(getenv("STRIX_HC_CN_WHOLE")) != 0;
        auto kernel = (whole && n_embd == 5 * 2 * HC_CN_BLOCK2)
            ? (a.inject_w ? hc_combine_norm_f32_b256<true, 5> : hc_combine_norm_f32_b256<false, 5>)
            : (a.inject_w ? hc_combine_norm_f32_b256<true>    : hc_combine_norm_f32_b256<false>);
        ggml_cuda_kernel_launch(kernel, lp2,
            (const float *) a.inject->data, (const float *) a.residual->data,
            (const float *) a.block_out->data, (const float *) a.gamma->data,
            (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32, a.xn_bf16_ld ? a.xn_bf16_ld : hc * n_embd,
            a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
            (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps, a.inject_w, a.inject_part, a.row_scale);
        static unsigned h2 = 0; if (h2++ < 2) fprintf(stderr, "HC_CN shape=256x4 n_embd=%lld tokens=%lld\n", (long long) n_embd, (long long) n_tokens);
        return;
    }
    GGML_ASSERT(!a.row_scale && "the per-row scale is the 256-thread kernel's");
    const ggml_cuda_kernel_launch_params launch_params(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_norm_f32, launch_params,
        (const float *) a.inject->data, (const float *) a.residual->data,
        (const float *) a.block_out->data, (const float *) a.gamma->data,
        (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32, a.xn_bf16_ld ? a.xn_bf16_ld : hc * n_embd,
        a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
        (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps);
}

bool ggml_cuda_hc_inject_fusable(const ggml_cuda_hc_combine_norm_args & a) {
    static const int shape = getenv("LLAMA_HC_CN_SHAPE") ? atoi(getenv("LLAMA_HC_CN_SHAPE")) : 0;
    static const bool on = !getenv("STRIX_HC_INJECT_FUSE") || atoi(getenv("STRIX_HC_INJECT_FUSE")) != 0;
    return on && shape == 1 && a.out_res->ne[1] == 4 && a.out_res->ne[0] % 2 == 0 && a.out_res->ne[3] == 1 &&
        a.out_res->ne[2] <= HC_INJ_MAX_T;
}

// one buffer for the partials of the pending inject: allocated once at its largest, so its address never changes under
// a captured graph; released with the backend (ggml_cuda_hc_release)
bool ggml_cuda_hc_cn_b256(const ggml_cuda_hc_combine_norm_args & a) {
    static const int shape = getenv("LLAMA_HC_CN_SHAPE") ? atoi(getenv("LLAMA_HC_CN_SHAPE")) : 0;
    return shape == 1 && a.out_res->ne[0] % 2 == 0 && a.out_res->ne[3] == 1 && a.out_res->ne[1] <= 4 &&
        a.out_res->ne[2] * a.out_res->ne[1] <= (int64_t) HC_INJ_MAX_T * 4;
}

static ggml_cuda_pool_alloc<float> * g_hc_row_scale = nullptr;
float * ggml_cuda_hc_row_scale_buf(ggml_backend_cuda_context & ctx) {
    if (!g_hc_row_scale) {
        g_hc_row_scale = new ggml_cuda_pool_alloc<float>(ctx.pool(), (size_t) HC_INJ_MAX_T * 4);
    }
    return g_hc_row_scale->get();
}

static ggml_cuda_pool_alloc<float> * g_hc_inject_part = nullptr;
float * ggml_cuda_hc_inject_part(ggml_backend_cuda_context & ctx) {
    if (!g_hc_inject_part) {
        g_hc_inject_part = new ggml_cuda_pool_alloc<float>(ctx.pool(), (size_t) HC_INJ_MAX_T * 4 * 4);
    }
    return g_hc_inject_part->get();
}

void ggml_cuda_hc_release() {
    delete g_hc_inject_part;
    g_hc_inject_part = nullptr;
    delete g_hc_row_scale;
    g_hc_row_scale = nullptr;
}

void ggml_cuda_hc_inject_reduce(ggml_backend_cuda_context & ctx, const float * part, ggml_tensor * dst) {
    const int64_t hc = dst->ne[0], n = ggml_nelements(dst);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && hc == 4 && ggml_nrows(dst) <= HC_INJ_MAX_T);
    hc_inject_reduce_f32<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>(part, (float *) dst->data, (int) hc, n);
    CUDA_CHECK(cudaGetLastError());
}
