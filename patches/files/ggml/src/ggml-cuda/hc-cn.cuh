#pragma once
#include "common.cuh"

struct ggml_cuda_hc_combine_norm_args {
    const ggml_tensor * inject;     // [hc, T]  F32 contiguous (pre-scale/sigmoid)
    const ggml_tensor * residual;   // [n_embd, hc, T] F32 contiguous
    const ggml_tensor * block_out;  // [n_embd, 1, T]  F32 contiguous
    const ggml_tensor * gamma;
    ggml_tensor *       out_res;
    ggml_tensor *       out_xn;     // [n_embd, hc, T]
    float               s1, b1, s2, b2;
    float               eps;
    uint16_t *          out_xn_bf16 = nullptr;
    int64_t             xn_bf16_ld  = 0;         // strixllama: the BF16 copy's token stride when padded (0: hc * n_embd)
    bool                store_xn_f32 = true;     // false: consumers all read the BF16 copy
    const uint16_t *    res_in_bf16  = nullptr;   // `residual` is BF16 in place (marked bf16-only)
    const uint16_t *    blk_in_bf16  = nullptr;
    uint16_t *          res_out_bf16 = nullptr;
    const float *       inject_w     = nullptr;   // strixllama: the next hc_inject's weights [hc * n_embd, hc], fused (hc == 4)
    float *             inject_part  = nullptr;   // its per-stream partial dot products [T][hc][4]
    float *             row_scale    = nullptr;   // strixllama: each row's rsqrt(mean(x^2) + eps) [T * hc] (STRIX_HC_XRES)
};

// the largest ubatch the fused inject takes (its partial buffer is sized for it)
#define HC_INJ_MAX_T 65536

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & args, int warp_size);
void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & args);
bool ggml_cuda_hc_inject_fusable(const ggml_cuda_hc_combine_norm_args & args);
float * ggml_cuda_hc_inject_part(ggml_backend_cuda_context & ctx);
void ggml_cuda_hc_inject_reduce(ggml_backend_cuda_context & ctx, const float * part, ggml_tensor * dst);
void ggml_cuda_hc_release();
// strixllama: the combine kernel the per-row scale needs (LLAMA_HC_CN_SHAPE=1), and a buffer for the scales
bool ggml_cuda_hc_cn_b256(const ggml_cuda_hc_combine_norm_args & args);
float * ggml_cuda_hc_row_scale_buf(ggml_backend_cuda_context & ctx);
