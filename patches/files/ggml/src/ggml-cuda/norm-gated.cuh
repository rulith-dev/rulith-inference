#pragma once
#include "common.cuh"
struct ggml_cuda_norm_gated_match { const ggml_tensor * x; const ggml_tensor * w; const ggml_tensor * z; ggml_tensor * dst; float eps; int pre = -1; /* node index of the gate MUL_MAT to compute first (or -1) */ };
int  ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m);
int  ggml_cuda_norm_rows_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
struct ggml_cuda_norm_scale_match { const ggml_tensor * x; ggml_tensor * dst; float eps, scale, bias; };
int  ggml_cuda_norm_scale_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_scale_match & m, int64_t min_rows = 4096);
void ggml_cuda_op_norm_scale(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_scale_match & m);
bool ggml_cuda_rms_rows_plain(const float * x, float * dst, int ncols, int64_t nrows, int64_t nchannels, int64_t nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample, float eps, cudaStream_t stream);
