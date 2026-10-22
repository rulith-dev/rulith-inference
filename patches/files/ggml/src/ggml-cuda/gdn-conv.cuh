#pragma once
#include "common.cuh"
struct ggml_cuda_gdn_conv_match {
    int concat_idx = -1, conv_idx = -1;   // node indices
    const ggml_tensor * x = nullptr;      // [C, T] F32 contiguous (root of the transpose view)
    const ggml_tensor * state = nullptr;  // [3, C] F32 contiguous
    const ggml_tensor * concat = nullptr; // [T+3, C] F32
    const ggml_tensor * w = nullptr;
    ggml_tensor * conv_out = nullptr;
    int64_t C = 0, T = 0, tail_from = 0;  // first concat column that must be materialized
    bool silu = false;
    bool alias = false;                   // the SiLU's output is x's own memory (the halos are copied first)
    // strixllama: a decode batch - the state copies after the concat (a CPY of its last columns into the cache rows, a
    // rollback slot each) done by the conv's kernel; the concat and the copies dispatch nothing
    int n_tail = 0;
    float * tail_dst[8] = {};
    int tail_col[8] = {};
    const ggml_tensor * tail_cpy[8] = {};
};
// strixllama: the q and k L2 norms right after the conv's SiLU (VIEW, RMS_NORM, SCALE twice), done in the conv's pass
struct ggml_cuda_gdn_conv_norm {
    ggml_tensor * q = nullptr, * k = nullptr;   // the SCALE nodes: [D, H, T], never written (the net reads the SiLU's rows)
    int64_t nq = 0;                              // channels of q (= of k): the first 2 nq of the conv's are normed
    float eps = 0.0f, scale = 1.0f, bias = 0.0f;
    int last = -1;                               // the last node it covers (the second SCALE)
    const ggml_tensor * vq = nullptr, * vk = nullptr;   // the views of the SiLU the norms read
    const ggml_tensor * gdn = nullptr;           // the net that reads the norms' outputs (only it)
};
bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_norm_match(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m, ggml_cuda_gdn_conv_norm & n);
void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
// a CPY the conv's kernel does (ggml_cuda_gdn_conv_match::n_tail): skip its dispatch
bool ggml_cuda_gdn_conv_tail_skip(const ggml_tensor * node);
// forget the marked copies (at the start of a graph's evaluation)
void ggml_cuda_gdn_conv_tail_reset();
void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_norm_run(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m, const ggml_cuda_gdn_conv_norm & n);
