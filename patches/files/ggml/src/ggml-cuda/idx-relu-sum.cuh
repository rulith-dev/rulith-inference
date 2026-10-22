#include "common.cuh"

struct ggml_cuda_idx_relu_sum_args {
    const ggml_tensor * score = nullptr;
    ggml_tensor *       dst   = nullptr;   // [n_blocks, n_tps, n_stream]
    int                 heads = 0;
};
bool ggml_cuda_idx_relu_sum_enabled();
void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args);

// strixllama: the lightning indexer's whole score for one query strip in one kernel (see ggml_cuda_match_idx_score)
struct ggml_cuda_idx_score_args {
    const ggml_tensor * keys   = nullptr;   // [128, n_blocks] F32, the block keys (F16 values out of the block-key cache)
    const ggml_tensor * q      = nullptr;   // [128, 4 * n_query] F32 view, column t * 4 + h
    const int32_t *     starts = nullptr;   // [n_blocks] first position that sees each block
    const int32_t *     tails  = nullptr;   // [n_query]  each query's position bound
    ggml_tensor *       out    = nullptr;   // [n_blocks, n_query] F32
};
void ggml_cuda_op_idx_score(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_args & args);

// strixllama: the same score for a decode-sized strip, with the block keys read straight out of the block-key cache
// (see ggml_cuda_match_idx_score_dec)
struct ggml_cuda_idx_score_dec_args {
    const ggml_tensor * kb     = nullptr;   // [128, n_rows] F16, the block-key cache (a view of it)
    const ggml_tensor * rows   = nullptr;   // [n_blocks] I32, each block's row in it
    const ggml_tensor * q      = nullptr;   // [128, 4 * n_query] F32 view, column t * 4 + h
    const int32_t *     starts = nullptr;   // [n_blocks] first position that sees each block
    const int32_t *     tails  = nullptr;   // [n_query]  each query's position bound
    ggml_tensor *       out    = nullptr;   // [n_blocks, n_query] F32
    // several sequences in the strip: 0/1 ownership, visible only where the block's and the query's slots meet
    const float *       seq_blk = nullptr;  // [n_slots, n_blocks]
    const float *       seq_tok = nullptr;  // [n_slots, n_query]
    int                 n_slots = 0;
};
void ggml_cuda_op_idx_score_dec(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_dec_args & args);
