#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
    // strixllama: each sequence's input state read from row state_rows[seq] of state_base (the whole state tensor: a
    // row of the sequence's own cell, its newest state or a rollback snapshot) instead of src[5] - the gather is skipped
    const float *   state_base = nullptr;
    const int32_t * state_rows = nullptr;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// strixllama: the next run of the net gdn reads q and k from q / k with these strides (in floats: head, token, sequence)
// instead of its sources' data - gdn-conv.cu's fused conv wrote them normed into the SiLU's rows, and the norms' outputs
// the sources name are never written
void ggml_cuda_gdn_qk_from(const ggml_tensor * gdn, const float * q, const float * k, int64_t s1, int64_t s2, int64_t s3);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// strixllama: the replay alone, outside a graph, for the cells a checkpoint or a seq_cp flattens (reached through
// ggml_backend_reg_get_proc_address as "ggml_backend_cuda_gdn_replay"): row `row` of each layer's state gets its
// n_rep pending records (from record index `first`) applied by the kernel the next batch would run, so the bits are
// the kernel's. Synchronous
void ggml_backend_cuda_gdn_replay(ggml_backend_t backend, int n_layer, ggml_tensor ** states, ggml_tensor ** recs,
                                  int32_t row, int32_t n_rep, int32_t first, int S_v, int H_v, int H_k);
