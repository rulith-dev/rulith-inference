#pragma once

#include "common.cuh"

// Selected-attention for the Qwen3.8-Next indexer path: the lightning indexer picks 4-key blocks
// per query, so attention only has to visit those. Needs the packed key/value layouts the graph
// builds in src[6]/src[7]; returns false for anything else, including the dense prefill case.
bool ggml_cuda_flash_attn_ext_qsa_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// strixllama: CONT(permuted cache view) -> RESHAPE -> PERMUTE -> CONT(packed K or V) in one pass; the nodes consumed
// after nodes[i], or 0
int ggml_cuda_qsa_pack_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i);
// the same match on the graph's structure alone: the source to keep allocated until `out` is computed (graph_optimize)
bool ggml_cuda_qsa_pack_deps(const ggml_cgraph * g, int i, ggml_tensor ** src_root, ggml_tensor ** out);

// strixllama: GGML_OP_QSA_PACK - a K or V cache view (F16, Q8_0, Q5_1) written into the packed layouts directly
bool ggml_cuda_qsa_pack_supported(const ggml_tensor * op);
void ggml_cuda_qsa_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// strixllama: CONT(strided F32 view) followed by its only reader SIGMOID, in one pass; the nodes consumed after
// nodes[i], or 0
int ggml_cuda_cont_sigmoid_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i);
bool ggml_cuda_cont_sigmoid_deps(const ggml_cgraph * g, int i, ggml_tensor ** src_root, ggml_tensor ** out);
