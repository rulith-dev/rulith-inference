#pragma once
// strixllama: fuse a run of consecutive elementwise nodes into one kernel.
//
// At batch 1 the qwen4exp graph is ~2300 dispatches per token of which ~1900 do no matrix work:
// SCALE -> SIGMOID -> SCALE on a [4] tensor, ADD -> SOFTPLUS -> MUL on [48], SIGMOID -> MUL -> ADD on
// the shared-expert gate, CLAMP -> DIV on the router weights, REPEAT -> MUL -> ADD on the HC
// inject. Each is a separate launch that touches a few hundred bytes. This fuses any consecutive
// chain of SCALE / CLAMP / UNARY / ADD / SUB / MUL / DIV (F32, contiguous) whose intermediates have
// a single consumer, evaluating the same F32 operations in the same order per output element, so
// the result is bit-identical to the unfused graph. REPEAT operands are folded into broadcast reads
// and RESHAPE aliases between chain nodes are seen through.
//
// ggml_cuda_strixllama_chain_try returns the number of nodes to skip after node i (0 = no fusion).
// STRIX_CHAIN_FUSE=0 disables it.
#include "common.cuh"

// defined in ggml-cuda.cu (made non-static for the STRIX_CHAIN_VERIFY self-check)
bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

int ggml_cuda_strixllama_chain_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
