#pragma once
// strixllama: gather straight into the type the consumer wants, instead of via an F32 copy.
//
// ggml_get_rows always produces F32 (ggml.c fixes the result type), so a graph that wants F16 rows
// out of an F16 source has to write ggml_cast(ggml_get_rows(...), F16) and pay for the widening:
// read F16, write F32, read F32, write F16. qwen4exp does this on the decode path for the selected
// K and V cells in qwen4exp_gather_attn, twice per attention layer.
//
// Measured at 97K context, per decoded token: fusing the 24 pairs removes 169 MB of traffic, about
// 0.8 ms at the 215 GB/s this machine reaches on a dense matvec.
//
// The CUDA get_rows kernels are already templated on the destination type and get_rows_cuda
// dispatches F16/BF16/I32 destinations, so the fused form is the same kernel with a different
// dst_t. Widening an F16 value to F32 and narrowing it back is lossless, so the fused result is
// bit-identical to the pair it replaces.
//
// Returns the number of nodes to skip after node i (0 = no fusion). LLAMA_GETROWS_CAST=0 disables.
#include "common.cuh"

int ggml_cuda_get_rows_cast_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
