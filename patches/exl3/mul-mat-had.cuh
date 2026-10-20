#pragma once
// strixllama: from ranma.cpp (saturnsky), MIT license

#include "common.cuh"

// GGML_OP_MUL_MAT_HAD for F32, F16 and EXL3 weights, had 32/64/128, K and N multiples of 128
bool ggml_cuda_mul_mat_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// strixllama: dst = MUL_MAT_HAD dst and other = the other product of a swiglu GLU (already computed): writes the GLU
// output glu = silu(gate)*up (dst_is_gate: dst is the gate) in dst's epilogue; dst itself is not written
void ggml_cuda_mul_mat_had_glu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * other,
        const ggml_tensor * glu, bool dst_is_gate);

// strixllama: the product with ids followed by a MUL by the routing weights [1, n_used, T]: y*w written by the epilogue
void ggml_cuda_mul_mat_had_rowmul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * row_mul);
