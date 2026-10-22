#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);

// strixllama: fused GGML_OP_SCALE + GGML_UNARY_OP_SIGMOID [+ GGML_OP_SCALE]
void ggml_cuda_op_scale_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src, bool post_scale);
