#include "softcap.cuh"

static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i]) * softcap;
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), stream);
}

// strixllama: the same shape as softcap above but with SIGMOID, which is what this architecture's
// hyper-connection gate actually uses - a LLAMA_GRAPH_TRACE census of a decode pass counts 195
// SIGMOID and zero TANH, so the upstream softcap fusion can never fire here and every
// scale/sigmoid/scale run costs three separate launches on tensors as small as [4, 512].
// Handles both the triple and the bare scale+sigmoid pair, and folds in each scale's bias so the
// matcher does not have to reject a nonzero one.
static __global__ void scale_sigmoid_f32(const float * x, float * dst,
                                         const float a, const float b0, const float c, const float b1, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    const float v = fmaf(a, x[i], b0);
    dst[i] = fmaf(1.0f / (1.0f + expf(-v)), c, b1);
}

static void scale_sigmoid_f32_cuda(const float * x, float * dst,
                                   const float a, const float b0, const float c, const float b1,
                                   const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_sigmoid_f32, launch_params, x, dst, a, b0, c, b1, k);
}

// src is the leading GGML_OP_SCALE, dst the last node of the run; post_scale says whether that
// last node is a second GGML_OP_SCALE rather than the sigmoid itself.
void ggml_cuda_op_scale_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src, bool post_scale) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float a, b0, c = 1.0f, b1 = 0.0f;
    memcpy(&a,  (float *) src->op_params + 0, sizeof(float));
    memcpy(&b0, (float *) src->op_params + 1, sizeof(float));
    if (post_scale) {
        memcpy(&c,  (float *) dst->op_params + 0, sizeof(float));
        memcpy(&b1, (float *) dst->op_params + 1, sizeof(float));
    }

    scale_sigmoid_f32_cuda(src0_d, dst_d, a, b0, c, b1, ggml_nelements(src0), stream);
}
