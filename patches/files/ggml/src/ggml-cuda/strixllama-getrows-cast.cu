#include "strixllama-getrows-cast.cuh"
#include "getrows.cuh"

#include <cstdlib>

int ggml_cuda_get_rows_cast_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_GETROWS_CAST");
        return !e || std::atoi(e) != 0;
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }

    const ggml_tensor * rows = cgraph->nodes[i];
    ggml_tensor *       cast = cgraph->nodes[i + 1];

    if (rows->op != GGML_OP_GET_ROWS || cast->op != GGML_OP_CPY) {
        return 0;
    }

    // ggml_cast: a CPY whose second source is the node itself, so it is a pure retype, not a
    // write into somebody else's buffer
    if (cast->src[0] != rows || cast->src[1] != cast) {
        return 0;
    }

    const ggml_tensor * src0 = rows->src[0];
    const ggml_tensor * src1 = rows->src[1];

    // only worth it when the gather actually changes type, and only into what get_rows_cuda writes
    if (rows->type != GGML_TYPE_F32 || cast->type == GGML_TYPE_F32) {
        return 0;
    }
    if (cast->type != GGML_TYPE_F16 && cast->type != GGML_TYPE_BF16) {
        return 0;
    }
    if (src1->type != GGML_TYPE_I32 || !ggml_are_same_shape(rows, cast)) {
        return 0;
    }

    // the asserts ggml_cuda_op_get_rows makes, restated for the destination we are substituting
    if (src1->ne[2] != 1 || rows->ne[3] != 1 ||
        src0->nb[0] != ggml_type_size(src0->type) ||
        src1->nb[0] != ggml_type_size(src1->type) ||
        cast->nb[0] != ggml_type_size(cast->type)) {
        return 0;
    }

    const ggml_op ops[2] = { GGML_OP_GET_ROWS, GGML_OP_CPY };
    const int indices[2] = { i, i + 1 };
    const int output     = i + 1;
    if (!ggml_can_fuse_subgraph_ext(cgraph, indices, 2, ops, &output, 1)) {
        return 0;
    }

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, cast->data, cast->type,
        src0->ne[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->nb[0], src1->nb[1], src1->nb[2],
        cast->nb[1], cast->nb[2], cast->nb[3], ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    return 1;
}
