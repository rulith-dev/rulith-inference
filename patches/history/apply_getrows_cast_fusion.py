#!/usr/bin/env python3
"""Fuse GET_ROWS -> ggml_cast(F16) into one gather that writes F16 directly.

ggml_get_rows always produces F32 (ggml.c fixes the result type), so a graph that wants F16 rows out
of an F16 source has to write ggml_cast(ggml_get_rows(...), F16) and pay a widening round trip:
read F16, write F32, read F32, write F16. qwen4exp_gather_attn does exactly this for the selected K
and V cells on the decode path, twice per attention layer.

The CUDA get_rows kernels are already templated on the destination type and get_rows_cuda already
dispatches F16/BF16/I32 destinations, so the fused form is the same kernel with a different dst_t.
Widening an F16 value to F32 and narrowing it back is lossless, so the result is bit-identical.

Measured at 97K context: the decode graph goes from 2378 to 2354 dispatches, which is exactly the 24
CPY nodes (12 attention layers x K and V) the fusion absorbs. By traffic, each of those 24 gathers
stops writing 4.7 MB of F32 and reading it back, and writes 2.36 MB of F16 instead: 169 MB per
decoded token, about 0.8 ms at the 215 GB/s this machine reaches. A wall-clock A/B on the same
binary moved 50.42 -> 49.95 ms/token, which is consistent but inside the +/-1.5 ms run-to-run
spread, so the dispatch count and the traffic arithmetic are the real evidence, not the clock.
tools/decode_lab.py --gen output is byte-identical with the fusion on and off.

LLAMA_GETROWS_CAST=0 disables it at runtime.

Usage: python patches/apply_getrows_cast_fusion.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
CUDA = os.path.join(tree, "ggml", "src", "ggml-cuda")
MAIN = os.path.join(CUDA, "ggml-cuda.cu")

CUH = '''#pragma once
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
'''

CU = '''#include "strixllama-getrows-cast.cuh"
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
'''

OLD_INC = '#include "ggml-cuda/strixllama-chain.cuh"'
NEW_INC = '#include "ggml-cuda/strixllama-chain.cuh"\n#include "ggml-cuda/strixllama-getrows-cast.cuh"'

OLD_HOOK = '''    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_hc_mix_args args;
        const int count=ggml_cuda_hc_mix_closed(cgraph,i,args);'''

NEW_HOOK = '''    // strixllama: gather straight into F16 rather than through an F32 copy (strixllama-getrows-cast.cuh)
    if (node->op == GGML_OP_GET_ROWS) {
        if (const int skip = ggml_cuda_get_rows_cast_try(*cuda_ctx, cgraph, i)) {
            return skip;
        }
    }

    if (GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        ggml_cuda_hc_mix_args args;
        const int count=ggml_cuda_hc_mix_closed(cgraph,i,args);'''


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    if "strixllama-getrows-cast" in s:
        print("already applied")
        return
    for old in (OLD_INC, OLD_HOOK):
        if s.count(old) != 1:
            sys.exit("anchor found %d times, expected once, in %s:\n%s" % (s.count(old), MAIN, old[:120]))
    io.open(os.path.join(CUDA, "strixllama-getrows-cast.cuh"), "w", encoding="utf-8", newline="").write(CUH)
    io.open(os.path.join(CUDA, "strixllama-getrows-cast.cu"), "w", encoding="utf-8", newline="").write(CU)
    s = s.replace(OLD_INC, NEW_INC).replace(OLD_HOOK, NEW_HOOK)
    io.open(MAIN, "w", encoding="utf-8", newline="").write(s)
    print("wrote strixllama-getrows-cast.{cu,cuh} and patched", MAIN)
    print("note: the ggml-cuda CMakeLists globs *.cu at configure time, so re-run cmake before ninja")


if __name__ == "__main__":
    main()
