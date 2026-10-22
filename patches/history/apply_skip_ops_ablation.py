#!/usr/bin/env python3
"""Diagnostic: STRIX_SKIP_OPS replaces whole op classes by a zero fill in the HIP backend.

STRIX_SKIP_OPS=SCALE,UNARY,UNARY:SIGMOID,GLU:SWIGLU,MUL_MAT_ID,... (op names from ggml_op_name,
optional :subtype for UNARY / GLU). Each matching node is cudaMemsetAsync'ed to zero instead of
computed, so the wall time of an op class can be measured by difference. The output is garbage,
so this is for tools/decode_probe.py timing runs only (results/hip-rocm101-20260917.json,
decode_ablation_20260918). Off unless the variable is set.

Usage: python patches/apply_skip_ops_ablation.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
MAIN = os.path.join(tree, "ggml", "src", "ggml-cuda", "ggml-cuda.cu")

HELPER = '''// strixllama ablation helper: which op classes STRIX_SKIP_OPS asks to replace by a zero fill
static bool strixllama_skip_op(const ggml_tensor * node) {
    static const std::vector<std::string> wanted = []() {
        std::vector<std::string> v;
        if (const char * e = getenv("STRIX_SKIP_OPS")) {
            std::string cur;
            for (const char * c = e; ; ++c) {
                if (*c == ',' || *c == 0) { if (!cur.empty()) { v.push_back(cur); } cur.clear(); if (*c == 0) { break; } }
                else { cur.push_back(*c); }
            }
            if (!v.empty()) { fprintf(stderr, "STRIX_SKIP_OPS: %zu op classes zero-filled instead of computed\\n", v.size()); }
        }
        return v;
    }();
    if (wanted.empty()) { return false; }
    const char * op = ggml_op_name(node->op);
    for (const auto & w : wanted) {
        const size_t colon = w.find(':');
        if (colon == std::string::npos) { if (w == op) { return true; } continue; }
        if (w.compare(0, colon, op) != 0) { continue; }
        if (node->op == GGML_OP_UNARY && w.compare(colon + 1, std::string::npos, ggml_unary_op_name(ggml_get_unary_op(node))) == 0) { return true; }
        if (node->op == GGML_OP_GLU && w.compare(colon + 1, std::string::npos, ggml_glu_op_name(ggml_get_glu_op(node))) == 0) { return true; }
    }
    return false;
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {'''

HOOK_OLD = '''                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
'''
HOOK_NEW = '''                // strixllama ablation: STRIX_SKIP_OPS=SCALE,UNARY,UNARY:SIGMOID,... replaces those nodes by a
                // zero fill so the per-token cost of an op class can be measured (output is garbage)
                if (strixllama_skip_op(node)) {
                    CUDA_CHECK(cudaMemsetAsync(node->data, 0, ggml_nbytes(node), cuda_ctx->stream()));
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
'''


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    if "strixllama_skip_op" in s:
        print("already applied")
        return
    anchor = "static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {"
    if s.count(anchor) != 1 or s.count(HOOK_OLD) != 1:
        sys.exit("anchors not found in %s" % MAIN)
    s = s.replace(anchor, HELPER).replace(HOOK_OLD, HOOK_NEW)
    io.open(MAIN, "w", encoding="utf-8", newline="").write(s)
    print("patched", MAIN)


if __name__ == "__main__":
    main()
