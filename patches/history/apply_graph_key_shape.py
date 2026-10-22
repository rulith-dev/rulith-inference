#!/usr/bin/env python3
"""Key HIP graphs on the graph's shape as well as its first node.

The CUDA backend keeps one captured graph per key, and the key was the first node's tensor pointer.
A context that alternates between batch shapes therefore fought itself: the MTP draft context runs
its catch-up at 1+n_draft tokens and then its draft steps at 1 token, the target runs 1+n_draft, and
every switch compared the new graph against the other shape's properties, saw a change, reset the
warmup, ran the graph direct, and captured again on the next call. Per speculative pass at 85K
context that was: catch-up direct, draft step 1 direct, draft step 2 captured, draft step 3 replayed.

With the shape (n_nodes, every node's op and ne) folded into the key each shape keeps its own
instance, and all four draft graphs replay. Measured with LLAMA_GRAPH_TIMING at 85K, per pass:

  catch-up      submit 0.7 -> 0.2 ms, gpu 2.0 -> 2.0
  draft step 1  submit 0.7 -> 0.2 ms, gpu 5.2 -> 4.4
  draft step 2  submit 0.9 -> 0.1 ms, gpu 4.3 -> 4.3

about 2.6 ms of an 89 ms pass. Wall clock on the same binary, 160 tokens after 85K of real prose:
29.95 -> 29.35 ms/token (+2.0%), identical draft counts (106/156), byte-identical output. Instances
are still evicted after 10 s unused, so the set stays small (a handful of batch sizes per context).

STRIX_GRAPH_KEY_SHAPE=0 restores the first-node key at runtime.

Usage: python patches/apply_graph_key_shape.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
SRC = os.path.join(tree, "ggml", "src", "ggml-cuda", "ggml-cuda.cu")

OLD_KEY = '''static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}'''

NEW_KEY = '''static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    // strixllama: fold the graph's shape into the key. One context alternates between batch shapes on the
    // same first node (the MTP draft's catch-up at 1+n_draft tokens then its draft steps at 1, the
    // target at 1+n_draft), and keying on the first node alone made every switch look like a property
    // change: warmup reset, the next graph run direct, the one after captured again. With the shape in
    // the key each shape keeps its own instance and replays. The value is only ever a map key.
    // STRIX_GRAPH_KEY_SHAPE=0 restores the first-node key.
    static const bool by_shape = [] {
        const char * e = getenv("STRIX_GRAPH_KEY_SHAPE");
        return !e || atoi(e) != 0;
    }();
    if (!by_shape || cgraph->n_nodes == 0) {
        return cgraph->nodes[0];
    }
    uint64_t h = (uint64_t) (uintptr_t) cgraph->nodes[0];
    h ^= (uint64_t) cgraph->n_nodes * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        h = (h ^ (uint64_t) t->op) * 0x100000001B3ull;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = (h ^ (uint64_t) t->ne[d]) * 0x100000001B3ull;
        }
    }
    return (const void *) (uintptr_t) h;
}'''

# the property check recomputed the key; hand it the one graph_compute already computed
OLD_UPD = '''static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);'''

NEW_UPD = '''static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const void * graph_key) {
    bool res = false;

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);'''

OLD_CALL = '            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);'
NEW_CALL = '            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph, graph_key);'


def main():
    s = io.open(SRC, encoding="utf-8").read()
    if "STRIX_GRAPH_KEY_SHAPE" in s:
        print("already applied")
        return
    for old in (OLD_KEY, OLD_UPD, OLD_CALL):
        if s.count(old) != 1:
            sys.exit("anchor found %d times, expected once, in %s:\n%s" % (s.count(old), SRC, old[:120]))
    s = s.replace(OLD_KEY, NEW_KEY).replace(OLD_UPD, NEW_UPD).replace(OLD_CALL, NEW_CALL)
    io.open(SRC, "w", encoding="utf-8", newline="").write(s)
    print("patched", SRC)


if __name__ == "__main__":
    main()
