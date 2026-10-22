#!/usr/bin/env python3
"""Fuse runs of elementwise nodes in the HIP backend into one kernel (decode dispatch count).

Adds ggml/src/ggml-cuda/strixllama-chain.{cu,cuh} (copied from patches/chain-fusion/) and wires them
into ggml-cuda.cu:
  - ggml_cuda_try_fuse() tries the generic chain after every specific pattern
  - ggml_cuda_compute_forward() loses its `static` so the STRIX_CHAIN_VERIFY self-check can
    recompute a chain node by node and compare (run with GGML_CUDA_DISABLE_GRAPHS=1)

Chains: consecutive SCALE / CLAMP / UNARY{SIGMOID,SILU,SOFTPLUS,EXP,TANH,RELU,NEG,ABS,GELU_QUICK}
/ ADD / SUB / MUL / DIV nodes (F32, contiguous) whose intermediates have one consumer, evaluated
per output element in the original order, so the result is bit-identical to the unfused graph.
RESHAPE / contiguous VIEW aliases between chain nodes are seen through. A REPEAT is folded into a
broadcast read of its source ONLY when the REPEAT node itself sits inside the chain: a REPEAT that
already ran before the chain is read as is, because its source's lifetime ended at the REPEAT and
the allocator may have reused that memory (this was the PLE `ple_gated_value` corruption).

Switches: STRIX_CHAIN_FUSE=0 disables; STRIX_CHAIN_ONLY=FIRSTOP-LASTOP keeps one chain
signature (bisection); STRIX_CHAIN_NO_REPEAT=1 stops chains at REPEAT nodes;
STRIX_CHAIN_VERIFY=1 self-checks every fused chain; LLAMA_GRAPH_TRACE=1 prints them.

Measured (Qwen3.8-Flash-Next UD-IQ4_XS, gfx1151, batch-1 decode, spec off): 41.0 -> 39.8 ms/token.
Output at temperature 0 identical to the unfused runtime over 600 tokens; self-check 0 mismatches.

Usage: python patches/apply_chain_fusion.py [tree]
"""
import io
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
CUDA_DIR = os.path.join(tree, "ggml", "src", "ggml-cuda")
MAIN = os.path.join(CUDA_DIR, "ggml-cuda.cu")
SRC_DIR = os.path.join(ROOT, "patches", "chain-fusion")

EDITS = [
    ('#include "ggml-cuda/gated_delta_net.cuh"\n',
     '#include "ggml-cuda/gated_delta_net.cuh"\n#include "ggml-cuda/strixllama-chain.cuh"\n'),
    ('static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {',
     'bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {'),
    ('    return 0;\n}\n\nstatic void ggml_cuda_graph_evaluate_and_capture(',
     '    // strixllama: consecutive elementwise chains (see strixllama-chain.cuh), tried after every specific pattern\n'
     '    if (const int chain = ggml_cuda_strixllama_chain_try(*cuda_ctx, cgraph, i)) {\n'
     '        return chain;\n'
     '    }\n'
     '    return 0;\n}\n\nstatic void ggml_cuda_graph_evaluate_and_capture('),
]


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    if "strixllama-chain.cuh" in s:
        print("already applied")
        return
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit("anchor not unique in %s: %r" % (MAIN, old[:60]))
    for old, new in EDITS:
        s = s.replace(old, new)
    for name in ("strixllama-chain.cu", "strixllama-chain.cuh"):
        shutil.copyfile(os.path.join(SRC_DIR, name), os.path.join(CUDA_DIR, name))
        print("copied", name)
    io.open(MAIN, "w", encoding="utf-8", newline="").write(s)
    print("patched", MAIN)
    print("re-run cmake configure so the new .cu is picked up by the source glob, then build ggml-hip")


if __name__ == "__main__":
    main()
