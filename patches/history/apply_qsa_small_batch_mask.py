#!/usr/bin/env python3
"""Keep the causal mask on sparse-attention batches the qsa3 kernel will not take.

With LLAMA_QSA_NO_DENSE_MASK=1 (the maskless trial, patches/apply_qsa_maskless_trial.py) the QSA
attention op is built with block indices and no kq_mask: the qsa3 kernel (ggml-cuda/qsa.cu) applies
the selection itself. But qsa3 declines batches under 128 queries, and on HIP the generic
flash-attention kernels ignore the indices (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse() is
CUDA-only), so every such batch ran DENSE attention over the whole cache with NO causal mask.

A single-token decode has nothing to leak. A speculative verification batch does: its 2-8 tokens
are the drafts, and the target's logits for token i could see tokens i+1.. - the very drafts they
verify. Once the context passed indexer_top_k (QSA active, ~2K tokens) MTP answers drifted into
repetition and stopped mid-sentence on <|im_end|>. Measured on corpus/ppl-en-code.txt, ctx 4096:

    v3 kernel on (prefill)          2.4724
    dense attention (qsa off)       2.4761
    generic kernel, ubatch 512      1.0841   <- impossible PPL: future tokens visible
    generic kernel, ubatch 4        1.8397   <- the MTP verification shape
    generic kernel, ubatch 4, fixed 2.4646

The fix keeps `maskless` only for n_tps >= 128, where qsa3 actually runs, and mirrors that in
can_reuse(). Small batches carry the [n_kv, n_tokens] f16 kq_mask again - a few MB at most.

Usage: python patches/apply_qsa_small_batch_mask.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
SRC = os.path.join(root, "src", "models", "qwen4exp.cpp")

OLD_BUILD = '''        qsa->compact = scalar && qwen4exp_qsa_flag("LLAMA_QSA_COMPACT_METADATA");
        qsa->maskless = scalar && qwen4exp_qsa_flag("LLAMA_QSA_NO_DENSE_MASK");
'''
NEW_BUILD = '''        qsa->compact = scalar && qwen4exp_qsa_flag("LLAMA_QSA_COMPACT_METADATA");
        // strixllama: only the qsa3 kernel honours the indices without a mask, and it declines batches under
        // 128 queries. On HIP the generic flash-attention kernels ignore the indices (sparse gather is
        // CUDA-only), so a maskless op there runs dense attention over the whole batch with no causal
        // mask: every query of a multi-token decode batch sees the tokens after it. That is exactly the
        // MTP verification batch (2-4 tokens), whose target logits then see the drafts they are meant to
        // check. Keep the mask for anything the qsa3 kernel will not take.
        qsa->maskless = scalar && qwen4exp_qsa_flag("LLAMA_QSA_NO_DENSE_MASK") && n_tps >= 128;
'''
OLD_REUSE = '''        res &= maskless == (scalar && qwen4exp_qsa_flag("LLAMA_QSA_NO_DENSE_MASK"));
'''
NEW_REUSE = '''        res &= maskless == (scalar && qwen4exp_qsa_flag("LLAMA_QSA_NO_DENSE_MASK") &&
                params.ubatch.n_tokens/n_stream >= 128);
'''


def main():
    s = io.open(SRC, encoding="utf-8").read()
    if "n_tps >= 128;" in s and "n_tokens/n_stream >= 128" in s:
        print("already applied")
        return
    if s.count(OLD_BUILD) != 1 or s.count(OLD_REUSE) != 1:
        sys.exit("anchors not found in %s" % SRC)
    s = s.replace(OLD_BUILD, NEW_BUILD).replace(OLD_REUSE, NEW_REUSE)
    io.open(SRC, "w", encoding="utf-8", newline="").write(s)
    print("patched", SRC)


if __name__ == "__main__":
    main()
