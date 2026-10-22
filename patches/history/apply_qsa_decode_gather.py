#!/usr/bin/env python3
r"""Decode-side sparse attention: gather the selected cells instead of reading the whole cache.

Only the qsa3 kernel honours the block indices (src[5]), and it needs the packed K/V layout that
qwen4exp builds only for batches of >= 128 tokens; every other kernel ignores the indices and reads
the entire cache. So a single decoded token attended densely over the whole context: per-node timing
put that at 15.6 ms of a 58 ms token at 97K (results/hip-rocm101-20260917.json,
decode_node_timing_97k_20260918). QSA3_FORCE=1 does not help - it lifts the kernel's batch gate but
the packed operands are still absent, so the predicate declines anyway (measured: a null test).

For a decode-sized batch the selection is small and fixed (indexer_top_k + ratio - 1 = 2051 cells),
so this materialises it: gather those cells out of the K and V caches into a compact pair, pad the
length to a multiple of FATTN_KQ_STRIDE, mark "no cell" (-1 from block-graph.inc) and the padding
with a -inf mask, and run the ordinary flash-attention kernel over 2051 keys instead of 97512. No
packed layout, no extra memory. The gate requires the cache to be at least twice the selection:
gathering 2304 of 2329 cells costs more than reading them in place.

This makes decode sparse, which is what the model intends and what prefill already does, so it is
not bit-identical to the dense decode it replaces. Quality at a depth where the selection bites,
ctx 8192 ubatch 4: 1.9867 with the gather against 1.9909 without. Prefill is untouched (ctx 4096
ubatch 512 = 2.4741 either way) and a six-turn MTP + n-gram run completes 6/6 at 28-30 tok/s.

Measured at the 96 GB carve, spec off, ms/token by context (block-key cache on in both):
       2.3K   16K    32K    65K    97K     slope per 1000 tokens
  on   40.46  42.84  42.91  44.94  47.91   0.078
  off  41.07  43.63  46.74  52.70  58.95   0.188

LLAMA_QSA_DECODE_GATHER=1 enables it (manager.py sets it with the other QSA gates);
LLAMA_QSA_DECODE_GATHER_MAX_T caps the batch size it applies to (default 8).

Usage: python patches/apply_qsa_decode_gather.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
MODEL = os.path.join(tree, "src", "models", "qwen4exp.cpp")

HELPER_ANCHOR = "// Dense GQA self-attention restricted to the cells that top_k names."
HELPER = '''// strixllama: decode-side sparse attention by gathering.
//
// The qsa3 kernel is the only one that honours the block indices, and it needs the packed K/V layout
// that qwen4exp builds only for batches of >= 128 tokens; every other kernel ignores src[5] and reads
// the whole cache. So a single decoded token attends densely over the entire context - 15.6 ms of a
// 58 ms token at 97K (results/hip-rocm101-20260917.json, decode_node_timing_97k_20260918).
//
// For a decode-sized batch the selection is small and fixed (indexer_top_k + ratio - 1 = 2051 cells),
// so it is cheaper to materialise it: gather those cells out of the cache into a compact K/V and run
// the ordinary flash-attention kernel over 2051 keys instead of 97512. The selection carries -1 for
// "no cell" (block-graph.inc), which becomes a -inf mask entry, and the gather length is padded to a
// multiple of FATTN_KQ_STRIDE with the same -inf so every kernel variant can take it.
//
// This makes decode sparse, which is what the model intends and what prefill already does; it is not
// bit-identical to the dense decode it replaces.
static ggml_tensor * qwen4exp_gather_attn(ggml_context * ctx0, ggml_tensor * q, ggml_tensor * k,
        ggml_tensor * v, ggml_tensor * top_k, int64_t n_query, float kq_scale) {
    const int64_t head_k  = k->ne[0];
    const int64_t n_head_kv = k->ne[1];
    const int64_t width   = top_k->ne[0];
    const int64_t n_sel   = (width + 255) & ~255;   // FATTN_KQ_STRIDE

    // [head_k, n_head_kv, n_kv] -> [head_k*n_head_kv, n_kv], so a cell is one row
    ggml_tensor * k2d = ggml_view_2d(ctx0, k, head_k*n_head_kv, k->ne[2], k->nb[2], 0);
    ggml_tensor * v2d = ggml_view_2d(ctx0, v, v->ne[0]*v->ne[1], v->ne[2], v->nb[2], 0);

    ggml_tensor * out = nullptr;
    for (int64_t qi = 0; qi < n_query; ++qi) {
        ggml_tensor * idx = ggml_cont(ctx0, ggml_view_2d(ctx0, top_k, width, 1, top_k->nb[1], qi*top_k->nb[1]));

        // shift by one so "no cell" (-1) becomes 0 and the zero padding means the same thing
        ggml_tensor * shifted = ggml_pad(ctx0, ggml_scale_bias(ctx0, ggml_cast(ctx0, idx, GGML_TYPE_F32), 1.0f, 1.0f),
                (int) (n_sel - width), 0, 0, 0);
        ggml_tensor * valid = ggml_step(ctx0, shifted);
        ggml_tensor * rows  = ggml_cast(ctx0, ggml_relu(ctx0, ggml_scale_bias(ctx0, shifted, 1.0f, -1.0f)), GGML_TYPE_I32);
        ggml_tensor * mask  = ggml_cast(ctx0, ggml_log(ctx0, valid), GGML_TYPE_F16);   // 0 where selected, -inf elsewhere
        mask = ggml_cont(ctx0, ggml_reshape_4d(ctx0, mask, n_sel, 1, 1, 1));

        ggml_tensor * kg = ggml_cast(ctx0, ggml_get_rows(ctx0, k2d, rows), GGML_TYPE_F16);
        ggml_tensor * vg = ggml_cast(ctx0, ggml_get_rows(ctx0, v2d, rows), GGML_TYPE_F16);
        kg = ggml_permute(ctx0, ggml_reshape_3d(ctx0, kg, head_k,    n_head_kv, n_sel), 0, 2, 1, 3);
        vg = ggml_permute(ctx0, ggml_reshape_3d(ctx0, vg, v->ne[0], n_head_kv, n_sel), 0, 2, 1, 3);

        // q is [head_k, n_head_q, n_query]; one query at a time, laid out as the kernel wants
        ggml_tensor * q1 = ggml_permute(ctx0, ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], 1,
                q->nb[1], q->nb[2], qi*q->nb[2]), 0, 2, 1, 3);

        ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q1, kg, vg, mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        out = out ? ggml_concat(ctx0, out, cur, 2) : cur;
    }
    return out;
}

'''

BRANCH_OLD = """        ggml_tensor * cur;
        if (direct_indices) {"""
BRANCH_NEW = '''        // strixllama: decode-sized batches gather the selected cells instead of reading the whole cache
        static const bool gather_on = []() {
            const char * e = getenv("LLAMA_QSA_DECODE_GATHER");
            return e != nullptr && atoi(e) != 0;
        }();
        static const int64_t gather_max_t = []() {
            const char * e = getenv("LLAMA_QSA_DECODE_GATHER_MAX_T");
            return e ? (int64_t) atoll(e) : (int64_t) 8;
        }();
        const bool gather = gather_on && n_query <= gather_max_t && n_stream == 1 &&
            cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap &&
            k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && v->nb[1] <= v->nb[2] &&
            q->ne[0] == 256 &&
            // only worth it when the selection is a real reduction: gathering 2304 of 2329 cells costs
            // more than reading them in place (measured 42.9 vs 41.1 ms/token at 2.3K context)
            2*(((top_k->ne[0] + 255) & ~255)) <= k->ne[2];

        ggml_tensor * cur;
        if (gather) {
            ggml_tensor * idx = ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k);
            cur = qwen4exp_gather_attn(ctx0, q, k, v, idx, n_query, kq_scale);
            // same layout the other branches hand to qwen4exp_append_strip: [head_v*n_head_q, n_query]
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
            ggml_build_forward_expand(gf, cur);
        } else if (direct_indices) {'''


def main():
    s = io.open(MODEL, encoding="utf-8").read()
    if "qwen4exp_gather_attn" in s:
        print("already applied")
        return
    if s.count(HELPER_ANCHOR) != 1 or s.count(BRANCH_OLD) != 1:
        sys.exit("anchors not found in %s" % MODEL)
    s = s.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR).replace(BRANCH_OLD, BRANCH_NEW)
    io.open(MODEL, "w", encoding="utf-8", newline="").write(s)
    print("patched", MODEL)


if __name__ == "__main__":
    main()
