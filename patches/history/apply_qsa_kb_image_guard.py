#!/usr/bin/env python3
"""Keep the QSA block-key cache out of graphs whose memory holds image cells.

With the vision projector loaded, any image in a context long enough for QSA block selection aborted
the server:

    llama-memory-hybrid-idx.cpp:660
    GGML_ASSERT(n_ns == 1 && !ranked && "qsa block-key cache: single stream, 1-D positions") failed

M-RoPE repeats one position across an image, so set_input_qsa sorts cells by rank instead of using
the position (`ranked`), and the block-key cache's staleness tracking is keyed on 1-D positions.

The graph already had a guard - qwen4exp.cpp's `qwen4exp_pos_scalar(ubatch)` - but it asks whether
*this ubatch's* position axes agree, which is true for a plain text ubatch. `ranked` fires on
`dup && ubatch->is_pos_2d()`, where `dup` comes from image cells already in the cache and
`is_pos_2d()` is `n_pos >= 3`, a model constant (4 for M-RoPE) that is always true here. So a text
turn after an image still took the cache path and aborted.

A SECOND abort, same area, is fixed here too:

    llama-memory-hybrid-idx.cpp:616
    GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window") failed

`oor` means a cell's position/ratio lands past the block window. M-RoPE makes positions and cells
drift apart, so this is reachable exactly when images are involved. The rank the `dup` path already
builds is dense by construction, so ranking fixes `oor` as well - hence `(dup || oor)`.

llama_memory_hybrid_idx now decides, in init_batch, whether the graph may wire the cache in
(kb_pos_dup()), setting the flag on either trigger for ranking:
  - an image ubatch (positions differ per axis), or
  - a position gap, i.e. a ubatch that does not continue the sequence.
The gap half is not optional. The MTP draft context is a hybrid-idx memory too, and
common_speculative_impl_draft_mtp::process() skips embedding batches, so the draft never stores an
image's tokens: no image ubatch ever reaches its init_batch, its cells get a hole, and every later
position sits past its cell. Without gap detection the draft keeps the cache wired, takes the ranked
path and aborts - which is what a multi-turn image conversation does within a few turns.

can_reuse compares the flag recorded at build time, so a graph built before an image is rebuilt
rather than reused after one.

The flag clears ONLY when every sequence is dropped (seq_id < 0). A whole-sequence seq_rm is not
enough: the server calls that when it reuses a slot by longest-common-prefix and then keeps the
cached prefix, image cells included. Clearing there re-wires the cache under ranked cells and aborts;
that failure was reproduced directly before the condition was narrowed back.

Scoped to the conversation, which is the point: a text-only chat keeps the cache. Measured at 85K on
real prose, same binary and prompt, acceptance 73% (tools/decode_lab.py --config spectime, n=4
passes, sigma ~8 ms):

    cache on, pre-guard    pass 85.1 ms, target decode 68.6   28.27 ms/token   35.37 tok/s
    with this guard        pass 87.7 ms, target decode 70.7   29.11 ms/token   34.36 tok/s
    cache off              pass 91.0 ms, target decode 74.2   30.69 ms/token   32.59 tok/s

so most of the cache's 6.5% is retained; whether the remaining 2.6 ms is the draft's own cache going
off on rollback gaps or run-to-run drift is not resolved at this sample size.

Verified with vision on: image at 85K answers correctly; a text turn in the same conversation
afterwards answers correctly; tmp/vis_stress.py (7 turns growing to 15340 tokens with an image, then
two divergent branches that force LCP slot reuse) completes with zero asserts.

Usage: python patches/apply_qsa_kb_image_guard.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
HDR = os.path.join(tree, "src", "llama-memory-hybrid-idx.h")
MEM = os.path.join(tree, "src", "llama-memory-hybrid-idx.cpp")
MDL = os.path.join(tree, "src", "models", "qwen4exp.cpp")

OLD_HDR_API = '''    bool          kb_needs_full() const;      // positions moved (shift, restore, clear) since the last full write
    void          kb_mark_full() const;

private:'''

NEW_HDR_API = '''    bool          kb_needs_full() const;      // positions moved (shift, restore, clear) since the last full write
    void          kb_mark_full() const;
    // strixllama: true once a ubatch with per-axis positions (an image under M-RoPE) has been written.
    // Such cells repeat one position across the image, so set_input_qsa ranks cells instead of using
    // the position, which the block-key cache cannot track - the graph must not wire the cache in.
    bool          kb_pos_dup() const;

private:'''

OLD_HDR_MEM = '''    uint64_t                             kb_gen      = 1;
    mutable uint64_t                     kb_full_gen = 0;'''

NEW_HDR_MEM = '''    uint64_t                             kb_gen      = 1;
    mutable uint64_t                     kb_full_gen = 0;
    // set in init_batch when an image ubatch or a position gap arrives, cleared only when every
    // sequence is dropped - see kb_pos_dup() and the reset in seq_rm()
    bool                                 kb_dup      = false;'''

OLD_HDR_CTX = '''    ggml_tensor * get_kb(int32_t il) const;
    uint32_t      kb_scratch_row() const;
    bool          kb_needs_full() const;
    void          kb_mark_full() const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;'''

NEW_HDR_CTX = '''    ggml_tensor * get_kb(int32_t il) const;
    uint32_t      kb_scratch_row() const;
    bool          kb_needs_full() const;
    void          kb_mark_full() const;
    bool          kb_pos_dup() const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;'''

OLD_MEM_API = '''void llama_memory_hybrid_idx::kb_mark_full() const {
    kb_full_gen = kb_gen;
}'''

NEW_MEM_API = '''void llama_memory_hybrid_idx::kb_mark_full() const {
    kb_full_gen = kb_gen;
}

bool llama_memory_hybrid_idx::kb_pos_dup() const {
    return kb_dup;
}

// strixllama: does this ubatch carry a position per axis, i.e. an image under M-RoPE? A text token has
// the same value on every axis (the whole batch is then "position scalar"), an image token does not,
// and the cells it writes repeat one position across the image. Mirrors qwen4exp_pos_scalar().
static bool hybrid_idx_ubatch_pos_dup(const llama_ubatch & ubatch) {
    if (!ubatch.pos || ubatch.n_tokens == 0) {
        return false;
    }
    for (uint32_t axis = 1; axis < ubatch.n_pos; ++axis) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i + axis*ubatch.n_tokens] != ubatch.pos[i]) {
                return true;
            }
        }
    }
    return false;
}'''

OLD_MEM_SPLIT = '''        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }'''

NEW_MEM_SPLIT = '''        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // strixllama: decide here whether the block-key cache may be wired into the graph built from this
        // context (see kb_pos_dup), because set_input_qsa will then rank cells instead of using their
        // position, and the cache cannot track ranked cells. Two ways to get there, both sticky:
        //   - an image ubatch: M-RoPE gives it a position per axis and repeats one position across the
        //     image, so cells share positions (`dup`);
        //   - a gap: this ubatch does not continue the sequence, so cells and positions drift apart and
        //     a cell can land outside the block window (`oor`). The MTP draft is where this bites -
        //     common_speculative_impl_draft_mtp::process() skips embedding batches, so the draft never
        //     stores the image's tokens and every later position is shifted past its cell.
        {
            std::map<llama_seq_id, llama_pos> next_pos;
            for (const auto & ub : ubatches) {
                if (hybrid_idx_ubatch_pos_dup(ub)) {
                    kb_dup = true;
                    break;
                }
                if (ub.n_tokens == 0 || !ub.pos || !ub.seq_id || !ub.seq_id[0]) {
                    continue;
                }
                const llama_seq_id s  = ub.seq_id[0][0];
                const auto         it = next_pos.find(s);
                const llama_pos    expect = it != next_pos.end()
                    ? it->second
                    : get_mem_attn()->seq_pos_max(s) + 1;
                if (ub.pos[0] > expect) {
                    kb_dup = true;
                    break;
                }
                next_pos[s] = ub.pos[ub.n_tokens - 1] + 1;
            }
        }'''

OLD_MEM_CLEAR = '''void llama_memory_hybrid_idx::clear(bool data) {
    kb_gen++;   // strixllama: block keys depend on positions and cell contents; rebuild them all once
    llama_memory_hybrid::clear(data);'''

NEW_MEM_CLEAR = '''void llama_memory_hybrid_idx::clear(bool data) {
    kb_gen++;   // strixllama: block keys depend on positions and cell contents; rebuild them all once
    kb_dup = false;   // strixllama: no cells left, so no image cells either
    llama_memory_hybrid::clear(data);'''

OLD_MEM_RM = '''bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }'''

NEW_MEM_RM = '''bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    // strixllama: only a request that drops every sequence is proof the image cells are gone. A whole-
    // sequence seq_rm is NOT: the server calls it when it reuses a slot by longest-common-prefix and
    // then keeps the cached prefix, image cells included, so clearing the flag there wires the cache
    // back in under ranked cells and aborts. Measured that failure directly (tmp/vis_stress.py).
    if (seq_id < 0 && p0 <= 0 && p1 < 0) {
        kb_dup = false;
    }'''

OLD_MEM_CTX = '''bool llama_memory_hybrid_idx_context::kb_needs_full() const {
    return mem ? mem->kb_needs_full() : false;
}'''

NEW_MEM_CTX = '''bool llama_memory_hybrid_idx_context::kb_needs_full() const {
    return mem ? mem->kb_needs_full() : false;
}

bool llama_memory_hybrid_idx_context::kb_pos_dup() const {
    return mem ? mem->kb_pos_dup() : false;
}'''

OLD_MDL_REUSE = '''        if (kb_bid_rows) {
            res &= kb_full == mctx->kb_needs_full();'''

NEW_MDL_REUSE = '''        // the block-key cache is wired in only while no image cells are present, so a graph built
        // before one arrived cannot be reused after (it would take the ranked path and abort)
        res &= kb_dup == mctx->kb_pos_dup();
        if (kb_bid_rows) {
            res &= kb_full == mctx->kb_needs_full();'''

OLD_MDL_FIELD = '''    ggml_tensor * kb_bid_rows    = nullptr;   // I32 [n_blocks]
    bool          kb_full        = false;'''

NEW_MDL_FIELD = '''    ggml_tensor * kb_bid_rows    = nullptr;   // I32 [n_blocks]
    bool          kb_full        = false;
    bool          kb_dup         = false;     // the memory held image cells when this graph was built'''

OLD_MDL_GUARD = '''        // strixllama: block-key cache inputs (llama_memory_hybrid_idx::qsa_kb_inputs); single stream, 1-D positions only
        if (mctx_hyb->get_kb(il) != nullptr && n_stream == 1 && qwen4exp_pos_scalar(ubatch)) {'''

NEW_MDL_GUARD = '''        // strixllama: block-key cache inputs (llama_memory_hybrid_idx::qsa_kb_inputs); single stream, 1-D
        // positions only. qwen4exp_pos_scalar covers this ubatch; kb_pos_dup covers the cells already
        // in the cache, because an image written earlier makes set_input_qsa rank cells even for a
        // later text ubatch, and the cache cannot track ranked cells (it asserts !ranked).
        qsa->kb_dup = mctx_hyb->kb_pos_dup();
        if (mctx_hyb->get_kb(il) != nullptr && n_stream == 1 && qwen4exp_pos_scalar(ubatch) &&
                !qsa->kb_dup) {'''

OLD_MEM_INC = '''#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>'''

NEW_MEM_INC = '''#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>'''

OLD_MEM_RANK = '''        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {'''

NEW_MEM_RANK = '''        // mrope repeats one position across an image, so rank cells instead of using the position.
        // strixllama: `oor` needs the same treatment. M-RoPE also advances the position past an image by
        // its grid extent rather than by its token count, so positions run ahead of the cells holding
        // them and a cell can land outside the block window - the rank is dense by construction, so
        // ranking fixes that too. Without it the assert below aborts the server on the first long
        // enough conversation that contains an image.
        if ((dup || oor) && ubatch->is_pos_2d() && one_seq) {'''

PAIRS = [
    (HDR, OLD_HDR_API,   NEW_HDR_API),
    (HDR, OLD_HDR_MEM,   NEW_HDR_MEM),
    (HDR, OLD_HDR_CTX,   NEW_HDR_CTX),
    (MEM, OLD_MEM_INC,   NEW_MEM_INC),
    (MEM, OLD_MEM_API,   NEW_MEM_API),
    (MEM, OLD_MEM_RANK,  NEW_MEM_RANK),
    (MEM, OLD_MEM_SPLIT, NEW_MEM_SPLIT),
    (MEM, OLD_MEM_CLEAR, NEW_MEM_CLEAR),
    (MEM, OLD_MEM_RM,    NEW_MEM_RM),
    (MEM, OLD_MEM_CTX,   NEW_MEM_CTX),
    (MDL, OLD_MDL_REUSE, NEW_MDL_REUSE),
    (MDL, OLD_MDL_FIELD, NEW_MDL_FIELD),
    (MDL, OLD_MDL_GUARD, NEW_MDL_GUARD),
]


def main():
    texts = {p: io.open(p, encoding="utf-8").read() for p in (HDR, MEM, MDL)}
    if "kb_pos_dup" in texts[HDR]:
        print("already applied")
        return
    for path, old, _ in PAIRS:
        if texts[path].count(old) != 1:
            sys.exit("anchor found %d times, expected once, in %s:\n%s" % (texts[path].count(old), path, old[:140]))
    for path, old, new in PAIRS:
        texts[path] = texts[path].replace(old, new)
    for path, s in texts.items():
        io.open(path, "w", encoding="utf-8", newline="").write(s)
    print("patched", HDR, MEM, "and", MDL)


if __name__ == "__main__":
    main()
