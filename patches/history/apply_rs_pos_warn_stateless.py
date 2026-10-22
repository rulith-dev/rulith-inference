#!/usr/bin/env python3
"""Silence the recurrent memory's position warning when it holds no state.

llama_memory_recurrent::find_slot warns "non-consecutive token position X after Y" whenever a batch
does not continue from the cell's last position. That is a real diagnostic for a memory with
recurrent state: the state cannot be rewound mid-batch. With LLAMA_MTP_QSA=1 the MTP draft context
is a llama_memory_hybrid_idx whose recurrent part filters out every layer ("the nextn layer is not
recurrent"), so it keeps no state at all - but its cell bookkeeping still runs, and speculative
decoding backtracks the draft on every pass (draft step 1 after a catch-up that covered rejected
positions, the catch-up after three draft steps). Result: one or two of these lines per pass, 1174 in
one 1758-token reply in Jan's developer log, all from the draft, all meaningless.

The memory now counts the layers it actually allocated state for and only warns when there are
any. The target's recurrent memory (48 layers with state) keeps the warning; its rollbacks go through
seq_rm's snapshot index, which updates the cell position, so it does not fire there.

Usage: python patches/apply_rs_pos_warn_stateless.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
HDR = os.path.join(tree, "src", "llama-memory-recurrent.h")
SRC = os.path.join(tree, "src", "llama-memory-recurrent.cpp")

OLD_HDR = '''    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;
'''

NEW_HDR = '''    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // strixllama: layers this memory actually keeps state for; the layer filter can leave none (the MTP
    // draft's hybrid-idx memory), and then the cell position bookkeeping has nothing to protect
    uint32_t n_layer_recr = 0;
'''

OLD_CTOR = '''        r_l[i] = r;
        s_l[i] = s;
'''

NEW_CTOR = '''        r_l[i] = r;
        s_l[i] = s;
        n_layer_recr++;
'''

OLD_WARN = '''        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\\n",'''

NEW_WARN = '''        // strixllama: a memory with no state (every layer filtered out) has nothing that a backtrack could
        // corrupt, and the MTP draft backtracks on every speculative pass
        if (n_layer_recr > 0 && cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\\n",'''

PAIRS = [(HDR, OLD_HDR, NEW_HDR), (SRC, OLD_CTOR, NEW_CTOR), (SRC, OLD_WARN, NEW_WARN)]


def main():
    texts = {p: io.open(p, encoding="utf-8").read() for p in (HDR, SRC)}
    if "n_layer_recr" in texts[HDR]:
        print("already applied")
        return
    for path, old, _ in PAIRS:
        if texts[path].count(old) != 1:
            sys.exit("anchor found %d times, expected once, in %s:\n%s" % (texts[path].count(old), path, old[:120]))
    for path, old, new in PAIRS:
        texts[path] = texts[path].replace(old, new)
    for path, s in texts.items():
        io.open(path, "w", encoding="utf-8", newline="").write(s)
    print("patched", HDR, "and", SRC)


if __name__ == "__main__":
    main()
