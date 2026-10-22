#!/usr/bin/env python3
"""Let the MTP draft context use a smaller ubatch than the target.

common_speculative_init copies the target's batch sizes into the draft context, so the draft's
compute buffers scale with the target ubatch. On a unified-memory machine with a large carve that
can fail outright: at ctx 262144 / ubatch 8192 the draft asks for a 3488 MiB buffer and
`cudaMalloc failed: out of memory` kills the load, even though the target itself fits.

STRIX_SPEC_DRAFT_UBATCH=<n> caps only the draft context. Unset keeps upstream behaviour.

Usage: python patches/apply_spec_draft_ubatch.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
SPEC = os.path.join(root, "common", "speculative.cpp")

OLD = """    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);"""

NEW = """    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // strixllama: the draft also inherits the target's batch sizes, so its compute buffers grow with
    // the target ubatch and can exceed the device carve on a unified-memory box. Cap the draft
    // alone; unset leaves the upstream behaviour untouched.
    if (const char * strixllama_draft_ub = getenv("STRIX_SPEC_DRAFT_UBATCH")) {
        const int strixllama_ub = atoi(strixllama_draft_ub);
        if (strixllama_ub > 0 && (uint32_t) strixllama_ub < cparams.n_ubatch) {
            LOG_INF("%s: capping draft ubatch %u -> %d (STRIX_SPEC_DRAFT_UBATCH)\\n",
                    __func__, cparams.n_ubatch, strixllama_ub);
            cparams.n_ubatch = (uint32_t) strixllama_ub;
            if (cparams.n_batch < cparams.n_ubatch) {
                cparams.n_batch = cparams.n_ubatch;
            }
        }
    }"""


def main():
    s = io.open(SPEC, encoding="utf-8").read()
    if "STRIX_SPEC_DRAFT_UBATCH" in s:
        print("already applied")
        return
    if s.count(OLD) != 1:
        sys.exit("anchor not found in %s" % SPEC)
    io.open(SPEC, "w", encoding="utf-8", newline="").write(s.replace(OLD, NEW))
    print("patched", SPEC)


if __name__ == "__main__":
    main()
