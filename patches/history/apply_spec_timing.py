#!/usr/bin/env python3
"""Per-pass CPU-side timing of speculative decoding in llama-server (STRIX_SPEC_TIMING=1).

LLAMA_GRAPH_TIMING shows each backend graph, but not what the server does between them. With
STRIX_SPEC_TIMING=1 the server prints one line per speculative pass, measured at the draft call
and covering the pass before it:

  ST pass=88.5ms n=4 draft=15.8 decode=67.4 process=3.2 sample=0.5 other=1.6

  pass     wall time from one draft call to the next
  n        tokens in the target's verification batch (1 + drafted)
  draft    common_speculative_draft, the draft steps' llama_decode calls included
  decode   the target's llama_decode plus its synchronize
  process  common_speculative_process, i.e. the MTP catch-up decode
  sample   common_sampler_sample_and_accept_n over the verified positions
  other    the rest: accept bookkeeping, seq_rm, process_token, batch assembly

The line above is 85K context on the Q6_K shared head: three draft steps are 18% of a pass, the
target 76%, and target-side sampling of four positions is half a millisecond, which is why the
draft's cost is its graphs, not the CPU sampling around them.

Off by default; no cost when the variable is unset beyond one static bool per call site.

Usage: python patches/apply_spec_timing.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
SRC = os.path.join(tree, "tools", "server", "server-context.cpp")

OLD_DECL = '''constexpr int HTTP_POLLING_SECONDS = 1;
'''

NEW_DECL = '''constexpr int HTTP_POLLING_SECONDS = 1;

// strixllama: CPU-side breakdown of one speculative pass (STRIX_SPEC_TIMING=1). A pass starts at the
// draft call; the line printed there covers the previous pass: the draft (its llama_decode calls
// included), the target decode, the catch-up decode inside common_speculative_process, the target
// sampling in sample_and_accept_n, and whatever is left, which is server bookkeeping.
struct strixllama_spec_timing {
    static bool enabled() { static const bool e = getenv("STRIX_SPEC_TIMING") && atoi(getenv("STRIX_SPEC_TIMING")); return e; }
    int64_t t_pass0 = 0, draft = 0, decode = 0, process = 0, sample = 0;
    int n_batch_tokens = 0;
    void begin_pass() {
        const int64_t now = ggml_time_us();
        if (t_pass0) {
            const int64_t pass = now - t_pass0;
            fprintf(stderr, "ST pass=%.1fms n=%d draft=%.1f decode=%.1f process=%.1f sample=%.1f other=%.1f\\n",
                    pass / 1000.0, n_batch_tokens, draft / 1000.0, decode / 1000.0, process / 1000.0, sample / 1000.0,
                    (pass - draft - decode - process - sample) / 1000.0);
        }
        t_pass0 = now; draft = decode = process = sample = 0; n_batch_tokens = 0;
    }
};
static strixllama_spec_timing g_spec_timing;
'''

OLD_DRAFT = '''        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }'''

NEW_DRAFT = '''        if (!drafting.empty()) {
            const bool st = strixllama_spec_timing::enabled();
            if (st) { g_spec_timing.begin_pass(); }
            const int64_t t0 = st ? ggml_time_us() : 0;
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
            if (st) { g_spec_timing.draft += ggml_time_us() - t0; }
        }'''

OLD_DECODE = '''        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });'''

NEW_DECODE = '''        int ret = 0;
        const bool st = strixllama_spec_timing::enabled();
        const int64_t t_dec0 = st ? ggml_time_us() : 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
        if (st) { g_spec_timing.decode += ggml_time_us() - t_dec0; g_spec_timing.n_batch_tokens += batch_view.n_tokens; }'''

OLD_PROCESS = '''        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });'''

NEW_PROCESS = '''        if (spec) {
            bool ok = true;
            const int64_t t_proc0 = st ? ggml_time_us() : 0;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });
            if (st) { g_spec_timing.process += ggml_time_us() - t_proc0; }'''

OLD_SAMPLE = '''                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                auto accepted = synth_probs.empty()
                    ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)
                    : server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                slot.spec_i_batch.clear();'''

NEW_SAMPLE = '''                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                const int64_t t_smp0 = strixllama_spec_timing::enabled() ? ggml_time_us() : 0;
                auto accepted = synth_probs.empty()
                    ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)
                    : server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                if (t_smp0) { g_spec_timing.sample += ggml_time_us() - t_smp0; }
                slot.spec_i_batch.clear();'''

PAIRS = [(OLD_DECL, NEW_DECL), (OLD_DRAFT, NEW_DRAFT), (OLD_DECODE, NEW_DECODE), (OLD_PROCESS, NEW_PROCESS),
         (OLD_SAMPLE, NEW_SAMPLE)]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    if "STRIX_SPEC_TIMING" in s:
        print("already applied")
        return
    for old, _ in PAIRS:
        if s.count(old) != 1:
            sys.exit("anchor found %d times, expected once, in %s:\n%s" % (s.count(old), SRC, old[:120]))
    for old, new in PAIRS:
        s = s.replace(old, new)
    io.open(SRC, "w", encoding="utf-8", newline="").write(s)
    print("patched", SRC)


if __name__ == "__main__":
    main()
