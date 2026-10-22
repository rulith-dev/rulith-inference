#!/usr/bin/env python3
"""Per-phase CPU timing inside llama_decode (STRIX_DECODE_TIMING=1).

STRIX_SPEC_TIMING (patches/apply_spec_timing.py) splits a speculative pass into draft / target /
catch-up / sampling, but its "decode" is llama_decode plus llama_synchronize, so it cannot say how
much of that is CPU orchestration and how much is waiting for the GPU. llama_decode is asynchronous:
it returns after submitting. This prints one line per call:

  DT decode n=4 total=10.3ms prep=1.52 graph=1.76 inputs=1.33 compute=5.69 post=0.01 reused=18/40

  prep     entry to the end of output_reserve: balloc->init, memory_update, init_batch, sched_reserve
  graph    per ubatch: mctx->apply, graph_params, can_reuse, and on a miss build_graph + sched_alloc
  inputs   per ubatch: res->set_inputs
  compute  per ubatch: graph_compute - with HIP graphs this is the replay submit, not the GPU
  post     the rest: output reordering and the logits/embeddings/h_nextn readbacks
  reused   ubatches that hit the graph-reuse path, out of ubatches in this call

WHAT IT FOUND (85K real prose, 2026-09-19, speculative pass of ~86 ms)

  target verify decode   total 10.3 ms CPU: prep 1.5, graph 1.8, inputs 1.3, submit 5.7
  draft step (n=1)       total 0.63 ms CPU each
  the other ~58 ms of the target's 69 ms in ST terms is llama_synchronize waiting for the GPU

So CPU orchestration ahead of the submit is 4.6 ms of the pass, ~5%, and none of it overlaps the GPU
(the previous draft step was already synchronised). Two things worth knowing:

- Graph reuse on n=4 decodes is only 45%, because the draft context alternates shapes: catch-up at
  1+n_draft tokens, then draft steps at 1. There is a single gf_res_prev per context, so every switch
  rebuilds. (Fixing that means more than one live llm_graph_result per context, which the scheduler's
  single allocation buffer does not currently allow - not attempted.)
- `inputs` is almost entirely llama_kv_cache::set_input_kq_mask, which evaluates one cell of the
  85K cache per query token. Rows after the first are already optimised upstream (copy the previous
  row, patch the ~32 cells whose positions can differ), so what is left is one full scan, ~0.9 ms.

Off by default; one static bool per call site when off.

Usage: python patches/apply_decode_timing.py [tree]
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")
SRC = os.path.join(tree, "src", "llama-context.cpp")

OLD_DECL = '''#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>'''

NEW_DECL = '''#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// strixllama: where one llama_decode's CPU time goes (STRIX_DECODE_TIMING=1). One line per call:
//
//   DT decode n=4 total=69.1ms prep=0.6 graph=0.3 inputs=1.9 compute=64.8 post=1.5 reused=1/1
//
//   prep     entry to the end of output_reserve: balloc->init, memory_update, init_batch, sched_reserve
//   graph    per ubatch: mctx->apply, graph_params, can_reuse, and on a miss build_graph + sched_alloc
//   inputs   per ubatch: res->set_inputs
//   compute  per ubatch: graph_compute, i.e. the backend (submit + whatever of the GPU it waits for)
//   post     the rest: output reordering and the logits/embeddings/h_nextn readbacks
//   reused   ubatches that hit the graph-reuse path, out of ubatches in this call
//
// Everything but `compute` is CPU orchestration. Off by default; one static bool per call site when off.
struct strixllama_decode_timing {
    static bool enabled() { static const bool e = getenv("STRIX_DECODE_TIMING") && atoi(getenv("STRIX_DECODE_TIMING")); return e; }
    int64_t prep = 0, graph = 0, inputs = 0, compute = 0;
    int n_ubatch = 0, n_reused = 0;
    void reset() { prep = graph = inputs = compute = 0; n_ubatch = n_reused = 0; }
};
static strixllama_decode_timing g_decode_timing;'''

OLD_PU = '''llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {'''

NEW_PU = '''llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const bool dt = strixllama_decode_timing::enabled();
    const int64_t dt_t0 = dt ? ggml_time_us() : 0;

    if (mctx && !mctx->apply()) {'''

OLD_REUSE = '''        ggml_backend_sched_prepare_inputs(sched.get());

        n_reused++;
    } else {'''

NEW_REUSE = '''        ggml_backend_sched_prepare_inputs(sched.get());

        n_reused++;
        if (dt) { g_decode_timing.n_reused++; }
    } else {'''

OLD_INPUTS = '''    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {'''

NEW_INPUTS = '''    const int64_t dt_t1 = dt ? ggml_time_us() : 0;

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const int64_t dt_t2 = dt ? ggml_time_us() : 0;

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (dt) {
        const int64_t dt_t3 = ggml_time_us();
        g_decode_timing.graph   += dt_t1 - dt_t0;
        g_decode_timing.inputs  += dt_t2 - dt_t1;
        g_decode_timing.compute += dt_t3 - dt_t2;
        g_decode_timing.n_ubatch++;
    }
    if (status != GGML_STATUS_SUCCESS) {'''

OLD_ENTRY = '''    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;'''

NEW_ENTRY = '''    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\\n", __func__);
        return -1;
    }

    const bool    dt    = strixllama_decode_timing::enabled();
    const int64_t dt_t0 = dt ? ggml_time_us() : 0;
    if (dt) { g_decode_timing.reset(); }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;'''

OLD_PREP = '''    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;'''

NEW_PREP = '''    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    if (dt) { g_decode_timing.prep = ggml_time_us() - dt_t0; }

    int64_t n_outputs_prev = 0;'''

OLD_END = '''    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}'''

NEW_END = '''    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    if (dt) {
        const auto & g = g_decode_timing;
        const int64_t total = ggml_time_us() - dt_t0;
        fprintf(stderr, "DT decode n=%d total=%.1fms prep=%.2f graph=%.2f inputs=%.2f compute=%.2f post=%.2f reused=%d/%d\\n",
                (int) batch_inp.n_tokens, total / 1000.0, g.prep / 1000.0, g.graph / 1000.0, g.inputs / 1000.0,
                g.compute / 1000.0, (total - g.prep - g.graph - g.inputs - g.compute) / 1000.0,
                g.n_reused, g.n_ubatch);
    }

    return 0;
}'''

PAIRS = [(OLD_DECL, NEW_DECL), (OLD_PU, NEW_PU), (OLD_REUSE, NEW_REUSE), (OLD_INPUTS, NEW_INPUTS),
         (OLD_ENTRY, NEW_ENTRY), (OLD_PREP, NEW_PREP), (OLD_END, NEW_END)]


def main():
    s = io.open(SRC, encoding="utf-8").read()
    if "STRIX_DECODE_TIMING" in s:
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
