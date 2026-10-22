# IQ3_S MMB kernel — the largest measured win, and why it is a whole-file drop

Five files that `bootstrap.py` copies over upstream before any patch script runs. They are **live in
every build**: `SNAPSHOT` in `bootstrap/bootstrap.py` lays them down, and the scripts that follow
edit `ggml-cuda.cu` on top of this version of it.

They are whole files rather than an `apply_*.py` because the change is a new kernel plus its
templating, not an edit that anchors cleanly — and because several later scripts anchor against text
that only exists once these are in place. `bootstrap/UPSTREAM.json` records what the result must hash
to, so "whole file" does not mean "unverifiable": `tools/replay_bootstrap.py` reproduces all 24 files
of the delta from clean upstream, these included.

## What it does

The MMB path (the bf16 WMMA dequant GEMM that carries prefill on gfx1151) gated on IQ4_NL, Q8_0 and
Q6_K. Unsloth's UD-IQ4_XS is a mixed quant whose body — 64.9 GB excluding the 28.8 GB PLE table — is
**52.2% IQ3_S**: 94 tensors, exactly `blk.N.ffn_gate_exps` ×47 and `blk.N.ffn_up_exps` ×47. Those two
are the inputs of the fused GLU mmid kernel, so one kernel covers all of them; `ffn_down_exps` was
already IQ4_NL. Before this, half the model body never reached the matrix cores and fell back to the
stock MMQ DP4A path.

- `mmb.cu` — `mmb_dq_row_iq3s` dequantizes 64 weights, a quarter of a 110-byte 256-weight
  superblock: `d` at 0, 16 `qs` bytes at 2+16q, 2 `qh` at 66+2q, 8 sign bytes at 74+8q, and one
  `scales` byte at 106+q whose two nibbles split the halves. `mmb_tile_gemm_glu` and
  `mmb_routed_glu_kernel` are templated on `WT` (0 = IQ4_NL, 3 = IQ3_S), and `load_regs` prefetches
  those 29 bytes into registers exactly as the IQ4_NL path does. Adding that prefetch was worth
  +2.0% over the first version and halved the run-to-run spread.
- `mmb.cuh` — `ggml_cuda_mmb_supported_mmid` grew an `allow_iq3s` flag defaulting to **false**,
  because the plain mmid kernel still assumes the 36-byte IQ4_NL step. Only the fused-GLU caller
  passes true.
- `softcap.cu` / `.cuh` — `ggml_cuda_op_scale_sigmoid`, a separate experiment: upstream fuses
  SCALE→UNARY→SCALE only for TANH, and a decode pass here dispatches 195 SIGMOID and zero TANH.
- `ggml-cuda.cu` — the fusion rules, plus `LLAMA_GRAPH_TRACE` / `LLAMA_GRAPH_DUMP` instrumentation
  (graph usage, re-capture rate, a dispatched-kernel census, a node dump). All behind `getenv`,
  inert unless set.

`STRIX_MMB_IQ3S=0` sends IQ3_S weights back to the stock MMQ path — the A/B switch, and the thing to
reach for if a future model file makes this kernel suspect.

## Measured

Both arms on the current runtime, same binary, verified by `type=IQ3_S` appearing in one server log
and not the other. `tools/truncation_test.py --fill 70000`, 30 numbered sections at temperature 0:

| | prefill | answer | sections |
|---|---|---|---|
| IQ3_S on | **876.7 t/s** | 1110 tokens, `finish_reason=stop` | **30/30** |
| IQ3_S off | 830.6 t/s | 1085 tokens, `finish_reason=stop` | **30/30** |

+5.6% prefill, quality indistinguishable. Perplexity at ctx 2048 is 2.3786 against a 2.3840 baseline,
and at ctx 4096 (where sparse attention is active) 2.4646 / 2.4741 at ubatch 4 / 512 — production
exactly.

## It was once blamed for truncation, wrongly

This kernel shipped on 2026-09-17, was blamed the same night for long answers stopping mid-list at
depth, and was rolled back. The real cause was found on 09-18 in a different place: speculative
verification batches ran dense attention with **no causal mask**, so the target could see the draft
tokens it was meant to be checking (`apply_qsa_small_batch_mask`). Every HIP build had it. Re-tested
on the fixed runtime, both arms reach 30/30 — the table above.

Two things are worth keeping from that episode:

- **The gate that let it through was a ctx-2048 perplexity pass.** Perplexity over 2048-token chunks
  cannot see a deep-context regression, and it reported 2.3786 in every variant including the broken
  one. Any numerics kernel has to pass `truncation_test.py --fill` at a context where sparse
  attention is active, at small *and* large ubatch.
- **A rollback note is a claim with an expiry date.** This file said "It is not in production" for a
  day after the kernel was back in it, which is how the project ended up with its largest measured
  win documented as reverted. `patches/MANIFEST.md` records the three independent proofs that
  contradicted it.
