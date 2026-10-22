# Elementwise chain fusion (HIP backend)

Sources: `strixllama-chain.cu`, `strixllama-chain.cuh` (copies of the vendored files), wired in by
`patches/apply_chain_fusion.py`.

## What it does

At batch 1 the qwen4exp decode graph is ~2300 dispatches of which ~1900 do no matrix work, and many
of those are short runs of elementwise ops on tiny tensors: `SCALE -> SIGMOID -> SCALE` on 4
floats, `ADD -> SOFTPLUS -> MUL` on 48, `SIGMOID -> MUL -> ADD` on the shared-expert gate,
`CLAMP -> DIV` on the router weights, `MUL -> SIGMOID -> MUL` on the PLE gate. Each was its own
launch. `ggml_cuda_strixllama_chain_try` (called at the end of `ggml_cuda_try_fuse`, after every
specific pattern) collects a maximal run of consecutive SCALE / CLAMP / UNARY / ADD / SUB / MUL /
DIV nodes (F32, contiguous) whose intermediates have a single consumer and launches one kernel
that evaluates the same operations in the same order per output element (modulo-broadcast reads,
`fc_read`). RESHAPE / contiguous VIEW aliases between chain nodes are seen through.

## The REPEAT rule (the bug that cost a day)

A REPEAT operand is folded into a broadcast read of its source only when the REPEAT node itself
lies inside the chain and is being skipped. A REPEAT that ran before the chain is read as is. The
PLE gating chain `MUL(sgn, mag) -> SIGMOID -> MUL(repeat(value), gate)` has its REPEAT before the
chain; folding it read `value` after ggml-alloc had ended `value`'s lifetime at the REPEAT and
handed the memory to later tensors, corrupting `ple_gated_value` in a way the token-level test
caught but that looked like correct math on paper.

Other guards: an operand must not be rooted in the chain (single-use intermediates only), and no
read may overlap the output unless it is exactly the output (same pointer, shape, contiguous:
each thread reads its own element before writing it).

## Verification

- `STRIX_CHAIN_VERIFY=1 GGML_CUDA_DISABLE_GRAPHS=1`: every fused chain is recomputed node by
  node into its real buffers (operands snapshotted and restored around it, the compute stream
  synchronised), then the fused kernel runs and the two outputs are compared element-wise.
  Result on a 32-token prefill + 8 decode tokens: 0 mismatching chains.
- `tools/spec_ab.py --runtime bin/hip-rocm101-chain --set mtp=false ngram_spec=false
  --ref logs/spec-off.txt`: 600 greedy tokens byte-identical to the unfused runtime.

## Measured (2026-09-18, spec off)

`tools/decode_probe.py --n 160`, two alternating rounds (ms/token):

| runtime      | round 1                  | round 2                  |
|--------------|--------------------------|--------------------------|
| production   | 40.99 40.04 41.81 41.20  | 39.67 38.91 39.93 38.62  |
| chain fusion | 40.79 39.28 40.12 39.39  | 40.19 39.22 39.78 39.01  |

llama-bench tg128 x5, unified memory off, two rounds: production 24.60 / 24.90 tok/s, chain
runtime 24.97 / 25.27 (the chain runtime also carries the IQ3_S vec-dot HIP branch, which alone
measures 25.10 / 25.41). So the chain fusion itself is worth at most ~1 ms/token and is not
separable from run-to-run drift; it is kept because it is exact, bounded, and removes a class of
launches that any further dispatch work builds on. Unbounded it cost 3% of a 55K-token prefill
(968 -> 932 t/s); with the 262144-element bound prefill is neutral (936-940 vs 942-946 t/s).
Perplexity ctx 2048: 2.3786, same as production.

## Switches

- `STRIX_CHAIN_FUSE=0` disables the fusion.
- `STRIX_CHAIN_ONLY=FIRSTOP-LASTOP` keeps only chains with that signature (bisection).
- `STRIX_CHAIN_NO_REPEAT=1` ends chains at REPEAT nodes.
- `LLAMA_GRAPH_TRACE=1` prints `CHAIN_FUSE: n nodes [OP 'name' .. OP 'name'] in=[..] out=[..]`.
