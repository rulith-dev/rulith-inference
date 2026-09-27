# Measured results

Everything here was measured on the target machine, against the production runtime, with the command
shown. Where a number moved for a reason other than the change being tested — usually draft
acceptance — it says so. See [measuring.md](measuring.md) before comparing any two of these.

## Machine

| | |
|---|---|
| CPU / GPU | Ryzen AI Max+ 395, Radeon 8060S (gfx1151) |
| Memory | 128 GB unified, **96 GB carved to the GPU** (Windows is left 31.6 GB) |
| Power | 140 W |
| OS | Windows 11 |
| Runtime | `pwilkin/llama.cpp` @ `f5daaa3` + this patch set, built against TheRock ROCm 10.2 (10.1 up to 0.2.2) |
| Model | Qwen3.8-Flash-Next 125B-A6B, Unsloth UD-IQ4_XS, 93.7 GB |
| Server flags | ctx 262144, batch/ubatch 8192, `-np 1`, flash attention on, KV f16, MTP draft, `--load-mode none --lazy-mode on-direct` |

The carve is not a detail. At 64 GB the model does not fit, ~9.8 GB spills into shared memory, and
the driver pages during decode: prefill 839 vs 942 t/s, decode 49.6 vs 38.6 ms/token, multi-turn
17-19 vs 26-30 tok/s — and that comparison was already in the 64 GB arm's favour, at 140 W against
120 W. **Set the carve before tuning anything else.**

## Headline

Measured 2026-09-28 on 0.3.0, at temperature 0, `cache_prompt: false` so each run pays a full prefill,
**on a freshly started server** (see the image note below — that qualifier is load-bearing), with the
model files of [the README's list](../README.md#model-files) and the app's default settings (eight
slots, MTP with three drafts, images on, f16 K/V):

| | | |
|---|---|---|
| prefill, 95.6K tokens of real text | **1217 t/s** | 1215.8 / 1217.3 / 1221.4 over 3 runs |
| decode, 86K context | **28.0 ms/token** (35.8 tok/s) | 28.04 / 27.95 / 27.80, draft acceptance 63%, 2.84 tokens per pass |
| decode, short context | **22.3 ms/token** (44.9 tok/s) | 22.29 / 23.36 / 22.23, acceptance 67%, 2.94 tokens per pass |
| decode, 3 / 4 conversations at once | **58.8 / 62.6 tok/s** summed | four slots, ~4K tokens each, default sampling with a new seed every round, nine rounds each; one conversation 40.9 |
| image input | works | Qwen3-VL projector, 904 MB |

Against 0.2.3 on the same protocol (2026-09-26, the rows below): prefill 1183 -> 1217 t/s, decode at 86K
30.7 -> 28.0 ms/token and at short context 24.4 -> 22.3 at the same acceptance and tokens a pass - the
0.2.5-0.2.9 rounds (the prefix cache, GDN rollback without snapshots, ragged ubatches, the GDN and norm
kernels). Several conversations: 53.6 / 61.4 -> 58.8 / 62.6 summed, while one conversation measured the
same way went 35.8 -> 40.9, so the ratios read 1.44× / 1.53× instead of 1.50× / 1.71×. Details:
`docs/results/stable-030-20260928.json`. The 2026-09-26 table:

| | | |
|---|---|---|
| prefill, 95.6K tokens of real text | 1183 t/s | 1191.2 / 1167.4 / 1182.8 over 3 runs |
| decode, 86K context | 30.7 ms/token (32.6 tok/s) | 32.14 / 30.65 / 29.81, draft acceptance 63%, 2.84 tokens per pass |
| decode, short context | 24.4 ms/token (41.0 tok/s) | 25.31 / 24.38 / 24.27, acceptance 66%, 2.90 tokens per pass |
| decode, 3 / 4 conversations at once | 53.6 / 61.4 tok/s summed | 0.2.4, ~4K tokens each, default sampling with a new seed every round, nine rounds each |

The first run of each decode group is a warm-up: 32.14 against 30.2 for the other two, 25.31 against
24.3. The prefill figures need no such caveat — runs land within 1-3% of each other.

The text is llama.cpp's own docs, tool READMEs and `src/llama-*.cpp` concatenated; 340,000 characters of
it are 95,582 tokens, 302,000 are ~86K:

- prefill: `tools/decode_lab.py --config dectime --words 700 --n 4 --gen 16 --gen-prefix <text> --gen-prefix-chars 340000`
- decode at 86K: `tools/decode_lab.py --config base --words 700 --n 4 --gen 400 --gen-prefix <text> --gen-prefix-chars 302000`
- decode at short context: the same without `--gen-prefix`, so the prompt is the lab's one-line question
- 0.3.0 ran each with `--runtime <build> --port 8091 --model ba09dd01dc9c5fd1df9d`: the lab's default model is
  the catalog's first, which on this machine now sorts to another model, and until 0.3.0 it launched on the
  production port (8080) and stopped the app's server whatever `--port` said

Earlier prefill on the same text: 0.2.0 1187.4 / 1183.7 / 1200.6 (2026-09-25), 0.1.17 999.1 / 965.5 /
992.4, 0.1.9 982.1 / 979.6 / 988.7 (2026-09-23), 0.1.8 888.3 / 892.9 / 889.7, and the 2026-09-19 build
886.2 / 883.7 / 887.6 on 85K tokens of prose. Decode at 86K on the same text: 0.2.0 29.10 / 29.67 / 29.48
ms/token at 62% acceptance, so 0.2.3 is ~2-3% slower a step at depth (the truncation gate at 77K agrees:
84.0 ms a pass against 0.2.2's 82.4); not yet traced. Until 2026-09-26 the decode rows were the
2026-09-19 build's on Chinese prose, where drafts are accepted more often: 28.7 ms/token (34.9 tok/s)
at 85K with 68-69% acceptance, and 26.8 ms/token (37.4 tok/s) at short context on the lab's random-word
probe at 60%. The row of several conversations is 0.2.4's side of the 0.2.3 / 0.2.4 A/B (below), with
the server's default sampling, a new seed every round and the prompts cached; one conversation gets 35.8
tok/s the same way, so three give 1.50× and four 1.71×. One conversation drafts three tokens and has 60%
of them accepted, three or four draft two and have 65-69% accepted, and a single round moves by ±10%
with the acceptance of the text it samples. Until 0.2.4 the row read 55.7 / 60.4 (1.47× / 1.59×), from
0.2.3's A/B with a fixed seed per conversation, which flattered 0.2.3 (below). Details:
`docs/results/headline-20260926.json`, and for the row of several conversations
`docs/results/verify-small-products-20260926.json`.

### A slot that has served an image decodes ~7% slower until it is cleared

Measured on one server, same prompt, before and after running `tools/vision_stress.py`:

| | decode at 85K | acceptance |
|---|---|---|
| fresh server | 28.7 ms/token | 68-69%, 2.99 per pass |
| after an image conversation | 30.3 ms/token | 67-69%, 2.97 per pass |

Acceptance is identical, so this is a real per-pass cost and not the speculation lottery. The cause
is the image guard: M-RoPE decouples cell indices from positions, which the block-key cache's
staleness tracking cannot represent, so the cache is switched off for any memory that has seen an
image — and it stays off until a **full** clear, because an ordinary whole-sequence removal also
happens during slot reuse while the cached prefix is deliberately kept. The block-key cache is worth
about 6.5%, which is exactly what this costs.

This is the right trade (without it, images abort the server outright), but it means **a benchmark
run on a server that has served an image is not comparable to one that has not.** An earlier draft of
this file reported 32.1 ms/token at 85K for precisely that reason.

The command, against the running production server — an 85K-token prefix of real prose followed by a
question, which is what makes the draft-acceptance figure mean anything (see below):

```bash
python tools/decode_lab.py --keep --words 700,28800 --n 64
```

`--keep` probes whatever server is already listening rather than launching one, so this measures
production itself. The depth probe uses random words; for the acceptance-sensitive numbers above,
point `--gen-prefix` at a long document of your own and use `--gen`:

```bash
python tools/decode_lab.py --config base --gen 400 --gen-prefix <a long document> --gen-prefix-chars 85000
```

**`--gen-prefix-chars` is characters, not tokens**, and the ratio is entirely
language-dependent — the corpus used here is Chinese and runs at almost exactly 1 token per
character, where English prose is nearer 1 per 4. Ask the server (`POST /tokenize`) instead of
assuming; a prefix meant to be 85K tokens was briefly 340K, which the server rejected outright.

## How decode got from 21.5 to ~31 tok/s at 85K

In order, each measured when it landed. These are not additive — later entries were measured on top
of earlier ones, and acceptance drifts between them.

| change | effect |
|---|---|
| QSA block-key cache | 97K decode 63.5 → 58.5 ms/token; context slope 0.241 → 0.182 ms per 1000 tokens |
| sparse attention at decode (gather) | 97K decode 58.9 → 47.9 ms/token; slope 0.188 → 0.078 |
| draft head with its own IQ4_XS output projection | +5% English, +9% Chinese; acceptance unchanged |
| HIP graphs keyed on shape | +2%, and every draft graph replays instead of resetting its neighbour |
| `draft_max` 5 → 3 | 30.2 → 27.9 ms/token (see [decode-budget.md](decode-budget.md)) |

The two attention changes are the substantial ones, and they are the same shape of bug: work that
only prefill's layout supported, so a single decoded token fell back to reading the entire cache.

## Prefill

| | |
|---|---|
| IQ3_S reaching the matrix cores | 830.6 → **876.7 t/s** (+5.6%), quality unchanged |
| ~~`GGML_HIP_ENABLE_UNIFIED_MEMORY=0`~~ | retracted: nothing reads this variable (below) |
| overlapped PLE gather (depth 1 → 16) | gather 3.08 → 2.28 s, but end-to-end a wash: a detached prefetch thread was already hiding it |

**Retracted 2026-09-24.** The struck row credited +4.2% (867.1 → 903.6 t/s) and a six times
steadier prefill to `GGML_HIP_ENABLE_UNIFIED_MEMORY=0`, and the manager set it on every load until
0.1.14. No code in the runtime reads it: ggml reads only `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, and the
HIP runtime neither - checked in the source and in every DLL the release ships. Both arms of that
comparison ran the same program, so the gap was run-to-run variation; 0.1.15 stops setting it.

**Where prefill time goes (2026-09-23).** Per 8192-token ubatch the GPU time is ~7.6 s near the
start and ~9.2 s at 73K: sparse attention keeps depth cheap. Of the 7.6 s, the routed experts are
~2.1 s and already on the fused matrix-core path; the hyper-connections are ~2.1 s. The base has fused
kernels for the HC gate (GEMM + sigmoid + stream mix) and the PLE conv taps, but only for IQ4_NL HC
weights and an F16 conv weight; Unsloth's file has Q8_0 and F32, so `LLAMA_HC_GATEMIX` and
`LLAMA_PLE_CONV` were inert. (The sigmoid + mix itself was already one kernel; what the gate fusion
removes is the [10240, T] gate's round trip through memory.) `apply_hc_q8_fusions` gives both to these
types with the unfused path's numerics - the same greedy tokens and top-5 logprobs - and takes the
first ubatch 7592 → 7245 ms. GDN is ~12%. A 95.6K-token real-text prefill, production flags:
**846 t/s on 0.1.7, 888-892 with this round** (the fusions, then the next batch's PLE rows gathered
while the current one computes). The graph-timing instrumentation had made that gather look like idle
GPU time; it synchronises after every graph, and without it most host work overlaps the previous
batch. Two cheap A/Bs measured nothing worth their cost: ubatch 16384 +2.7%, `ROCBLAS_USE_HIPBLASLT=1`
+0.3%. Details: `docs/results/prefill-profile-20260923.json`, `docs/results/perf-round-20260923.json`.

**The expert gate/up, rebuilt for this GPU (0.1.9).** On gfx1151 the VALU and the WMMA unit never run at
the same time on a SIMD: every vector instruction adds about a cycle to a stream of 32-cycle WMMAs, so a
dequantizing GEMM is priced in vector instructions per matrix op. The routed IQ3_S gate/up + SwiGLU
spent ~9 of them per weight and waited on its weight loads at every step. `apply_moe_glu3` spreads
the dequantization over the whole block, puts the sign on an F16 copy of the grid so the scale
product is one `v_fma_mix` (exact, so the same BF16), keeps the loads whole and off the critical path,
and takes experts of 32 rows or more as big tiles. At 8192 tokens on the model's recorded routing a
layer went 35.9 → 19.9 ms; the 95.6K-token prefill 890 → 983 t/s; every output token and top-5
probability is unchanged. The IQ4_NL down projection is a different animal: without its WMMAs it still
takes 74% of its time - ten steps per tile, weights streamed in and 839 MB of F32 expert outputs a
layer written out for the weighted sum. Details: `docs/results/moe-glu3-20260923.json`.

**Several slots (2026-09-23).** With more than one slot the cache is one pool, and each step's
sparse-attention bookkeeping used to span all of it: one conversation beside 70K tokens of idle ones
decoded at 47.6 ms/token against 43-44 alone. The block list now covers only the conversations in the
batch: 44.8 ms/token, 25.5 tok/s with MTP (23.4 before; one slot 29), bitwise the same tokens, and an
18.9K prefill beside them 999 -> 1064 t/s. Saving a conversation that was decoded alongside others also stops
stalling the server: its state was read from the device one cell range at a time (6-10 s in a real
four-conversation session), now one run of ranges at a time. Details:
`docs/results/multi-slot-20260923.json`.

**One run of cells per conversation (2026-09-24).** The block list was only half of it: every graph still
viewed the pool from cell 0, and the MTP draft attends densely, so with MTP a conversation beside idle ones
drafted over all of them: 32.8 tok/s against 37.4 alone, and different drafts. Now each conversation keeps one
run of cells, packed in slot order with room after each, and a batch's graph views only its own run. When a
batch does not fit, the pool is laid out again first: the conversations keep their order and slide, each one
of the batch gets the same room after it and idle ones none (device copies on the graphs' own stream: three
idle conversations of 70K cells in 16 pieces, ~40 ms of host work). Beside 70K tokens of idle conversations
one conversation decodes at 44.1 ms/token (one slot 43.5) and 38.5 tok/s with MTP (one slot 37.8), and
computes bitwise what it computes alone - every token and top-3 probability, the draft's acceptance too -
also across forced re-layouts and a prompt extension that set one off. Four conversations generating
together: 47.4 tok/s against 40.7. The same round found the "bad allocation" seen once in ~16 multi-slot
runs: ggml-alloc reallocated the 3.4 GB compute buffer when a graph needed a few MB more, ROCm on Windows
keeps the freed buffer's commit, and the server ran into the machine's 127.6 GB commit limit. Compute
buffers now get a little headroom and are never reallocated: the four conversations peak at 83.7 GB with
6.9 GB of commit to spare, where 0.1.10 got to within 0.18 GB. Details:
`docs/results/kv-regions-20260923.json`.

**A KV pool larger than the context (2026-09-24).** Several slots share one pool of the context's size,
so two long conversations cannot both stay resident. `kv_pool` makes it larger (llama.cpp's own
`--kv-unified-per-slot` keeps each conversation at the context): at 393216 cells, conversations of 173K
and 155K tokens both stayed resident and each answered its next question in under a second, and four
concurrent conversations ran as fast as before (47.2 against 47.7 tok/s). Each cell costs ~39 KiB of GPU
memory and ~24 KiB of Windows commit, and commit is what runs out on this machine: 524288 cells fitted
the carve but failed with "bad allocation". The same test found the default pool re-processing a
conversation sent back to its emptied slot by id - 186 s where the disk tier had it all - fixed:
7 s now. Details: `docs/results/kv-pool-20260924.json`.

**Disk tier version 3 (2026-09-24).** Version 2 still gathered a conversation's whole state into one
buffer to save it and read it back whole - 5 GB at 173K tokens, which took the machine to its commit
limit when two long conversations swapped (GitHub issue #1) - and since appending to the KV cache moves
bytes in every layer's section of a serialised state, every save rewrote ~200 chunks. Version 3 never
serialises a state. A conversation's attention rows (29 KB a token, draft included) go to disk once
each, 4096 positions at a time as they are computed, named by their hash; its recurrent state, which
changes with every token, only when the conversation leaves memory - the state at its end and the last
prompt's two latest checkpoints, 0.33 GB - because another conversation needs its cells or the server
is stopped (the manager's unload asks it to write first). Conversations of 173K and 155K tokens swapped
through the default pool read back in 1.6 and 1.4 s (4.9 GB/s off the drive); 9.2 GB of rows were
written for 329K positions and 0.9 GB of state for three evictions, and commit never went below 9.6 GB
(version 2: 0.01). Two 33K chats taking ten turns with five evictions answered bitwise as they did kept
resident - after one fix: the graph that rebuilds the sparse-attention block keys, the first after any
restore or rewind, scored with the F32 keys it had just computed where every other graph scores with
the F16 keys the cache holds; it now reads them back too (perplexity unchanged to four places). The
store keeps only what the build writes: entries of an older format are deleted at startup. The disk
tier is off by default since 0.1.13. Details: `docs/results/disk-tier-v3-20260924.json`.

**A slot that fails part way loses its conversation (2026-09-24, GitHub issue #1).** Long agent sessions
near the commit limit logged "non-consecutive token position" - batches that fed the recurrent state
positions it had already seen, or skipped some - and ended in a GPU fault. Every place the server
caught an exception mid-turn released the slot and kept its conversation, although part of a batch had
been recorded in its tokens and not run; under commit exhaustion `std::bad_alloc` comes from ordinary
work such as a 110 MB checkpoint, and with one slot the leftover tokens aborted the server on an
assert. A failing slot now leaves the batch and loses its conversation (processed again, or read back
from the disk tier), and a prompt batch starts only on a cache whose positions match its tokens.
Injected failures at three sites recover to the tokens of a server that never failed; an agent
workload (regenerate, aborted streams, trimmed tool output, two conversations on one slot) ran to 132K
tokens with neither warning. The same round cut what a resident conversation keeps: its last prompt's
checkpoints and one per 32K tokens, at most 0.9 GB of system RAM a slot instead of 3.5, and the disk
tier stores that spacing too, so a conversation read back from disk can still be rewound cheaply.
Details: `docs/results/slot-state-guard-20260924.json`.

**The build layout ran on the display driver's HIP runtime (found 2026-09-24).** A build run from
`bin/hip-rocm101` or the build directory had no ROCm DLLs beside it, and Windows searches System32 before
PATH, so it loaded the driver's `amdhip64_7.dll` instead of the SDK's that the bundle ships. Same binaries,
same prompt: 29.5 tok/s with MTP there against 37.8 in the bundle (acceptance 0.48 against 0.71), and
different text from the ninth token. Measurements made through the manager before this date ran on the
driver's runtime; comparisons inside a round hold, absolute MTP figures are low. `bootstrap.py --build`
now puts the SDK's copies beside the build, and the two layouts agree bitwise. See `docs/measuring.md`.

**The rest of a prefill batch (0.2.0, 2026-09-25).** With the expert kernels at their limit since 0.1.9, a
2K-token batch still spent a quarter of its GPU time on work that should have been cheap. The
hyper-connection inject (10240 inputs, 4 outputs, F32) ran through a GEMM tile 64 rows wide. The HC down
projection read BF16 activation rows whose 4 KB stride put all of them on the same memory channels. The
lightning indexer scored a query strip in a matrix product, a ReLU, a sum and more small passes. Each
layer's sparse-attention key and value packs were full copies of the cache, made twice. 0.2.0 computes the inject inside the combine-norm kernel while its
input is in registers, pads those rows by 64 elements, scores a strip in one kernel, selects TOP_K by a
radix select over a row held in registers (the same selected set), writes the packs from the cache in one
pass, and runs the gated delta net's prefill with 16 state rows and four columns a lane group and the state
scaled by its decay, so a token's two reductions run together. Narrow F32 weights get their own kernels,
and F32 GEMMs from 9 columns go to the WMMA kernel instead of hipBLAS, which loads each of its kernels
from disk on first use: stalls of 20-400 ms, again whenever a context length reaches a new one.

GPU time summed over a pp2048 batch's dispatches (`STRIX_NODE_TIMING=1`, graphs off), against the same
build with every new switch off: 1860 -> 1573 ms at depth 0, 2276 -> 1788 ms at depth 64K. The largest
moves at 64K, in ms over the batch: inject 177 -> 5, HC down 118 -> 57, indexer scores 153 -> 29, TOP_K
51 -> 21, GDN 118 -> 93, and the packs' two copies (45) become one pass. The attention gate's copy+sigmoid
fusion is worth ~2 ms: it displaced 0.1.17's sigmoid+multiply fusion. End to end, the two releases
alternating on one machine, each run on a fresh server:

| | 0.2.0 | 0.1.17 |
|---|---|---|
| prefill, 95.6K tokens of real text | 1187.4 / 1183.7 / 1200.6 t/s | 999.1 / 965.5 / 992.4 |
| prefill, 86.1K tokens of real text | 1160.6 / 1119.1 / 1143.8 | 978.8 / 963.5 / 959.5 |
| pp2048 at depth 0 / 64K, MTP off | 1070 / 906 | 919 / 739 |
| decode after the 86.1K tokens, 400 tokens | 29.10 / 29.67 / 29.48 ms/token, acceptance 62% | 29.85 / 30.60 / 31.47, 63% |
| four conversations at ~20K tokens, q8_0 | 49.7 / 49.8 / 50.8 tok/s summed | 47.1 (its release check) |

Decode does not move: a step is weight bandwidth, which none of this touches (at 2.3K tokens the median
speculative pass is 66.9 against 68.0 ms). The four-conversation gain is the GDN reading its states where
they are instead of gathering them (3 MB a sequence a layer) and its output projection as one product.

0.2.0 is not bitwise 0.1.17: the inject's partial sums, the GDN's scaled state and the small F32 GEMMs add
in another order, and this model amplifies a last-bit difference. The 18.6K-token equivalence probe gives
the same 48 tokens, but its first token at probability 0.925 against 0.953. Perplexity at 8K context over
8 chunks: 2.6811 against 0.1.17's 2.6841 (f16), 2.6927 against 2.6951 (q8_0); on the 09-17 gate
(`corpus/ppl-en-code.txt`, ctx 4096, 6 chunks, ubatch 512) 2.4730 against 2.4741.
`STRIX_HC_INJECT_FUSE=0 STRIX_GDN_R16=0 STRIX_SKINNY_F32=0 STRIX_MMB_F32_MIN_T=512` gives 0.1.17's output
bit for bit. The release checks found one real bug on the way: the two copy fusions read their source
while they write, and with a q8_0 cache the source was the dequantized f16 copy in the compute buffer,
whose memory the allocator had already given to the fused output (perplexity 9.75). `graph_optimize` now
keeps the source allocated until the output is computed, and the dispatch refuses overlapping memory.
Details: `docs/results/prefill-kernels-20260925.json`.

**Several conversations at once (0.2.1, 2026-09-25).** After 0.2.0 three conversations decoding together
got 34 tok/s summed with MTP on, less than with it off (46). Per step, with speculation off and ~4K tokens
each (`LLAMA_GRAPH_TIMING=1`), the GPU spends 33.8 ms plus ~8.8 ms for every further conversation - ~4.5
of it the weights of the ten experts a new token routes to, read at the memory's bandwidth, the rest
the per-sequence GDN, sparse-attention and dispatch work - and the host 7.6-10.6 ms. With MTP the
drafts of the conversations came out of different lengths, and the hybrid memory runs a batch as
ubatches with the same tokens for every sequence: 19% of the steps became two or three passes of the
whole model, each a graph shape the HIP graph cache had not seen (a rebuild of ~28 ms and 8500 separate
launches), and took 35% of the time. `apply_spec_even_drafts` gives the drafts of one step one length:
three conversations at 4K tokens 34.3 -> 45.7 tok/s, at 20K 39.6 -> 45.7; two 39-42, where they had
swung between 26 and 41; four, which do not draft, 49.4 / 49.9 against 48.6 / 50.3; one unchanged.
The same round found a 272 ms stall in the first step three conversations decoded together: the
compact scorer's block-membership product (K = the number of sequences) went to hipBLAS, which loads a
kernel from disk on first use of a shape; `apply_small_k_membership` gives it a kernel of its own,
0.02 ms and exact. gufo-org/gufo's multi-user table sends every user the same prompt, so all route to
the same experts and read them once; it adds 4.3-5.3 ms a user where we add ~8.8 with distinct prompts.
Details: `docs/results/multi-stream-20260925.json`.

**First-use stalls in a chat (0.2.2, 2026-09-25).** The new part of a chat turn is usually 17-511 tokens,
and in a batch that size the sparse-attention indexer's BF16 projections went to hipBLAS, which loads
each GEMM kernel from disk the first time it meets a shape: a second turn's 38-token batch took 572 ms
instead of 200. The MTP draft head's Q6_K projection did the same above 256 columns (208 ms in the
draft's first prompt batch). `apply_bf16_mid_batches` and `apply_q6k_mmq_rdna35` keep both on this
fork's kernels. A five-turn chat on a fresh server, two runs of each build: the first message's time to
first token 4.34 / 4.95 s -> 3.52 / 3.98 s, the second turn's 1.50 / 1.59 s -> 1.10 / 1.19 s; later
turns differ with the answers. An 18.6K prompt in one batch is bitwise 0.2.1; perplexity at 8K context
is 2.6814 against 2.6811, from the perplexity tool's 8192-row output projection now on MMQ (2.6811 with
`STRIX_MMQ_Q6K_ANY=0`; the server's are 1-16 rows). Details:
`docs/results/chat-ttft-20260925.json`.

**Several conversations at once, the verify step (0.2.3, 2026-09-26).** A verify step of several
conversations carries 5-16 tokens, and two of its products fell off the vector kernel there onto tiles
they barely filled. The MoE router, an F32 [2560 × 512], took MMB's 128-row tiles from 9 columns: four
workgroups for the whole GPU, ~250 µs a product. The routed experts past the vector kernel's limit (4
tokens for IQ3_S, 6 for IQ4_NL) took MMQ, which stages a tile per expert for the one or two tokens an
expert gets: a gate/up product at nine tokens took 397 µs where four take 90, the down 499.
`apply_small_batch_decode` runs both on the vector kernel over chunks of the batch (the router up to 32
columns, the experts in chunks of 4 up to 16 tokens): gate/up 397 → 289 µs, down 499 → 387, and a
nine-token verify step 120-123 → 103-104 ms. With that, drafting pays at four conversations (62.6 tok/s
summed with drafts of 2 against 55.7 without; at ~20K tokens each 56.1 against 49.8-51.4) and still
loses from six (60.7 against 64.4; eight 64.0 against 70.7), so the manager's cap is now 3,2,2,2,0. The
ROCm pin moves to TheRock 10.2.0a20260925:
perplexity 2.6814 on both, the 18.6K equivalence probe bitwise, prefill and one conversation the same,
three and four conversations +3-4%. 0.2.2 against 0.2.3, alternating, three passes of two rounds with
the server's default sampling, ~4K tokens each, 512 tokens a stream: one conversation unchanged at ~38
tok/s, three 47.7 → 55.7 summed (1.26× → 1.47× one), four 55.4 → 60.4 (1.46× → 1.59×). 0.2.3's
acceptance at three and four moves between ~65% and ~72% from round to round (0.2.2's at three stays at
66-67%), so its single runs spread more: 52.9-57.4 and 58.4-63.2. At three the GPU is busy ~89% of the
time, and a step still adds ~8 ms for every further token it verifies, ~5 of it expert weights read
near the memory's bandwidth; `--backend-sampling` measured nothing. Details:
`docs/results/multi-stream-20260926.json`.

**Several conversations at once, three more small products (0.2.4, 2026-09-26).** After 0.2.3 a
nine-token verify step took ~105 ms against a floor of ~51 ms for the weights it reads, and three of its
products still ran a few workgroups for the whole GPU. The GDN conv input concatenates each sequence's
conv state with its transposed tokens; below 32 tokens that took `concat_non_cont`, one 256-thread block
per channel and sequence for 3 + tokens values, ~30K blocks at three conversations. The indexer's BF16 k
projection [2560 × 128] took MMB's one 128-row tile at 9-32 columns. Quantized weights of at most 1024
output rows - the hyper-connection down projection [10240 → 320] 96 times a step, the attention k and v,
the shared expert's gate and up - took 3-5 MMQ tiles, and MMQ has no stream-k on RDNA.
`apply_verify_small_products` sends the concat to the tiled transpose from 2 tokens and runs both kinds
of weight on the vector kernel over chunks of 8 columns; the [6144 → 2560] projections stay on MMQ, where
reading the weight a second time cost more than the idle tiles. Development A/B, greedy, the verify
step's median per run: nine tokens 103.7-106.7 → 99.0-101.5 ms, twelve 122.5-124.9 → 118.9-121.5, one
conversation 54.8 / 55.0 → 54.2 / 54.7.

Release A/B, 0.2.3 against 0.2.4, alternating, three passes of three rounds, ~4K tokens each, 512
tokens a stream, the server's default sampling with a new seed every round (the same seeds for both
builds). The server's own timing over ~1,900 nine-token and ~1,800 twelve-token steps a build: verify
106.4 → 103.0 ms and 129.4 → 123.5, the whole step 124.1 → 120.7 and 151.5 → 144.5. A round's tok/s
follows the acceptance of the text it samples (54-86% over these rounds, ~0.4-0.5 tok/s a point), and the
builds drew different acceptance by chance, in both directions (three conversations 71.3% against 64.7%,
four 65.4% against 68.6%), so the tok/s are read at the same acceptance, from a least-squares line per
build with a pooled slope: three conversations 53.8 → 55.0 tok/s summed (+2.2%), four 58.1 → 60.7
(+4.5%); the plain means were 55.2 → 53.6 and 57.3 → 61.4. One conversation's steps carry at most four
tokens and are unchanged: 56.3 ms a verify step on both, 37.5 / 37.6 tok/s in the first A/B (below).

The first release A/B fixed each conversation's seed, as 0.2.3's had, and read 0.2.4 as no faster (three
conversations 55.9 → 54.5 tok/s) at a lower acceptance (71.5% → 66.7%). It was measuring the sampled
text. A server that computes a step the same way every time samples the same text in every round with a
fixed seed, and the acceptance is that text's: 0.2.4 did at three conversations (the same three texts in
four rounds out of four, 67% in nine rounds out of ten), 0.2.3 did not (three different outcomes in four
rounds, 65-79% over ten). The 0.2.4 build with 0.2.3's routing through the switches samples 0.2.3's
texts at 0.2.3's acceptance, so the text, not the build, set it. The rounds differ where the
conversations do not join the same steps: up to 0.2.3 these products changed kernel at 9 tokens, so a
token's result could depend on the size of the step it landed in; four conversations still vary. The
same fixed seed flattered 0.2.3 against 0.2.2 (above): 0.2.2 sampled nearly the same text every round at
three conversations (66-67%), 0.2.3 several (65-72%), and at the same acceptance its gain was +12.8%, not
+17% (four conversations, which 0.2.2 did not draft for: +8%, not +9%). Perplexity
with 16-token batches, where every batch takes the new paths, is 4.0732 against 4.0870 with 0.2.3's
routing (ctx 4096, 2 chunks); at 8K with 8192-token batches 2.6814 on both.

What did not pay, parked behind switches in the development tree: a MoE kernel that reads each distinct
expert once for all its tokens (nine tokens of three conversations route 90 expert-token pairs to 54
distinct experts, but the 32 MB MALL already serves the repeats: neutral once the grouping was one extra
kernel, -30% before); a one-pass Q8_0 kernel for 9-16 columns (1.5-3× MMQ in the op bench, where one
weight stays in the MALL; equal to the chunks in the model); strided copies with rows under 1 KB on the
copy kernel instead of `hipMemcpy2DAsync` (neutral). Where the rest of a nine-token step goes: the
routed experts run at ~85% of the DRAM peak for the distinct experts read; the GDN kernel writes one full
recurrent state per verified token for rollback, 3 MB a sequence a layer, ~1 GB a step at three
conversations; the host leaves the GPU ~3 ms before each verify step. Three measuring traps of this
round - per-dispatch times from graph event nodes, op benches whose weight stays in the MALL, and leaving
ops out, which changed the MoE routing - are in `docs/results/verify-small-products-20260926.json`.

**Agents and several conversations: a prefix cache across slots, rollback without snapshots (0.2.6,
2026-09-27).** Agent harnesses run many sessions and sub-agents on one long system prompt, with tool
calls in between. `tmp/pc/cache_workload.py` replays three kinds of traffic through the chat API
and counts, per request, the prompt tokens the server processed against the ideal: the prompt less its
longest common prefix with any earlier prompt of the run. The recurrent state can be resumed only where
a checkpoint of it was kept, and upstream's retention (eight checkpoints; when full, every other task's
checkpoint within `--checkpoint-min-step`, 32768, of the previous one erased) left the first prompt's
T-8196 checkpoint and the current task's after a few turns. So a second agent session on the same 10K
system prompt went back to 2.4K and processed 8.1K of it again, and editing the second message went back
to 0. Three sessions started together each computed the same 10.6K prefix, 34 s for each first answer.
And a new session took over the slot that shared the most with it, cutting that conversation, while a
slot was free. `apply_prefix_cache_026` keeps checkpoints where later prompts branch (the first user
message, kept for good; the last one; the end of each prompt; turn starts spaced by
`--checkpoint-min-step`, now 4096, and a quarter of the distance to the end). A full list drops the
checkpoint whose loss costs least: gap before × gap after / distance from the end. A new session copies
a shared prefix from another slot instead of computing it: attention rows device to device, 44 ms for
9.7K positions (100 ms through the host, same answer and top-3 logprobs over 64 tokens), and the
recurrent state from the source's checkpoint. A session whose prefix another slot is computing waits for
that slot's anchor, then copies it.

| workload (`cache_workload.py`) | 0.2.5 | 0.2.6 |
| --- | --- | --- |
| agent: 10K system prompt, three sessions of five turns started together, three sub-agents, 4 slots | 73.3K processed (2.47× ideal), prompt time 172.5 s | 29.3K (0.98×), 52.2-60.0 s |
| edit: six turns, then regenerate, edit the last message, edit the second, 1 slot | 22.9K (1.16×), 24.3 s; the second-message edit 4266 tokens | 19.7K (1.00×), 21.2-22.6 s; 1096 tokens |
| chat: six conversations round-robin, 4 slots, disk tier off | 208.7K (2.92×) | unchanged: more conversations than slots |

In the agent workload the three first answers came after 15-17 s instead of 36-41 s. What a generating
stream still feels (`tmp/pc/stall_probe.py`, a stream decoding at ~63 ms a step): its longest step was
299-335 ms while another request copied a 9.7K prefix, and 1420 ms while a 47K conversation came back
from the disk tier (1.42 GiB read in 521 ms, then the rows set). A checkpoint costs 10-30 ms (restore
10 ms, create 19-31 ms for 112.6 MiB). The rest of such a step is the new prompt's tail run as extra
passes of the whole model, because the hybrid memory needs every sequence of a ubatch to have the same
token count: {A:4, B:14} runs as {A:4, B:4} + {B:10}.

The speculative verify wrote the whole recurrent state after every token of every sequence, 3 MB a
layer, so that a rejected draft could be rolled back. Writing every snapshot twice
(`STRIX_GDN_SNAP_DUP`, output unchanged) cost +0.65-0.75 ms per 108 MB, +6.8 / +8.9 ms on a 9 / 12-token
step. `apply_gdn_deferred_rollback` leaves the state where the batch found it and records per token what
a replay needs: delta [128 × 48], key [128 × 16] and gate [48], 33 KB. The next batch replays in
registers the records its rollback kept, then writes the state back. A replay op brings states up to
date before a longer batch, and a saved state is replayed on the host. The plain kernel's `g*s + k*d`
compiles to `fmaf(g, s, k*d)`; written as `fmaf(k, d, g*s)` the outputs changed, and with the same fmaf
everywhere they are 0.2.5's bit for bit. Verify step, same binary, greedy, `STRIX_GDN_LAZY=0` against 1,
two runs each: 9 tokens 98.85 → 96.05 ms (-2.8%), 12 tokens 119.4 → 114.6 (-4.0%), 4 tokens 54.2 →
54.1. Keeping one state row per cell instead of two (324 MiB a slot less) failed: a rollback right after
a batch longer than a verify had no state to go back to.

Release A/B, 0.2.5 against 0.2.6, one pass each (0.2.6 first), ~4K tokens a conversation, 256 tokens a
stream, six rounds, the server's default sampling with a new seed every round (the same seeds for both
builds). Read at the same acceptance (pooled slope 0.4-0.5 tok/s a point): three conversations 53.7 →
56.4 tok/s summed, four 60.0 → 61.9. One conversation sampled the same six texts on both builds at 37.9
→ 39.3 tok/s. The server's verify step: 9 tokens 102.1 → 97.7 ms, 12 tokens 122.4 → 116.7, 4 tokens
56.4 → 54.4. That is more than the same-binary A/B above, and the builds ran one after the other, so
part of it may be order. Four slots are the default now: 75.77 / 76.21 / 77.09 / 78.73 GB of dedicated
GPU memory at 1 / 2 / 4 / 8 slots, idle, pool = context, ~0.43 GB a slot. Details:
`docs/results/prefix-cache-gdn-20260927.json`.

**Several agents at once: slots, ragged batches, a prefill budget (0.2.7, 2026-09-27).**
`tmp/ragged/agent_sim.py` runs what an agent harness does: an orchestrator and five sub-agents spawned
together, on one 12.5K-token system prompt, each step a streamed answer, then a tool that runs 0.5-3 s,
then its 2-8K-character result appended. It records the time to each step's first streamed chunk, the
gaps between chunks, and the tokens processed against the ideal. On 0.2.6's four slots nearly every step
went back to the system prompt: 75K tokens processed where ~46K were new, a median of 10.1 s to the
first chunk, gaps up to 6.2 s. Two causes. Six agents on four slots push each other out whatever the
choice of slot (a cyclic working set larger than the cache: the same policy at four slots processed
1.8× the ideal). And upstream's choice by similarity let a new session take another's slot and cut its
history, because the shared system prompt made every slot look 97% similar (`f_keep` 0.97, above the
0.75 that sends a prompt to a free slot). `apply_agent_slots_027` takes a slot for what it holds only
when the prompt continues it: it keeps all that was last asked there, less the 16 tokens a template
may render differently once the answer is in the history. Anything else goes to a free slot, else the
least recently used one, and `copy_prefix` brings the shared part. The manager's default is now eight
slots. At eight, with upstream's choice, the one steal left (the orchestrator's slot, taken by the
first sub-agent) was an 11.3 s gap.

The rest of the stalls were the passes. The hybrid memory needs every sequence of a ubatch to have the
same token count, so five sub-agent prompt tails of 292-556 tokens ran as six passes over all the
weights (~590 t/s), and conversations decoding next to a prompt paid a pass of their own.
`apply_ragged_ubatch_027` makes one ubatch of sequences of any lengths (`split_ragged`); qwen4exp runs
its per-sequence parts (the gated delta net's conv and net, the per-layer-embedding conv) once per
group of one count and the rest once. Then, while conversations generate, a step takes at most 2048
prompt tokens (`STRIX_PREFILL_BUDGET`). A 19K-token prompt used to hold a streaming conversation for
6.9-7.3 s at a time; with the budget, it holds it for 2.2 s at a time, and the prompt takes 21 s
instead of 17. A budget of 1024 kept every pause under 1.3 s but cost the agents 20-30% of their time
to the first chunk.

| six agents, disk tier off | processed (ideal) | wall | first chunk p50 / p90 | gap p99 / max | gaps > 1 s |
| --- | --- | --- | --- | --- | --- |
| 0.2.6, 4 slots (its default) | 75.0K (~46K) | 117 s | 10.13 / 13.81 s | 2961 / 6200 ms | 80 |
| 0.2.6, 8 slots | 45.2K (46.0K) | 90 s | 3.80 / 8.46 s | 2538 / 5251 ms | 59 |
| 0.2.7, 8 slots (its default) | 44.8K (46.4K) | 96 s | 3.37 / 7.66 s | 2427 / 2746 ms | 50 |

The 0.2.7 run streamed 20.5 chunks a second against 17.2. Earlier runs of the same two configurations
spread by ±0.5 s on the p50 (tmp/ragged/results.md). Outputs of prompts batched together stay inside
the model's sensitivity to batch shape: the same prompt alone at ubatch 512 against 8192 moves its
first token's logprob by 0.15 on average (0.32 at most, 5 of 8 first tokens the same); prompts
processed together against alone, 0.16 / 0.36 with the equal split and 0.07 / 0.20 ragged. Several
conversations decoding without prompts are unchanged (at the same acceptance three 55.9 against 55.8
tok/s, four 62.8 against 61.6). Keeping more conversations costs RAM as well as GPU memory: each one
in use holds up to eight snapshots of its recurrent state in host memory, 112.6 MiB each (six agents:
the server's working set 6.3 GB at four slots, 9.8 GB at eight). Details:
`docs/results/agent-sessions-20260927.json`.

**A window the sparse kernel refused, and prompts that share a pass only when it pays (0.2.8,
2026-09-27).** A user's log on 0.2.7 (four slots, a 512K-cell pool, MTP off, conversations of 45-115K
tokens) showed prompts of 44-56K tokens arriving while another conversation answered at 415-750 t/s
against ~1250 alone, and prompts of 890-3348 tokens - a new chat, a conversation read back from the
disk tier with a short tail - at 42-46 t/s, ~22 ms a token, the other conversations standing still for
19-79 s. `tmp/ragged/multi_deep_probe.py --evict --restore` reproduces both at full size: a ~60K-token
conversation E answered and left idle; three conversations of 88K, 49K and 74K tokens (A, B, C), each
starting once the previous one streams; a new 2.4K-token chat D that takes E's slot; E again with a
3.9K-token tail, read back from the disk tier into the slot D leaves; a new 837-token chat F.

Both were 0.2.7's ragged ubatches: a 2048-token prompt chunk and the answering conversations' tokens in
one ubatch. The sparse attention scores its index over every block of the ubatch's sequences, not only
the query's own (~3 ns a query and cell of the others' contexts: `MUL_MAT [68480, 2048]` and the
visibility ops cost ~1.2 s of a 2051-token ubatch next to three conversations), and a ubatch of several
sequences loses the causal score bounds of a single one. The stalls were worse than slow. The window of
a ubatch of several sequences spans all of their cells, and the qsa3 kernel numbered blocks of 4 cells
in 16 bits: past 262140 cells it declined. The generic flash-attention kernels that ran instead ignore
the selected indices, and the graph had left the mask out (every prompt-sized ubatch is maskless), so
each query attended densely to every cell of the window, the other conversations' and later positions'
included. `STRIX_NODE_TIMING` on the tail of E: 36.6 of its ubatch's 40.0 s in the 12 `FLASH_ATTN_EXT`,
2051 queries over 298752 cells. Half the sizes (136K cells in all) stayed under the limit and never
showed it. The result was wrong, and the disk tier stored it: asked about the tail of E as a chat, 0.2.8
alone and 0.2.8 next to the three answers gave the same sentence (first-token logprobs -0.0069 / -5.21 /
-8.29 against -0.0064 / -5.31 / -8.54), 0.2.7 next to them a different one (-0.0093 / -5.52 / -6.30,
two nats off where batch shape moves them by 0.15-0.3), after 233 s instead of 13.

`apply_qsa3_wide_window_028` gives the kernel's union list 32-bit block numbers, so it takes a window of
any size; below 262141 cells it computes what it did (an 18.6K-token prompt: 48 tokens and their top-5
probabilities identical to 0.2.7; PPL 2.6814; the single-stream texts of six sampled rounds identical).
`apply_prompt_alone_028` puts the short sequences of a batch together (decode tokens, MTP verify drafts,
a prompt of a few tokens; at most 32 tokens in all, what the sparse attention's decode gather takes), and
a prompt chunk joins them only while the query-cells it adds - its queries over the others' contexts,
theirs over its own, the lost score bounds - stay under 6e7 (`STRIX_MIX_CELLS`, ~0.2 s, about a pass);
otherwise it runs alone next. Next to long conversations a chunk always runs alone; short tails next to
small conversations still share. `apply_disk_store_v4_028` renumbers the disk tier's layout, so a store
written by 0.2.7 or before is emptied once at startup. GPU memory is the same as 0.2.7's (71.78 GB
loaded at the default 4 slots, 131072 cells, MTP and images on; 78.89 GB at a 512K pool without MTP).

| four slots, 512K pool, MTP off | B 49K prompt | C 74K prompt | D 2.4K chat | E + 3.9K tail (from disk) | F 837-token chat |
| --- | --- | --- | --- | --- | --- |
| 0.2.7 | 793 t/s, first token 61.9 s | 695 t/s, 106.5 s | 4.8 s | 76.6 s, others paused 41.3 s | 2.0 s |
| 0.2.7 with the equal-length split (`STRIX_RAGGED=0`) | 1012 t/s, 48.6 s | 988 t/s, 75.1 s | 3.4 s | 4.4 s, 3.0 s | 1.6 s |
| 32-bit blocks, every chunk shares (`STRIX_MIX_MAX=0`) | 805 t/s, 61.0 s | 685 t/s, 108.2 s | 4.8 s | 6.8 s, 4.4 s | 2.0 s |
| 0.2.8 | 1035 t/s, 47.5 s | 1006 t/s, 73.7 s | 3.3 s | 4.3 s, 3.0 s | 1.5 s |

0.2.6, which had no prompt budget, took B at 1223 t/s (first token 40.3 s) and C in 61.7 s, but held the
answering conversations for each 8192-token step. Letting no chunk share at all cost the six agents of
`agent_sim.py` a fifth more prompt time (125 s against 106), the time to the first chunk 4.21 / 11.16 s
at p50 / p90 against 3.37 / 7.66; with the cost rule two runs gave 111 and 114 s, 3.60 / 8.63 and
4.57 / 7.78 s, and the longest pause 2.7 and 2.6 s against 2.7. Details:
`docs/results/prompt-alone-20260927.json`.

**Sparse attention between the two kernels, and two prefill kernels without scratch (0.2.9,
2026-09-27).** A ubatch of 33-127 queries a sequence sat between the two kernels that read the sparse
selection: the decode gather takes at most 32 (`LLAMA_QSA_DECODE_GATHER_MAX_T`) and the qsa3 prefill
kernel at least 128. Its block selection went to the generic flash attention with the plain causal mask,
which ignores the selection, so the queries attended densely to every visible cell - tool results and
short follow-up messages at depth, what agents send. `apply_qsa_between_029` gives such ubatches the
plain top-k and the top-k mask (the selection's -1 sentinels are for the kernels; the mask's `set_rows`
cannot take them), and `fattn.cu` now aborts when a maskless op with selected indices reaches a generic
kernel, so 0.2.7's wide-window bug and this one would have stopped the server instead of attending
densely. The GDN's r16 kernel staged each tile through per-thread register arrays that the compiler kept
in scratch (10 `scratch_store_b128` and 10 `scratch_load_b128` a tile, a private segment of 112-272
bytes a lane; `tmp/qsa/scratch_scan.py` lists every kernel's); loaded straight into LDS it needs none,
and a 64K-token prefill's GDN time goes 3006 -> 2582 ms. The per-head q/k norms (rows of 128 floats)
ran the generic kernel, a block a row; the gated norm's kernel, a wave a row, takes a variant without
weights: 449 -> 352 ms. Both are bitwise: an 18.6K-token prompt and 48 tokens with every top-5
probability identical to 0.2.8, PPL 2.6814, and the user's-case probe's first-token logprobs the same.
A 95.6K-token real-text prefill, alternating on fresh servers: 0.2.8 1226 / 1231 t/s, 0.2.9 1242 /
1245. Details: `docs/results/qsa-between-20260927.json`.

**The stable release: large KV pools with MTP, one pass over the cells' membership, and a soak (0.3.0,
2026-09-28).** With MTP on, a KV pool above ~512K cells failed to load: the draft context reserves a
dense mask over the whole pool for its ubatch, ~4.8 KiB a cell at the 2048 the manager gives it - 3.9 GB
at 768K, which a user's log showed failing after the target and its K/V had loaded. The manager now
shrinks the draft's ubatch past 512K cells in proportion (`draft_ubatch`: 1536 at 640K, 1280 at 768K,
1024 from 896K, never under 512), so the reserve stays where a 512K pool put it; only the draft's share
of a prefill, its one layer, runs in more passes. Loads at four slots with MTP, dedicated / shared GPU
memory after an 8K prompt: q8_0 512K 83.5 / 1.5 GB, 640K 87.5 / 1.4, 768K 91.5 / 1.4 (it did not load),
896K 92.0 / 4.9, 1M 92.1 / 8.8; f16 512K 87.7 / 1.5, 640K 92.3 / 1.9, 768K 92.1 / 7.3. The carve's
dedicated part tops out near 92 GB, and past it the display driver places the rest in shared memory,
slower but loaded, as the manager's defaults already assume. The user's-case probe (three long answers, a
new chat, a conversation back from disk with a tail) at a 768K q8_0 pool with MTP runs as at 512K
without it.

`apply_kq_mask_seq_bits_030`: each sequence's first KQ-mask row tested every cell's membership bitset
(256 bits, 32 bytes a cell), so a step of several conversations read the window's bitsets once per
conversation. The bits of the step's sequences are now gathered in one pass, two bytes a cell, and the
rows test those; the test and the cells are the same. `STRIX_KQ_MASK_CHECK=1` fills every mask the
upstream way as well and compares the two byte for byte: over several conversations of ~28K tokens,
images in several conversations and eight agents, 900+ masks, none different. Four conversations over
~112K cells: a step's inputs 2.33 -> 1.75 ms (median of 74-76 steps).

Checked for the release: the full gate set of 0.2.9 (an 18.6K-token prompt's text and top-5
probabilities identical to 0.2.9's, PPL 2.6814, 40 chunks at ctx 8192 identical chunk by chunk, 2.8134);
a prompt past the context is refused with HTTP 400 and `exceed_context_size_error` and the server stays
healthy; an answer that runs into the end of a 16K context with MTP drafting stops with `stop_type`
`limit` after 1112 tokens and the server answers the next request; a 254,723-token prompt prefills at 1127 t/s from an empty cache. And a soak (`tmp/qsa/soak.py`, 150 minutes, the app's default profile with images and the disk tier on):
rounds of six agents, the prefix cache's agent scenario, images in several conversations, four
conversations decoding together and a 34-50K-token prompt - 29 rounds, 145 steps, none failed, the server
up throughout. Its idle memory after each round: 85.50 GB dedicated GPU memory in every round, 5.83 GB
shared from the eighth on, 675 handles; private bytes rose from 95.2 to ~98 GB over the first nine rounds
as the slots filled with conversations and their checkpoints, then averaged 97.72 GB over rounds 10-19 and
97.82 over 20-29, within 96.7-98.4 throughout.
Details: `docs/results/stable-030-20260928.json`.

## Correctness

Two bugs that produced wrong output rather than slow output, both found late because the standard
gate could not see them:

- **Speculative verification ran dense attention with no causal mask.** Only the sparse kernel
  honours "indices, no mask", and it declines batches under 128 queries; on this backend every
  smaller batch fell through to the generic kernels. Single-token decode had nothing to leak, but a
  2-8 token verification batch let the target see the very tokens it was checking, so acceptance
  stopped being a check: long answers drifted, looped, and stopped early. Perplexity at ctx 2048
  read 2.3807 in every variant, because the bug only bites once sparse attention is active above
  ~2K. At ctx 4096 it is unmissable: 1.0841 at ubatch 512 and 1.8397 at ubatch 4, against 2.4646
  fixed. After the fix, 1500 speculative tokens are byte-identical to greedy decoding.
- **Image input aborted the server three separate ways**, all in the QSA block machinery, all in
  `llama-memory-hybrid-idx.cpp`. M-RoPE repeats one position across an image and advances positions
  past it by the image's extent, so cells and positions decouple in both directions, and the
  block-key cache's staleness tracking is keyed on 1-D positions. The third abort was in the *MTP
  draft* context, which never stores image tokens and so develops a hole its positions run past.

  None of the three reproduced on a single-turn test at short context. Two needed the third or
  fourth turn of a growing image conversation and one needed a divergent branch, so
  `tools/vision_stress.py` walks a conversation to ~17K tokens and then branches twice. Run it
  against any change near sparse attention, M-RoPE or the KV cache; it generates its own image, so
  it needs nothing but a server with a projector loaded.

  A fourth (0.2.6): the sparse attention ranks an image's cells, which share one position, and it
  ranked them only while the cache held one sequence, so an image in a second loaded conversation
  asserted in `set_input_qsa`. The app required one slot for image input, which hid it. Each sequence
  of a ubatch is now ranked among its own cells (`apply_image_rank_per_seq`), and
  `tmp/pc/vision_multi_conv.py` sends images to several loaded conversations, one at a time and at
  the same moment.

- **A prompt next to other conversations attended to all of them (0.2.7).** The sparse prefill kernel
  numbered blocks in 16 bits and declined a window past 262140 cells; the generic kernels that ran
  instead ignore the selection, and the prompt-sized graph had no mask, so every query attended to
  every cell of the window. 0.2.7's ragged ubatches, a prompt chunk next to answering conversations,
  span all of their cells, so four long conversations crossed it: ~20 ms a token and a different
  answer. No gate saw it: they load one or two conversations, and at half the sizes the window stays
  under the limit. `tmp/ragged/multi_deep_probe.py --evict --restore` fills a 512K pool (0.2.8 above).
- **Pieces of 33-127 tokens attended densely (0.2.9).** Between the decode gather (at most 32 queries)
  and the qsa3 kernel (at least 128), the block selection went to a generic kernel that ignores it, with
  the plain causal mask: dense attention over the whole visible context instead of the model's sparse
  attention, for exactly the tool results and short messages agents send at depth. Such ubatches take
  the top-k mask now, and a maskless op with selected indices that reaches a generic kernel stops the
  server. Perplexity could not tell the two apart (8 chunks move 1-2% between benign variants of this
  model); the kernel map could.

## What is still open

- **A generating conversation still pauses while another one's prompt runs.** Since 0.2.7 a step
  takes at most 2048 prompt tokens while conversations generate, and since 0.2.8 a chunk next to long
  conversations runs in a pass of its own, so a pause is ~2.2 s for a new prompt next to one answer,
  ~3 s next to three long ones, less for the 0.5-2K-token tool results agents send. What is left: a pass over all the
  weights costs ~0.25 s however few prompt tokens it carries, so smaller steps buy little; the
  recurrent-state checkpoints go through host memory (19-31 ms each, up to ~0.9 GB of RAM per
  conversation in use); a conversation read back from the disk tier holds the others for ~1.4 s.
- With MTP, a long answer after a checkpoint rewind can part from the first pass at a near tie. The
  target's rewind is exact - with MTP off, 400 tokens after rewinding a 93K-token prompt are bitwise
  the first pass - but with it on the two part at token 293 on a -1.615 / -1.628 tie that the draft's
  acceptance put in a different batch shape. The draft context after a rewind is the suspect
  (`rewind` in `docs/results/disk-tier-v3-20260924.json`).
- Graph reuse on speculative decodes is ~45%: the draft context alternates between two batch shapes
  against one cached graph result. More than one live result needs scheduler surgery.
- `set_input_kq_mask` scans a conversation's whole window once per decode (~0.9 ms at 85K cells). Since 0.3.0 a
  step of several conversations reads the cells' membership bits once for all of them instead of once each
  (four conversations over ~112K cells: the step's inputs 2.33 -> 1.75 ms); the scan itself stays.
- About 43 ms of the 85K pass is weight bandwidth and does not move without changing the file.
- Decode has roughly 2.5× of unused machine *with speculation off*: 1 stream 19.0 tok/s aggregate,
  2 streams 30.6, 4 streams 47.1, because the weight bytes are read once per batch regardless of
  how many sequences share it. It is the same lever speculation pulls for a single user, and the
  two do not stack: **with MTP on, four streams give 57.7 tok/s against 37.5 for one (1.54×)**,
  each stream at 14 tok/s, and eight streams add nothing (62.9). A step costs ~30 ms fixed plus
  ~1 ms per distinct routed expert it touches (512 experts, 10 per token, 116 MB each), so
  draft tokens and other users' tokens draw on the same budget. Before `apply_iq3s_mmq_hip` that
  made MTP lose past four users (62.9 against 71.4 at eight streams); with it the two tie (71.8
  against 72.7), and a per-step draft budget that thins the draft under load measured nothing, so
  the fixed draft of 3 stays at every stream count (`docs/results/concurrency-mtp-20260921.json`). Measured per
  kernel afterwards, the expert path was already near the ceiling in the one-user configuration;
  what was slow was IQ3_S through MMQ (the multi-user batch sizes) and a tiling cliff above 16
  tokens per step — `apply_iq3s_mmq_hip` fixes both: eight users with MTP 60.5 → 70.1 tok/s.
- **Several users at depth.** A ubatch that serves several slots carries several sequences; until
  `apply_multi_stream_qsa` the sparse path declined it and ran dense attention over the whole used
  pool, so four users each 35K deep got 28.3 tok/s together at 99 ms/token, less than one user alone
  (31.7). With mixed ubatches on the sparse path they get 33–39 tok/s at 70–87 ms/token, two users
  34 (was 22), and one user is 7% faster at depth (26.8 ms/token) because a multi-stream step no
  longer switches the block-key cache off for good. The mixed reserve graph lost its dense mask with
  it: four slots at 262144 × 8192 load and cost 1.3 GB over one slot (4096 used to cost 9 GB and
  8192 did not load), so the default `-np 1` is only about the ~1 GB and the slower prefill of a
  shared pool. Since 0.2.6 four slots are the default, and image input works with several.
- **What the prompt cache cost in system RAM.** `--cache-ram` defaults to 8192 MiB and the manager never
  set it, so the server held ~8 GB of system memory for cached conversation state on a machine whose GPU
  carve leaves 31.6 GB for the whole desktop — the KV cache is in the carve, this was the RAM tier above
  the disk one. Turning it down needed a change first: `alloc()` refuses a state larger than the limit and
  `persist()` only ever ran on a state `alloc()` had accepted, so a small limit would have quietly stopped
  every long conversation from reaching disk. With `persist()` taking the buffers directly and
  `--cache-ram 1024`, the same four conversations cycled through one slot leave the server at
  **4.13 GB resident instead of 8.07 GB**, and finish 83.4 s → 57.5 s faster because the disk reads
  stop competing with the cache for memory. Two defects surfaced with it: the disk tier
  ranked candidates by `f_keep`, so a short entry that was a complete prefix (`f_keep = 1.000`) shadowed
  every longer entry of the same conversation — the 79K-token chat loaded the 49370-token entry and
  prefilled 30071 tokens (46.3 s) where the right entry costs 2897 (**17.3 s, against 96.6 s cold**) — and
  reopening any cached conversation with MTP **off** hit `GGML_ASSERT(ctx_dft)` and killed the server. An
  entry written with MTP on is now usable without it (the draft half is dropped); the reverse is refused,
  because nothing can prime a draft context for a sequence the target is already deep into. The disk
  ceiling is a setting now, since one long conversation is 5.6 GB. Details:
  `docs/results/prompt-cache-20260922.json`.
- **Why a cache hit still cost 17 s.** Of a hit on the 79K-token conversation, 12.1 s was reading the
  5.630 GiB entry and 5.5 s was replaying 2897 tokens. The read ran at 505 MB/s; the same file read
  with `FILE_FLAG_NO_BUFFERING` comes off the drive at **3451 MB/s**, and a copy written in one pass
  reads no faster, so neither fragmentation nor the drive explains it. It was the cached I/O path:
  every GiB is copied into the page cache on the way past, which buys nothing for data that is read
  once and handed to the GPU, and costs dearly when the carve leaves 31.6 GB of system memory. Reading
  through one aligned staging buffer gives **1656 ms at 3482 MB/s**. Those end-to-end figures replayed
  the cached prompt unchanged; continuing a conversation - a new message appended - restores 79K tokens
  and processes only the message: **4.2 s** against 96.6 s cold. Resending a prompt the cache already
  ends on costs a replay on top: sampling needs logits for the last token, and a GDN recurrent state
  cannot be rewound by one token, so the server restores the nearest context checkpoint and replays
  from there - 2500-3600 tokens when the resent prompt includes the last answer, as that probe's did,
  but **4 tokens for a regenerate from the app**, which resends the messages without the answer
  (upstream keeps a checkpoint 4 tokens before every prompt's end; `tmp/regen_probe.py`). Those
  checkpoints are also ~3.3 GB of the 4.15 GB the server holds with one long conversation resident.
  Details: `docs/results/prompt-cache-20260922.json`, `regenerate` in `perf-round-20260923.json`.
- **Conversations stay put, and go to disk in blocks.** With more than one slot, upstream's
  `--cache-idle-slots` saved and cleared every idle slot on each new task, so switching between two
  long conversations read one back (2.6 s) and wrote the other out (3.4 s) on the main loop - ~5-6 s a
  switch although the KV cells were already allocated. With `--no-cache-idle-slots` switches take
  **0.3-0.4 s**, and conversations reach disk instead in blocks: once one has grown by 4096 tokens and
  the server is idle, the state is gathered (~0.1 s) and a background thread writes only what changed.
  Checkpoints never change once made, and the KV grows by rows appended to every layer, so
  content-defined chunks of the serialised state find the old bytes wherever they moved: a 79K-token
  conversation grown by 600 tokens wrote **0.52 GiB instead of 5.7**, and restoring it after a kill gave
  the same tokens as the warm slot. Idle slots hand their checkpoints' bytes back to the store and a
  rewind reads one back in ~40 ms (working set 5.86 -> 2.78 GB; token-identical to not paging). A
  completion makes room in the pool first, so kept conversations never make a restore fail. Details:
  `docs/results/disk-tier-v2-20260923.json`. A restore reads the chunks with four threads and leaves
  the checkpoints in the store, pinned and paged in by a rewind like idle ones: the 79K-token
  conversation came back as **2.33 GiB in 0.73 s instead of 5.63 GiB in 2.72 s**, its next reply
  4.75 → 2.57 s, the server at 2.74 GB instead of 4.52 (`restore` in `perf-round-20260923.json`).
