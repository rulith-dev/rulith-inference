# Measured and rejected

Ideas that were built and measured, and did not earn their place. They are here so they are not
rebuilt — several look obviously right and are not.

## Speculation

**Batching the per-query gather in the sparse attention path.** Removes ~900 nodes from a 4-token
graph. Exactly neutral: 83.8 ± 6.0 ms per pass per-query against 83.6 ± 5.7 batched, same binary,
same draft, same prompt. With HIP graphs replaying, node count is nearly free. This one first
appeared to be +10% — that was draft acceptance moving from 68% to 78%, not the change.

**Sparse attention on the draft's own context** (`LLAMA_MTP_QSA_MIN_T=1`). Slower. A single layer's
indexer costs more than the 0.8 ms dense read it replaces.

**Folding the catch-up into the first draft step.** No time saved, perturbs the drafts, and a
deferred stash breaks the batch position check when the cache is cleared.

**FR-Spec vocabulary pruning** (restrict the draft head to the frequent 64K). Chinese token ids sit
at 95K-131K in this vocabulary, so a 64K set covers only 84-96% of held-out text. The head is
bandwidth-bound, so a pruned head is genuinely cheaper — but not on text this model is used for.

## Kernels and quantisation

**A counting sort for expert ordering.** Reported elsewhere as +4.6-8.1%; here it lost ~4% (904/906
against 946). The atomic per-expert cursor serialises the scatter, and a 512-wide scan costs more
than a 1024-element all-LDS bitonic sort with no atomics.

**Elementwise chain fusion.** Correct and bit-identical (self-check finds 0 mismatches; 600 greedy
tokens identical), and worth at most ~1 ms/token — inside the decode probe's noise. It also loses to
the specialised binary kernels on prefill-sized tensors, so it is bounded to 262144 output elements.
Kept, because it is free and correct, but it is not a win.

Two traps it produced are worth keeping:

- **Folding a REPEAT that runs *before* the chain into a read of its source is a use-after-free** in
  allocator terms: the source's lifetime ends at the REPEAT node and the allocator hands that memory
  on. Only fold REPEATs inside the chain.
- The verification harness lied twice. A null-stream copy is not ordered against the non-blocking
  compute stream; and recomputing a chain "plain" *after* the fused write is invalid when the output
  is allocated in place over an operand. Snapshot operands, run plain first, restore, then fused.

**Requantising the trunk to IQ4_NL.** +10-12% perplexity. Double quantization: IQ3_S at 3.4 bpw
cannot be restored at 4.5 bpw. The same experiment on the whole model proved the prefill hypothesis
— an all-IQ4_NL file reached 1020 t/s against 903 — and must not ship for the same reason.

**Q6_K trunk, as a file or as in-memory twins.** A real trade-off, not a win: perplexity-neutral,
decode -7%, prefill -6%. The in-memory version costs +2.9 GB of commit, which at ctx 262144 / ubatch
8192 is enough to push production over: prefill fell to 816 t/s and a multi-turn chat died at turn 2
with `bad allocation`. Shipped inert, usable at ctx ≤ 131072, default off. The better design is a
native Q6_K dequant path in the matrix-core kernel rather than twins.

**Wider tiles for the IQ3_S expert gate/up.** The dequantization is paid per tile, so a wider token tile
would amortize it over more WMMAs - but gate and up keep 2 x 64 x BN F32 accumulators a block. BN 256
spills (41.5 ms against 25.2), BN 192 fits only at 4 waves per SIMD and is slower (27.1), BN 64 too
(28.4). Prefetching two steps ahead did nothing for the small tiles and spills the big one.

**Wave-local syncs in the MMB epilogue.** Each wave stages its fragments in its own 1 KB, so the two
block barriers per fragment are more than the epilogue needs. Bitwise the same, and no measurable change
(gate/up 20.9 vs 20.9 ms, down 12.7 vs 12.8). Neither did hoisting the valid-fragment dispatch out of the
IQ4_NL down projection's step loop (12.01 vs 12.01 ms): its tiles are ten steps deep, and its F32 output
write alone is ~11% of it.

**Sparse attention in the MTP draft beside idle conversations.** The draft attends densely over the
whole pool, idle conversations included, so the sparse path looked like a way out: `LLAMA_MTP_QSA_MIN_T=1`
gave 85.5 ms a pass against 85.1 dense (one slot: 82.6 against 81.0).

**Rebuilding the block keys after a conversation moves.** Correct, but a batched rebuild does not round like
the keys built as blocks complete: the moved conversation's second token went from -0.154 to -0.231. The
keys now move with their cells. Tracking "rebuild every key" as one flag for the pool was worse still: it
had to widen the window to the pool, and only a graph of 128+ tokens clears it, so after any restore the MTP
draft viewed the whole pool for the rest of the generation (+5 ms a pass). It is per sequence now.

**Moving conversations through `memory_update`.** Returning FAILED_PREPARE so that `llama_context::decode`
runs `memory_update(true)` and retries is the upstream route for cache maintenance, and it worked, but it
re-reserves the worst-case graph every time: 353 ms for the target. The moves are queued in `init_batch` on
the stream the graphs run on instead.

**Starting a new conversation in the middle of the largest free run.** The first layout of the regions round
gave every conversation room to grow that way, and moved one that ran out of room to a run twice its size.
It spread four conversations over the whole pool, and a step of several conversations views the span from the
first of them to the last: the mixed steps' host inputs took 0.5-0.7 ms more, and the MTP draft's dense
attention read the gaps. Conversations are now packed in order with a fixed room after each - four of ~24K
tokens sit in ~121K cells - and a batch that does not fit has the pool laid out again.

**F16 weights for the routed experts (0.4.3).** gufo, the fastest prefill on this chip (Linux), dequantizes the
expert weights to F16 for its WMMA and keeps the codes in LDS. Tried in glu3 with the same tiles: the output was fine
(not bitwise; the same perplexity was the bar), but glu3 got 5% slower. The dequantization is ~11% of the 2K-token
kernel (switches that skip each part, garbage output: WMMA 22%, LDS fragment reads ~1.1 ms, LDS writes and barriers
~0.5 ms, the skeleton ~1.8 ms), so a cheaper conversion had little to win, and the F16 path's extra LDS traffic cost more.

**Waves along the expert rows.** A wave owning 16 rows, its weights dequantized straight into WMMA registers and the
half-waves' duplicate operands exchanged with `permlanex16`, as gufo's waves along M: correct, but +16% (gate/up) and
+38% (down) at 2K tokens. The activations each wave then reads again outweigh the LDS writes saved.

**BF16 copies of the Q8_0 dense weights for prefill.** +10% slower: the dense GEMMs are bound by memory and LDS, not
by the Q8_0 dequantization, and the copies double their reads. Nor did `LLAMA_MMB_TALL=1` (slower) or wave barriers
in the MMB store epilogue (no change) help.

## Configuration

**A 64 GB carve.** Worse on both axes than 96 GB and now dead: the model does not fit, ~9.8 GB spills
into shared memory, and the driver pages during decode. 839 vs 942 t/s prefill, 49.6 vs 38.6 ms/token
decode, 17-19 vs 26-30 tok/s multi-turn — measured with the 64 GB arm at the *higher* power limit.
The user's own 84K chat: 96 GB 854.0 t/s / 58.5 ms against 64 GB 752.6 / 82.1.

**`parallel > 1` as a default.** Costs ~12 GB: the worst-case graph reserve uses a mixed-sequence
ubatch, which fails the sparse path's single-sequence visibility test and reserves the dense
per-block bias and the dense mask. Default is back to 1. (It remains free throughput for genuinely
concurrent users — see [decode-budget.md](decode-budget.md).)

**`-lzm on`** (mmap-style paging for the PLE table) is 4.2× slower than `-lzm on-direct` here.

**IOMMU off** (halogen reports +13-16% prefill, gfx1151-engine +5%) is not a lever for this stack, and
the GTR9 Pro's BIOS greys the option out anyway. Those engines read their weights in place from host
memory registered with the GPU, the path the IOMMU translates. Here the weights and the KV cache live
in the carve: with the model loaded and 33.6K tokens prefilled, the server's WDDM counters read 84.56 GB
dedicated and 1.62 GB shared (staging and pinned buffers).

**Windows power mode "Best performance", and Memory Integrity (HVCI) off.** Neither shows. With both
changed and the machine rebooted, the same 0.1.8 build: a speculative pass at short context
72.20 / 72.50 / 71.75 ms (median of 68 passes each, three runs) against 72.05 before; a 95.6K-token
real-text prefill 892.9 / 889.7 t/s against 888.3-891.9. The host work these would speed up is either
overlapped with the GPU (prefill) or a few percent of a pass that the kernel calls do not dominate
(decode). Memory Integrity is a security feature, so it is worth keeping on.

**`QSA3_FORCE=1`** is a null test, not a shortcut. It lifts the kernel's `< 128 queries` gate, but the
packed K/V operands it needs are only built under the prefill layout, so the predicate declines
anyway. Perplexity, equivalence and speed all came back identical. Feeding that kernel at decode
would mean maintaining a packed copy of the whole cache incrementally: +6.4 GB at ctx 262144.
Gathering the selected cells for the ordinary attention kernel is what worked instead.

## Upstream

**Moving past the pinned revision.** Upstream HEAD, ten commits later and none of them benchmarked
upstream, measured 752.0 t/s against 810.8 on the same config — and does not compile on Windows. The
pin is deliberate.
