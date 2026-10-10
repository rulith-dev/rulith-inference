# The patch set, and how it was audited

*This is the record of an audit run against the predecessor repository, kept because it explains why
the patch order is what it is and why five of these scripts were generated rather than written. The
tooling it names now lives in `bootstrap/bootstrap.py`.*

Audit of 2026-09-19, produced by `bootstrap/bootstrap.py --verify` and re-runnable at any
time. This exists because the repository accumulated 47 patch scripts across two backends, of which
only two were referenced from the manager or the scripts, and nothing recorded which ones produce the
binary that runs. Everything below is measured from the tree and the running process, not read off
documentation — one documentation claim turned out to be wrong (see IQ3_S).

## Upstream pin

    https://github.com/pwilkin/llama.cpp @ f5daaa3cfa6358e5dd398911ec741813745a5440

`bootstrap/UPSTREAM.json` carries the delta's provenance: for each of the 20 modified files, the
Git blob SHA-1 and SHA-256 it has upstream; for all 24, the SHA-256 the patch set produces. Those
two halves are what make the delta below provable rather than remembered — the first says what the
patches started from, the second what they must end at.

## The whole delta: 24 files

20 modified, 4 added, out of 3610.

| Area | Files |
| --- | --- |
| CUDA/HIP backend | `ggml-cuda.cu`, `mmb.cu`, `mmb.cuh`, `mmid.cu`, `mmvq.cu`, `softcap.cu`, `softcap.cuh`, `vecdotq.cuh` |
| added CUDA/HIP | `strixllama-chain.cu/.cuh`, `strixllama-getrows-cast.cu/.cuh` |
| llama core | `llama-context.cpp`, `llama-graph.cpp`, `llama-lazy-reader.h`, `llama-memory-hybrid-idx.cpp/.h`, `llama-memory-recurrent.cpp/.h`, `llama-model.cpp/.h` |
| model | `models/qwen4exp.cpp` |
| speculation / server | `common/speculative.cpp`, `tools/server/server-context.cpp` |

## The 17 patch scripts that are live

Verified by running each against a throwaway copy of the four source directories the patches touch
and keeping those that report "already applied". Each script gets a fresh copy, so one cannot mask
another.

    apply_chain_fusion            apply_qsa_block_key_cache
    apply_decode_timing           apply_qsa_decode_gather
    apply_getrows_cast_fusion     apply_qsa_kb_image_guard
    apply_graph_key_shape         apply_qsa_small_batch_mask
    apply_hip_ple_probe           apply_rs_pos_warn_stateless
    apply_iq3s_vecdot_hip         apply_skip_ops_ablation
    apply_mmvq_rdna35_rows        apply_spec_draft_ubatch
    apply_ple_overlapped_reads    apply_spec_timing
    apply_trunk_decode_twins

Six files are written by more than one script, so order matters when replaying onto a clean tree:

| File | Scripts, in the order they must run |
| --- | --- |
| `ggml-cuda.cu` | chain_fusion, getrows_cast_fusion, graph_key_shape, skip_ops_ablation, hip_ple_probe |
| `models/qwen4exp.cpp` | qsa_block_key_cache, qsa_decode_gather, qsa_small_batch_mask, qsa_kb_image_guard, hip_ple_probe |
| `llama-memory-hybrid-idx.cpp/.h` | qsa_block_key_cache, qsa_kb_image_guard |
| `strixllama-chain.cu/.cuh` | chain_fusion, getrows_cast_fusion |

Of the other 30 scripts, 29 do not apply to this fork at all — they target the Vulkan backend or
files this tree does not have — and one (`apply_mtp_pending_reset`) applies cleanly, meaning it is
not in the build.

## The gap: 5 modified files no script owns

`mmb.cu`, `mmb.cuh`, `softcap.cu`, `softcap.cuh`, `mmid.cu`.

**`mmid.cu` is a comment only**: a note recording that a counting sort was tried in place of the
bitonic expert sort and lost (906 against 946 t/s pp16384). No code change.

**The other four are the IQ3_S MMB kernel and the sigmoid fusion, and they are live in production.**
They exist only as whole-file copies in `patches/iq3s-kernel/`, with no `apply_*.py` script, so a
clean rebuild from the patch scripts alone would silently omit the single largest measured win in
the project: +4.8% prefill and +5.5% decode (pp16384 947.37 against 903.64, tg128 24.97 against
23.66), from letting the 52.2% of the model body stored as IQ3_S reach the matrix cores.

`patches/iq3s-kernel/README.md` states "It is not in production". That is wrong, on three
independent pieces of evidence:

1. The four files in that directory are byte-identical to the production tree.
2. `mmb.cu:921` passes `allow_iq3s = true` from the fused-GLU caller, with no runtime gate.
3. The running server logs `MMB_GLU fused gate/up+swiglu: ... type=IQ3_S`.

### Settled 2026-09-19: the rollback reason is obsolete, keep the kernel

The rollback was attributed by elimination, not root cause, and predates the real fix. On 2026-09-18
every "answer stops mid-sentence" report was traced to speculative verification batches running dense
attention with no causal mask, fixed by `apply_qsa_small_batch_mask.py`, which is live.

Re-measured on the current build. `STRIX_MMB_IQ3S=0` was added to `mmb.cu` so both arms are the
same binary with one variable; `tools/truncation_test.py --fill 70000 --reps 1`, fresh server per
arm, verified by `type=IQ3_S` appearing in one server log and not the other:

| | 68499-token prefill | answer | sections |
| --- | --- | --- | --- |
| IQ3_S on | **876.7 t/s** | 1110 tokens, `finish_reason=stop` | **30/30** |
| IQ3_S off | 830.6 t/s | 1085 tokens, `finish_reason=stop` | **30/30** |

Both reach 30/30: the truncation is gone. Prefill is **+5.6%**, matching the +4.8% originally
claimed. The kernel stays, and `STRIX_MMB_IQ3S=0` stays as the switch to reach for when a future
numerics or long-generation regression has to be attributed.

(The decode rates in that test are not a comparison - it is a speculative run and the two arms drew
different draft counts.)

## The deeper gap: the scripts cannot rebuild the tree on their own

`ggml-cuda.cu` also carries unscripted content, and the counts line up exactly with the snapshot in
`patches/iq3s-kernel/`, not with anything the live scripts insert:

| marker in `ggml-cuda.cu` | production tree | all 17 live scripts | iq3s snapshot |
| --- | --- | --- | --- |
| `GGML_UNARY_OP_SIGMOID` | 11 | 0 | 11 |
| `scale_sigmoid` | 2 | 0 | 2 |
| `NO_SIGMOID_FUSE` | 2 | 0 | 2 |
| `LLAMA_GRAPH_DUMP` | 3 | 0 | 3 |
| `GT compute` | 1 | 0 | 1 |
| `mmb_supported_mmid` | 3 | 0 | 3 |

So the production tree is **snapshot first, scripts second** - the 17 scripts layer onto
`patches/iq3s-kernel/`, not onto clean upstream. `GT compute` is the `LLAMA_GRAPH_TIMING` line the
whole decode-budget analysis is built on, and it exists only in that snapshot.

A bootstrap therefore has three steps, and this order is not optional:

1. check out `pwilkin/llama.cpp` at `f5daaa3...`;
2. overlay the six files in `patches/iq3s-kernel/` (whole files, hash-verifiable);
3. apply the 17 live scripts, respecting the ordering table above.

## Replay result, 2026-09-19: 24 of 24 — the bootstrap is proven

    clean-upstream check: ok
    patches   : 21 applied, 0 failed
    REPRODUCED 24 / 24

Every file of the delta is byte-identical to the production tree, starting from a clean upstream
reconstructed from hash-verified downloads. `bootstrap/bootstrap.py --verify` re-runs it in a few minutes.

Getting there needed one ordering fix and five new scripts:

- **`apply_win_lazy_reader` must run first.** It installs the cross-platform lazy reader that later
  patches anchor against. It had looked dead because it was being run after the snapshot handed it
  its own finished output.
- **The snapshot must not supply `llama-lazy-reader.h`.** Same trap from the other side: give
  `apply_ple_overlapped_reads` the finished file and it reports "already applied" and skips the half
  of its work that lives in `qwen4exp.cpp`.
- Five scripts generated by `tools/make_patch_script.py` from the tree itself, with anchors grown
  until unique, replacing what had drifted or was never captured:

| New script | Closes |
| --- | --- |
| `apply_node_timing` | `STRIX_NODE_TIMING`, which existed only as an uncommitted edit |
| `apply_iq3s_vecdot` | supersedes `apply_iq3s_vecdot_hip`, whose anchors matched nothing |
| `apply_trunk_twins` | supersedes `apply_trunk_decode_twins`, written against a modified tree |
| `apply_ple_prefetch_launch` | the PLE prefetch half no script performed |
| `apply_mmid_sort_note` | the comment recording why the expert sort stays bitonic |

All five are idempotent, report "already applied" against the production tree, and modify nothing
when re-run.

### The audit that got here (kept for the reasoning)

`tools/replay_bootstrap.py` does exactly that and is re-runnable. Only 20 files differ from
upstream, so only those are restored: normally out of the clone's own git objects, addressed by the
blob hash in `bootstrap/UPSTREAM.json`, or with `--fetch` through a mirror and checked against the
recorded SHA-256 — all 20 verified either way. The reconstructed tree then matches upstream in every recorded file,
the snapshot goes on, the scripts run in the order above, and every delta file is compared.

    clean-upstream check: ok
    patches   : 15 applied, 2 failed
    REPRODUCED 17 / 24

So most of the build is now provably reproducible, and the seven that are not each have a named
cause. None of them is guesswork:

| File(s) | Why it does not reproduce |
| --- | --- |
| `ggml-cuda.cu` | `STRIX_NODE_TIMING`, the per-dispatch HIP event instrumentation, is in **no script and not in the snapshot** — an unrecorded edit straight into the tree |
| `mmid.cu` | comment-only note about a rejected counting sort; unrecorded, harmless, fold into the snapshot |
| `vecdotq.cuh` | `apply_iq3s_vecdot_hip` is stale: its `OLD_LOOP` anchor occurs in neither upstream nor the production tree, so the script can no longer produce what the tree contains |
| `models/qwen4exp.cpp` | the PLE prefetch half (`launch_prefetch` / detached thread) has no working script; `apply_hip_ple_probe` reports "already applied" and writes nothing |
| `llama-model.cpp`, `llama-model.h`, `llama-graph.cpp` | `apply_trunk_decode_twins` fails with "anchor not unique" against clean upstream — written against an already-modified tree |

All five were closed the same day; see the result at the top. Had they not been, a clean rebuild
would have silently omitted the node timing, the IQ3_S vec-dot, the PLE prefetch and the trunk
decode twins — which is the whole reason this audit exists.

    python bootstrap/bootstrap.py --fetch --patch    # download upstream once, then replay
    python tools/replay_bootstrap.py            # replay again offline

## How to re-check

    python bootstrap/bootstrap.py --record     # hash what the patch set produces
    python bootstrap/bootstrap.py --verify     # is this tree still exactly that?

A file that appears under `--delta` and is named by no live script is a reproducibility gap: the
build cannot be recreated from the patch set alone.

## Addendum 2026-09-21: `apply_iq3s_mmq_hip`

The delta is now 27 files (23 modified, 4 added); the replay reports 27 / 27. One script, four files,
last in the order because it rewrites the sign table `apply_iq3s_vecdot` added:

| file | what |
|---|---|
| `ggml/src/ggml-cuda/vecdotq.cuh` | the IQ3_S sign-byte mask becomes two multiplies (`strixllama_iq3s_sign_mask(s4)`) instead of a 16-entry device table; the mmvq vec-dot uses it |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | the IQ3_S MMQ tile loader on HIP: no `__vcmpne4`/`__vsub4` byte loops, grid read from the LDS copy |
| `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q_process_tile` fills the LDS grid after the x tile (`mmq_get_nbytes_shared` reserves it); the fork's compact routed MoE path admits IQ3_S and IQ4_XS, not only IQ4_NL |
| `tests/test-backend-ops.cpp` | `STRIX_MOE_PERF` perf cases for this model's expert shapes (512 experts, 10 used, 640x2560 / 2560x640) |

Measured with `test-backend-ops perf -o MUL_MAT_ID` under `STRIX_MOE_PERF`: IQ3_S gate/up 123 -> 149 GB/s
at 8 tokens per step, 117 -> 157 at 16, 73 -> 156 at 32; IQ4_XS at 32 tokens 74 -> 170. See
`docs/results/concurrency-mtp-20260921.json` for the decode-level effect.

`tools/make_patch_script.py` learned to merge hunks whose anchors overlap after an earlier hunk has
been applied, and to replay its own output against the before-tree before writing it - the first
version of this script had four anchors in one loop body, of which the second no longer matched once
the first had been applied.

## Addendum 2026-09-21: `apply_prompt_cache_disk`

The delta is now 29 files (25 modified, 4 added); the replay reports 29 / 29. Three files, last in
the order because its server-loop anchors sit in text earlier scripts wrote:

| file | what |
|---|---|
| `tools/server/server-task.h` | `server_prompt_cache` gains a disk tier: an index of on-disk entries, `set_disk`, `persist`, `load_from_disk` |
| `tools/server/server-task.cpp` | the file format (tokens, target and draft state, checkpoints), writing on every RAM-cache insert, prefix-matched reads on a RAM miss, size-bounded eviction oldest first |
| `tools/server/server-context.cpp` | `STRIX_PROMPT_CACHE_DIR` / `STRIX_PROMPT_CACHE_MIB` wire it up after the RAM cache is created; `prompt_save` persists what it just captured |

Why it is a tier under the RAM cache and not a slot-file feature: the slot save/restore endpoints
carry the state but not the recurrent checkpoints, and without a checkpoint the server has to
re-process a hybrid model's prompt from the start ("forcing full prompt re-processing due to lack of
cache data") - measured: restore in 0.5 s, then 38 s of prefill anyway. The RAM cache entry is the
complete unit, so that is what goes to disk.


## Addendum 2026-09-21: `apply_multi_stream_qsa`

The delta is now 30 files (26 modified, 4 added); the replay reports 30 / 30. Four files, 38 hunks,
last in the order because its anchors sit in text the earlier scripts wrote:

| file | what |
|---|---|
| `src/llama-memory-hybrid-idx.h` | `qsa_mixed_inputs` (two membership inputs) threaded through `set_input_qsa_blocks` |
| `src/llama-memory-hybrid-idx.cpp` | the block-key cache gap check per token and per sequence; `qsa_scalar_visibility` accepts several sequences; the compact fill writes `seq_blk` / `seq_tok` |
| `src/models/block-graph.inc` | `qwen4exp_apply_compact_visibility` folds `seq_blk^T seq_tok` (exactly 0/1) into the visibility |
| `src/models/qwen4exp.cpp` | the decode gather batched on the flash-attention sequence axis with f16 rows (limit 32 queries); the membership inputs built, set and checked for reuse; `dirty_max` grows with the sequence count |

Why: with more than one slot the server's equal-length split puts several sequences in one ubatch.
The sparse path declined those, so they ran dense attention over the whole pool (four users 35K
deep: 28 tok/s together against 32 for one alone), a multi-stream step tripped the sticky block-key
cache gap flag for every stream, and the mixed reserve graph kept a dense f16 mask of n_kv × ubatch
that the driver would not fill at 262144 × 8192. Measured in `docs/results/concurrency-mtp-20260921.json`
(`multi_stream_qsa_20260921`): four users 35K deep 33–39 tok/s, four slots at 262144 × 8192 load
and cost 1.3 GB over one. `src/models/block-graph.inc` joins the upstream side of the delta
(`bootstrap/UPSTREAM.json`).

## Addendum 2026-09-21: `apply_small_m_mmvf`

Three files already in the delta (still 30 files, replay 30 / 30), 10 hunks:

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | a few-row F32/F16 src0 (≤ 64 rows) with 9–64 columns runs the vector kernel over column chunks of 8 instead of falling through to cuBLAS |
| `tools/server/server-context.cpp` | `STRIX_SPEC_TIMING` splits "other" into pre / post / gap |
| `tests/test-backend-ops.cpp` | `STRIX_DENSE_PERF` (the big Q8_0 projections, 8 distinct matrices so the working set beats the MALL) and `STRIX_SMALL_PERF` (the few-row shapes) |

Why: a 16-token verify step (four slots) ran the hyper-connection inject `[10240 × 4]` at 117 µs
and the GDN beta/alpha `[2560 × 48]` at 37 µs per call through cuBLAS — 95 + 72 calls per pass —
because `mul_mat_f` declines a src0 narrower than its row tile and the vector kernel stops at 8
columns. With chunks: 13 and 10 µs; the bare 16-token pass 139.8 → 130.4 ms. Measured in
`docs/results/concurrency-mtp-20260921.json` (`pass_budget_20260921`).

## Addendum 2026-09-22: `apply_cache_ram_and_mtp`

Three files already in the delta (still 30 files, replay 30 / 30), 17 hunks, after
`apply_prompt_cache_disk` because it rewrites the tier that one adds:

| file | what |
|---|---|
| `tools/server/server-task.h` | `persist()` overload taking the buffers, `has_disk()`, `disk_wants()` |
| `tools/server/server-task.cpp` | persist from buffers the RAM tier does not own; a media guard that can fire; rank disk candidates by the tokens they skip; say why a scan drops a file and remove one that can never be read; print what was on the shelf when nothing is taken; keep the target state when the entry has a draft one and this server does not draft, and refuse the reverse |
| `tools/server/server-context.cpp` | `prompt_save()` gathers into a temporary and writes to disk when the RAM tier declines the state |

Why: `--cache-ram` defaults to 8192 MiB, which was ~8 GB of system memory on a machine whose carve
leaves 31.6 GB, and it could not simply be turned down — `alloc()` refuses an over-limit state and
`persist()` only ran on states `alloc()` accepted, so a small limit would have stopped long
conversations reaching disk at all. Measured in `docs/results/prompt-cache-20260922.json`: resident
8.07 GB → 4.13 GB over the same four conversations, the 79K-token one 96.6 s of prefill → 17.3 s,
and MTP off no longer aborts the server on a cached conversation.

## Addendum 2026-09-22: `apply_spc_direct_io`

One file already in the delta (still 30 files, replay 30 / 30), 13 hunks, after
`apply_cache_ram_and_mtp` because it replaces the reader that patch leaves behind:

| file | what |
|---|---|
| `tools/server/server-task.cpp` | `spc_file`: unbuffered sequential reads through one aligned 8 MiB staging buffer on Windows, `std::ifstream` elsewhere; the read log line splits file time from buffer time; a file that cannot be opened is no longer treated as corrupt and deleted |

Why: a 5.630 GiB entry read at 505 MB/s while the drive gives 3451 MB/s unbuffered, and a copy of the
file written in one pass reads no faster — the cost was the page cache taking a copy of every GiB of
data that is read once and handed to the GPU. Now 1656 ms at 3482 MB/s, and a full cache hit on the
79K-token conversation takes 8.1 s instead of 18.1 (96.6 s cold). Note that `windows.h` defines `near`
as a macro, so the miss-line variable is `closest`.

## Addendum 2026-09-23: `apply_disk_tier_v2`

Three files already in the delta (still 30 files, replay 30 / 30), 62 hunks, after
`apply_spc_direct_io` because it rewrites the disk tier that and the two patches before it built:

| file | what |
|---|---|
| `tools/server/server-task.h` | the store's index (entries, refcounted chunks and checkpoints), the writer's queue and thread, checkpoint paging |
| `tools/server/server-task.cpp` | manifest + `ckpt/` + `chunks/` layout; content-defined chunking (gear hash, XXH3-128 names); one background writer that is the only thing that deletes; streaming conversion of version 1 entries; a startup scan that sweeps `.part` files and unreferenced objects; direct-I/O writes as well as reads |
| `tools/server/server-context.cpp` | `prompt_save` hands the state to the writer (and can skip the RAM tier); block writes when all slots are idle; checkpoints paged out when idle and back in for a rewind; `make_room` before a completion; `try_clear_idle_slots` takes the least recently used slot and writes it first |

Why: switching between two long conversations cost ~5-6 s because every idle slot was saved and cleared
on each new task, and each save wrote the whole state - 5.7 GB for a 79K-token conversation - at once.
Measured in `docs/results/disk-tier-v2-20260923.json`: switches 0.3-0.4 s, a block write 0.52 GiB,
checkpoint paging 5.86 -> 2.78 GB of working set, restores token-identical after a kill.

Before release it went through three independent reviews (concurrency, formats and crash safety,
slot lifecycle) and now also carries their fixes: checkpoints a queued save or a warm slot depends on
are pinned in the store; entries live in a directory named after the model and its file; checkpoint
files end with an XXH3-128 and are read against the manifest's sizes; a v1 conversion places the new
manifest before dropping the old file; startup leaves alone what it cannot open; the writer wakes the
main loop after each job; see `review_20260923` in the results file.

`tools/make_patch_script.py` changed with it: when fine hunks do not replay, nearby hunks are merged
with a doubling gap before falling back to one hunk for the whole span. This patch was 12247 lines as
one hunk and 2243 as 18; with the review fixes it is 2652 lines as 62.

## Addendum 2026-09-23: four patches after 0.1.7

The delta is now 33 files (29 modified, 4 added): `ggml/src/ggml-cuda/qsa.cu`, `ple-conv.cu` and
`include/llama.h` join the upstream side of `bootstrap/UPSTREAM.json`. Replay 33 / 33, 31 patches.
Measured in `docs/results/perf-round-20260923.json`.

| patch | files | what |
|---|---|---|
| `apply_hc_q8_fusions` | `mmb.cu`, `ggml-cuda.cu`, `ple-conv.cu` | the HC gate kernel (GEMM + sigmoid + stream mix) and the tall 384-row tile for Q8_0 weights, and the PLE conv fusion for an F32 weight - Unsloth's types, which left `LLAMA_HC_GATEMIX` and `LLAMA_PLE_CONV` inert. The Q8_0 gate kernel reads the streams in F32 and keeps the gate in F32, so it is bitwise the unfused path; the PLE match also refuses a graph in which anything outside the fused taps reads the concat's body |
| `apply_qsa3_bitonic` | `qsa.cu` | QSA3 selection rows sorted with a bitonic sort instead of an O(ns^2) rank sort (after pwilkin/llama.cpp 38477e8c); same rows |
| `apply_disk_restore_lazy` | `server-task.h`, `server-task.cpp`, `server-context.cpp` | a version 2 entry's chunks read by four threads, its checkpoints left in the store and pinned for the slot like paged-out ones: 5.63 GiB / 2724 ms became 2.33 GiB / 734 ms for a 79K-token conversation |
| `apply_ple_pregather` | `llama.h`, `llama-context.cpp`, `llama-lazy-reader.h`, `qwen4exp.cpp`, `server-context.cpp` | `llama_strix_prefetch`: the server gathers the next prompt batch's PLE rows while the current one computes, and that batch's `set_input` takes them; the prefetch hook, which `llama_decode` calls for every model, no longer casts a model of another architecture |

Every one of them was checked for identical output: 48 greedy tokens with their top-5 logprobs for the
fusions and the sort, the tokens of a continue / rewind / return sequence for the restore, the generated
text of a 95.6K-token prefill for the pregather.

## Addendum 2026-09-23: `apply_state_copy_trim`

`ggml/src/ggml-cuda/gdn-conv.cu` joins the delta (34 files, 30 modified, 4 added); replay 34 / 34, 32
patches. Two hunks: `build_conv_state_at` copies each rollback slot's conv-state tail straight out of
the concat instead of through a cont (144 fewer dispatches in a 4-token verify pass), and the GDN conv
fusion's matcher accepts that copy as a reader of the tail, so prefill keeps the fusion. Pure copies:
the output is bitwise the same; a verify pass measured 59.40 -> 58.90 ms.

## Addendum 2026-09-23: `apply_moe_glu3`

No new files in the delta (34: 30 modified, 4 added); replay 34 / 34, 33 patches. Measured in
`docs/results/moe-glu3-20260923.json`.

| patch | files | what |
|---|---|---|
| `apply_moe_glu3` | `mmb.cu`, `test-backend-ops.cpp` | `mmb_tile_gemm_glu3`, the IQ3_S expert gate/up + SwiGLU: the dequantization spread over the block (four threads per row), the sign on an F16 copy of the grid so the scale product is one `v_fma_mix` (exact, so the same BF16), raw bytes kept whole until the dequantization, uniform-base addressing and native-vector prefetch registers, the valid-fragment dispatch out of the step loop, big tiles from 32 rows. 35.9 -> 19.9 ms a layer at 8192 tokens on a recorded routing; `STRIX_MMB_GLU3=0` runs the previous kernel, `STRIX_MMB_GLU_DUMP` records a routing. test-backend-ops: the `MOE_GLU` case (`STRIX_MOE_GLU_PERF`, also in test mode) and `STRIX_MOE_IDS_FILE`, which replays a recorded routing |

Checked for identical output: 48 greedy tokens with their top-5 logprobs at every step, and the text
generated after a 95.6K-token prefill. The prefill of that text went 889.7-892.9 -> 969.4 t/s.

## Addendum 2026-09-23: two patches for several slots

`src/prefix.h` joins the delta (35 files: 31 modified, 4 added); replay 35 / 35, 35 patches. Measured in
`docs/results/multi-slot-20260923.json`.

| patch | files | what |
|---|---|---|
| `apply_qsa_active_blocks` | `llama-memory-hybrid-idx.cpp`, `.h`, `qwen4exp.cpp`, `prefix.h` | with several slots the cache is one pool, and the sparse-attention block list of a batch held every conversation's blocks, the idle ones only to be marked invisible. It now holds the batch's own sequences (`qsa_active_blocks`), sized from their positions: one conversation beside 70K tokens of idle ones 47.6 -> 44.8 ms/token (one slot: 44.1), a prefill beside them 999 -> 1064 t/s. Bitwise the same for a conversation with at least the selection budget of blocks; a shorter one can move in the last bits. `LLAMA_QSA_ACTIVE_BLOCKS=0` turns it off |
| `apply_state_read_coalesce` | `llama-context.cpp` | a slot's state is read from the device one run of nearby cell ranges at a time instead of one range at a time: saving a conversation decoded alongside others took 6-10 s with the whole server waiting; 347 -> 44 ms in the probe, the same bytes |

## Addendum 2026-09-23: one run of cells per conversation

`src/llama-kv-cells.h`, `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h`, `src/llama-graph.h` and
`ggml/src/ggml-alloc.c` join the delta (40 files: 36 modified, 4 added); replay 40 / 40, 38 patches.
Measured in `docs/results/kv-regions-20260923.json`.

| patch | files | what |
|---|---|---|
| `apply_kv_regions` | `llama-kv-cells.h`, `llama-kv-cache.cpp`, `.h`, `llama-graph.cpp`, `.h`, `llama-memory-hybrid-idx.cpp`, `.h`, `qwen4exp.cpp`, `prefix.h` | with several slots every conversation keeps one run of the pool, in slot order: its tokens go after its last cell, a new one starts after the last with room left to the one before it, and a batch that does not fit has the pool laid out again first - the conversations keep their order and slide, each one of the batch gets the same room after it, idle ones none (device copies of K/V, indexer keys and block keys, queued on the graphs' stream in `init_batch`). A batch's graph views only the run of its own conversations. One conversation beside idle ones computes bitwise what it computes on one slot, MTP included, at the same speed (MTP 32.8 -> 38.5 tok/s beside 70K idle tokens; one slot 37.8); block keys go stale per sequence. `LLAMA_KV_REGIONS=0` turns it off, `LLAMA_KV_WINDOW=0` the window, `LLAMA_KV_REGION_MOVES=0` the moves; `LLAMA_KV_REGION_HEADROOM` sets the room |
| `apply_compute_buffer_headroom` | `ggml-alloc.c` | compute buffers get 3% + 16 MiB of headroom: a graph a few MiB larger no longer frees and reallocates the 3.4 GiB target buffer, whose commit ROCm on Windows keeps - the intermittent "bad allocation". Four conversations: peak commit 90.4 -> 83.7 GB, with 6.9 GB to spare instead of 0.18. `GGML_ALLOC_COMPUTE_PAD=0` turns it off, `GGML_ALLOC_DEBUG=1` reports reallocations |
| `apply_alloc_failure_report` | `server-context.cpp` | a failing `operator new` prints its size and call stack before the server reports "bad allocation" |

## Addendum 2026-09-24: `apply_empty_slot_cache`

No new files in the delta (40: 36 modified, 4 added); replay 40 / 40, 39 patches. Measured in
`docs/results/kv-pool-20260924.json`.

| patch | files | what |
|---|---|---|
| `apply_empty_slot_cache` | `server-context.cpp` | a slot named by id that holds nothing takes the prompt-cache path: `f_keep` was 0/0 there, a NaN that compares false, so a 173K-token conversation sent back to its emptied slot was processed again from its first token (186 s) although the disk tier held all of it; now it is read back (7 s) |

## Addendum 2026-09-24: disk tier version 3

`tools/server/server-context.h` and `tools/server/server.cpp` join the delta (42 files: 38 modified, 4
added); replay 42 / 42, 41 patches. Measured in `docs/results/disk-tier-v3-20260924.json`.

| patch | files | what |
|---|---|---|
| `apply_qsa_kb_rebuild_f16` | `qwen4exp.cpp` | the graph that rebuilds the block-key cache (a conversation's first, and the first after a restore or a checkpoint rewind) scores with the keys read back from the cache, in F16, as every later graph does - it scored with the F32 keys it had just computed, so a restored conversation and a resident one selected slightly different blocks on their next batch (same tokens, logprobs up to 0.34 apart). Perplexity at ctx 4096 unchanged to four places at ubatch 4096 and 512, and identical to the cache switched off |
| `apply_disk_tier_v3` | `llama.h`, `llama-context.cpp`, `llama-kv-cache.cpp`, `.h`, `llama-memory-hybrid-idx.cpp`, `.h`, `server-task.cpp`, `.h`, `server-context.cpp`, `.h`, `server.cpp` | `llama_strix_kv_*`: a sequence's attention rows (KV cache and indexer) read and written by position, and cells allocated for a restore. The disk tier keeps a conversation's rows in runs of 4096 positions, each written once as it fills and named by its XXH3-128; the recurrent state only when the conversation leaves memory - a short last run, the state at its end and its last prompt's two latest checkpoints - when another conversation needs its cells or `POST /strix/persist` asks (the manager does before a stop). A restore allocates the cells, streams the runs in with the next read under way and puts back the latest checkpoint they reach. The store keeps only the format the server writes for the model: entries of an older one (version 1 anywhere, version 2 where rows are served) are deleted as it opens, with the objects only they named, and the version 1 conversion is gone |

Checked bitwise against the same conversations kept resident in a pool that holds both: a 173K and a 155K
conversation swapped through a one-conversation pool, and two 33K chats taking ten turns with five
evictions, gave the same tokens and the same top-3 logprobs at every step that carries them.

## Addendum 2026-09-24: two patches after 0.1.13

No new files in the delta (42: 38 modified, 4 added); replay 42 / 42, 43 patches. Measured in
`docs/results/slot-state-guard-20260924.json`.

| patch | files | what |
|---|---|---|
| `apply_slot_state_guard` | `server-context.cpp` | GitHub issue #1. Every place the server caught an exception mid-turn released the slot and kept its conversation, although part of a batch had been recorded in its tokens and not run (or run and not recorded) - under commit exhaustion std::bad_alloc comes from ordinary work, and the next request then fed the recurrent state positions it had seen or skipped ("non-consecutive token position"); with one slot, its tokens left in a batch nobody owned aborted the server on an assert. A slot that throws now leaves the batch being built and loses its conversation, as a decode error already made it, and so does every slot the pre-decode, decode and post-decode handlers catch. Before a prompt batch the slot's tokens must match its memory, or the prompt is processed from its start. `STRIX_FAULT=<site>:<n>` (ckpt, decode, post) throws on the n-th pass, for tests |
| `apply_disk_ckpt_step` | `server-context.cpp`, `server-task.h` | a leaving conversation's older checkpoints go to the disk tier as well, one every 32768 tokens, each written once: a conversation read back from disk kept only its last prompt's, and a deeper rewind (an agent trimming an early tool result) processed it again from its first token |

With them the manager passes `--ctx-checkpoints 8 --checkpoint-min-step 32768`: a resident conversation keeps its
last prompt's checkpoints and one per 32K tokens, at most 0.9 GB of system RAM a slot instead of 3.5.

## Addendum 2026-09-24: 0.1.15

No new files in the delta (42: 38 modified, 4 added); replay 42 / 42, 44 patches. Measured in
`docs/results/vision-disk-20260924.json`.

| patch | files | what |
|---|---|---|
| `apply_disk_v3_vision` | `server-context.cpp` | with a vision projector loaded every prompt counts as a media prompt to `server_tokens::get_tokens()`, which asserts `!has_mtmd`; version 3 of the disk tier called it to compare a prompt with the runs already written, so the server aborted ~10 s into the first long prompt whenever the disk tier and image input were both on (0.1.13, 0.1.14). It reads the text tokens, which a prompt without media has exactly as many of. The check `apply_slot_state_guard` added skipped every slot while a projector was loaded; it now skips only prompts that hold media, whose positions run ahead of their cells |

## Addendum 2026-09-24: Q8_0 K/V

`ggml/src/ggml-cuda/cpy.cu` joins the delta (43 files: 39 modified, 4 added); replay 43 / 43, 46 patches. Measured in
`docs/results/kv-q8-20260924.json`.

| patch | files | what |
|---|---|---|
| `apply_kv_q8_0` | `cpy.cu`, `ggml-cuda.cu`, `qsa.cu`, `llama-memory-hybrid-idx.cpp`, `llama-kv-cache.cpp`, `qwen4exp.cpp`, `server-task.cpp`, `.h`, `server-context.cpp` | a Q8_0 K/V cache for qwen4exp. The sparse prefill kernel (qsa3) reads K and V only through packed f16 layouts, so the graph dequantizes a Q8_0 cache into them (a Q8_0 -> f16 copy kernel) and the kernel accepts the type; the decode gather and the block-key rebuild read Q8_0 rows through the same get_rows. The indexer's keys stay f16. V is not rotated for this architecture: qsa3's matrix-core sums move in the last bit with the V of keys they weight by zero (free cells after a conversation's end, holding an earlier conversation's data), and the inverse rotation spread that far enough that a conversation read back from disk parted from the resident one. The disk tier deletes entries whose rows are another size, as it does older formats |
| `apply_state_hash_debug` | `server-context.cpp`, `server-task.cpp` | debugging aids, off unless set: `STRIX_STATE_HASH` logs hashes of a slot's recurrent and whole state as a task starts, after every prompt batch and as a conversation leaves; `STRIX_V3_VERIFY` reads every restored run back and checks it against its name |

## Addendum 2026-09-24: 0.1.17

No new files in the delta (43: 39 modified, 4 added); replay 43 / 43, 49 patches. Measured in
`docs/results/concurrency-20260924.json`.

| patch | files | what |
|---|---|---|
| `apply_spec_draft_by_slots` | `server-context.cpp`, `speculative.cpp` | `STRIX_SPEC_DRAFT_BY_SLOTS` caps a step's draft by how many slots generate ("3,2,2,0": 3 for one, 2 for two or three, none from four): in this MoE a verified token reads ~10 more experts' weights, which conversations decoding together cannot share, so several at once verify fewer tokens (four: 37.9 -> 47.4 tok/s summed). The MTP draft stops at the caller's cap instead of drafting to its own and being truncated |
| `apply_kv_zero_freed` | `llama-kv-cache.cpp`, `.h` | a freed cell's rows are zeroed (seq_rm, seq_keep, a move's vacated cells, clear), by async copies from a zero tensor kept in each cache buffer, on the graphs' stream. The matrix-core sums of the sparse attention moved in the last bit with the values of keys weighted by zero - the free cells after a conversation's end - so results depended on who used them before. V stays unrotated for qwen4exp: the disk tier's q8_0 conversations are stored so, in rows of the same size |
| `apply_mtp_carry_state` | `speculative.cpp`, `server-context.cpp`, `server-task.cpp`, `.h` | the MTP drafter's carried target row keeps its position (zeros when it does not belong) and goes with checkpoints (`data_spec`), the RAM tier's entries, the disk tier's restore point and child slots; a restored slot file clears it. After a rewind the first draft-cache row was computed from a stale row, so the same prompt drafted differently twice and greedy output parted at a near-tie |

## Addendum 2026-09-25: 0.2.0

Eight files join the delta (51: 47 modified, 4 added): `gated_delta_net.cu`, `.cuh`, `hc-cn.cu`, `.cuh`,
`idx-relu-sum.cu`, `.cuh`, `qsa.cuh`, `top-k.cu`; replay 51 / 51, 52 patches. Measured in
`docs/results/prefill-kernels-20260925.json`.

| patch | files | what |
|---|---|---|
| `apply_gdn_direct_rows` | `llama-memory-recurrent.cpp`, `.h`, `llama-graph.cpp`, `.h` | when every sequence of a batch reads its recurrent state from a row of its own cell (its newest state, or a rollback snapshot after rejected drafts), `build_rs` marks the state gather ('SGIP' in `op_params[0]`) and the HIP backend reads the rows `s_copy` names straight from the cache instead of gathering them into scratch (3 MB a sequence a layer). The condition is `s_copy_own_cell`, which unlike `s_copy` consumes no rollback index, and every hybrid input's `can_reuse` compares it |
| `apply_ple_unbuffered_cache` | `llama-lazy-reader.h`, `llama-model.cpp` | the per-layer-embedding table read with `FILE_FLAG_NO_BUFFERING`, 16 requests in flight on each of 16 workers, and no buffered handle or mapping of the file left open (either drops unbuffered reads to ~8K IOPS): a cold 2K-token gather 240 -> 58 ms. The rows read are kept raw in a direct-mapped RAM cache (`STRIX_PLE_CACHE_MB`, default 400) |
| `apply_prefill_kernels_020` | `ggml-cuda.cu`, `mmb.cu`, `.cuh`, `gated_delta_net.cu`, `.cuh`, `hc-cn.cu`, `.cuh`, `idx-relu-sum.cu`, `.cuh`, `qsa.cu`, `.cuh`, `top-k.cu`, `qwen4exp.cpp` | the hyper-connection inject computed inside the combine-norm kernel; narrow F32 weights on their own kernels; F32 GEMMs from 9 columns on WMMA instead of hipBLAS (whose kernels load from disk on first use: 20-400 ms stalls); BF16 activation rows padded off the memory channels' 4 KB stride; the indexer's strip score in one kernel; TOP_K as a register radix select (same selected set); the QSA key/value packs in one pass from the cache; the attention gate's copy and sigmoid in one kernel; a 16-row gated delta net prefill kernel. The copy fusions read their source while they write, so `graph_optimize` keeps the source allocated until the output is computed. Bitwise to 0.1.17 with `STRIX_HC_INJECT_FUSE=0 STRIX_GDN_R16=0 STRIX_SKINNY_F32=0 STRIX_MMB_F32_MIN_T=512` |

## Addendum 2026-09-25: 0.2.1

No new files in the delta (51: 47 modified, 4 added); replay 51 / 51, 54 patches. Measured in
`docs/results/multi-stream-20260925.json`.

| patch | files | what |
|---|---|---|
| `apply_spec_even_drafts` | `speculative.cpp`, `server-context.cpp` | the drafts of one speculative step are the same length for every conversation (`STRIX_SPEC_EVEN_DRAFTS`): the MTP drafter keeps a sequence that turns unconfident drafting while another is still confident and cuts all to the longest confident run; the server cuts every draft to the shortest when a generating slot drafted nothing, unless one replays accepted tokens. The hybrid memory ran a verify of uneven drafts as ubatches of equal tokens per sequence, two or three passes of the whole model in shapes the graph cache had not seen: three conversations at 4K tokens 34.3 -> 45.7 tok/s summed |
| `apply_small_k_membership` | `ggml-cuda.cu` | the compact scorer's block-membership product (F16, K = the ubatch's sequences) on a kernel of one thread an output instead of hipBLAS, which loaded a kernel from disk on first use of each shape (272 ms). Exact: the membership is 0 or 1. `STRIX_SMALL_K=0` uses hipBLAS |

## Addendum 2026-09-25: 0.2.2

`mmq.cu` joins the delta (52: 48 modified, 4 added); replay 52 / 52, 56 patches. Both patches take a
product off hipBLAS, which loads each GEMM kernel from disk the first time it meets a shape.

| patch | files | what |
|---|---|---|
| `apply_bf16_mid_batches` | `mmb.cu` | BF16 weights (the sparse-attention indexer's projections) on the MMB kernel from 9 columns (`STRIX_MMB_BF16_MIN_T`), as F32 weights since 0.2.0: `mul_mat_f` takes up to 16 columns and MMB took 512 on, so a batch of 17-511 tokens - the new part of most chat turns - went to hipBLAS. A 38-token batch 572 -> 200 ms the first time |
| `apply_q6k_mmq_rdna35` | `mmq.cu` | Q6_K on MMQ at every width on RDNA3.5 (`STRIX_MMQ_Q6K_ANY=0`: upstream's 256): the draft head's `attn_v` stalled 208 ms in the draft's first prompt batch, and a prompt's last batch is a new width almost every time. The target's Q6_K output projection gets at most 16 rows from the server, which MMQ already took; the perplexity tool's 8192 now take MMQ too (activations quantized): perplexity at 8K 2.6814 against 2.6811 |

## Addendum 2026-09-26: 0.2.3

No new files in the delta (52: 48 modified, 4 added); replay 52 / 52, 57 patches. The build pin moves
to TheRock ROCm 10.2.0a20260925. Measured in `docs/results/multi-stream-20260926.json`.

| patch | files | what |
|---|---|---|
| `apply_small_batch_decode` | `ggml-cuda.cu` | two products of a several-conversation verify step (5-32 tokens) on the vector kernel over chunks of the batch, not on tiles they barely fill: an F32 weight at 9-32 columns (`STRIX_F32_VEC_CHUNK_MAX`; the MoE router [2560 x 512] took MMB's 128-row tiles, four workgroups for the whole GPU, ~250 us a product), and the routed experts past the vector kernel's limit (IQ3_S 4 tokens, IQ4_NL 6) in chunks of 4 tokens up to 16 (`STRIX_MOE_VEC_CHUNK=0`: MMQ as before; `STRIX_MOE_VEC_CHUNK_MAX_T`). A nine-token verify step 120-123 -> 103-104 ms. Batches of one to four tokens, and of more than 32, are unchanged |

## Addendum 2026-09-26: 0.2.4

`concat.cu` joins the delta (53: 49 modified, 4 added); replay 53 / 53, 58 patches. Measured in
`docs/results/verify-small-products-20260926.json`.

| patch | files | what |
|---|---|---|
| `apply_verify_small_products` | `ggml-cuda.cu`, `concat.cu` | three more products of a several-conversation verify step off kernels that ran a few workgroups for the whole GPU: the GDN conv input's concat of a transposed batch of 2-31 tokens on the tiled transpose (`STRIX_CONCAT_T_MIN`, upstream 32: the non-contiguous kernel ran one 256-thread block per channel and sequence for 3 + tokens values, ~30K blocks at three conversations); at 9-32 columns a BF16 weight (`STRIX_BF16_VEC_CHUNK_MAX`; the indexer's k projection [2560 x 128] took MMB's one 128-row tile) and a quantized weight of at most 1024 output rows (`STRIX_Q_VEC_CHUNK_MAX`, `STRIX_Q_VEC_CHUNK_ROWS`; the hyper-connection down projection, the attention k and v, the shared expert took 3-5 MMQ tiles) on the vector kernel over chunks of 8 columns. A nine-token verify step 106.4 -> 103.0 ms, a twelve-token one 129.4 -> 123.5 (the server's timing, default sampling). Batches of one to eight tokens give the same output as before (the concat is a copy either way), and so do batches of more than 32 |

## Addendum 2026-09-27: 0.2.5

No new files in the delta (53: 49 modified, 4 added); replay 53 / 53, 59 patches. Measured in
`docs/results/issue2-image-history-20260927.json`.

| patch | files | what |
|---|---|---|
| `apply_image_dense_bound` | `llama-memory-hybrid-idx.cpp`, `ggml-alloc.c`, `ggml-cuda.cu` | once a conversation holds an image (or a position gap), its ubatches take the dense sparse-attention inputs, a KQ mask and a per-block bias over n_kv x n_tokens staged in pinned host memory; they grew with every image, and ROCm on Windows kept each pinned buffer the growth freed - 27 images of 1920x1080 on a 58K conversation took 12 GiB of RAM, and a 14K text append at batch 8192 then faulted (issue #2). Such ubatches are held to n_kv x n_tokens <= 2^27 (`STRIX_DENSE_UBATCH_BUDGET`, 0: off), a compute buffer that has to grow again grows by at least a quarter, and a kernel fault names the failing node and where its tensors lie in their buffers. The reported workload completes with 3.7 GB of shared GPU memory instead of 13.7; text before any image is bitwise unchanged |

## Addendum 2026-09-27: 0.2.6

Seven files join the delta (60: 56 modified, 4 added): `common/chat.h`, `ggml.h`, `ggml.c`, `ggml-cpu.c`,
`ops.cpp`, `server-queue.cpp`, `server-queue.h`; replay 60 / 60, 62 patches. Measured in
`docs/results/prefix-cache-gdn-20260927.json`.

| patch | files | what |
|---|---|---|
| `apply_prefix_cache_026` | `server-context.cpp`, `server-queue.cpp`, `server-queue.h`, `server-task.h`, `chat.h`, `llama-kv-cache.cpp`, `llama-kv-cache.h`, `llama-context.cpp`, `llama.h` | checkpoints at the anchors a later prompt forks from (the first user message, kept for good; the last one; the end of each prompt; turn starts spaced by `--checkpoint-min-step` and a quarter of the distance to the end), and a full list drops the one whose loss costs least; a prompt whose best-sharing slot holds another conversation takes a free slot and gets the shared prefix copied in (attention rows device to device, `llama_strix_kv_copy_rows`; the recurrent state from that slot's checkpoint); a prompt whose prefix a busy slot is computing waits for its anchor and copies it (`STRIX_PREFIX_WAIT=0`: off). Agent workload, 4 slots: 73.3K tokens processed -> 29.3K (ideal 29.8K), prompt time 172 -> 60 s |
| `apply_gdn_deferred_rollback` | `ggml.h`, `ggml.c`, `ops.cpp`, `ggml-cpu.c`, `gated_delta_net.cu`, `ggml-cuda.cu`, `llama-memory-recurrent.cpp`, `llama-memory-recurrent.h`, `llama-graph.cpp`, `llama-graph.h`, `qwen4exp.cpp` | a verify batch no longer writes a state snapshot per token (3 MB a layer a sequence): `ggml_gated_delta_net_lazy` leaves the state where it was, records per token what a replay needs (delta, key, gate), and the next batch replays what its rollback kept before it writes the state back; a replay op brings states up to date before a longer batch, and a saved state is replayed on the host with the same `fmaf`. Output bitwise 0.2.5's; a verify step at three / four conversations -2.8% / -4.0% (`STRIX_GDN_LAZY=0`: off) |
| `apply_image_rank_per_seq` | `llama-memory-hybrid-idx.cpp`, `llama-memory-hybrid-idx.h` | the sparse attention ranks an image's cells per sequence of the ubatch rather than only while the cache holds one sequence: an image in a second resident conversation asserted in `set_input_qsa` (`group_members`). One resident sequence is unchanged, bitwise. Also `kv_rows_copy` for the prefix cache |

## Addendum 2026-09-27: 0.2.7

Four files join the delta (64: 60 modified, 4 added): `llama-batch.cpp`, `llama-batch.h`, `models.h`,
`delta-net-base.cpp`; replay 64 / 64, 64 patches. Measured in `docs/results/agent-sessions-20260927.json`.

| patch | files | what |
|---|---|---|
| `apply_agent_slots_027` | `server-context.cpp` | a slot takes a prompt for what it holds only when the prompt continues it (keeps all that was last asked there, less the 16 tokens a template may render differently once the answer is in the history); anything else goes to a free slot, else the least recently used one, and `copy_prefix` brings the shared part. Upstream's choice by similarity let a new agent session take another session's slot and cut its history (`STRIX_SLOT_BY_SIMILARITY=1` restores it). While conversations generate, a step takes at most `STRIX_PREFILL_BUDGET` prompt tokens (default 2048, 0: `n_batch`) |
| `apply_ragged_ubatch_027` | `llama-batch.cpp`, `llama-batch.h`, `llama-memory-recurrent.cpp`, `llama-graph.cpp`, `llama-graph.h`, `llama-memory-hybrid-idx.cpp`, `models.h`, `delta-net-base.cpp`, `qwen4exp.cpp` | ragged ubatches: `split_ragged` puts sequences of different token counts in one ubatch (`llama_ubatch::ragged`, per-set counts and offsets); the recurrent memory takes per-set counts, a ragged ubatch replays the deferred rollback first and never reuses a graph, and qwen4exp builds the gated delta net's conv and net and the per-layer-embedding conv once per group of one count, the per-token rest once. Equal-length ubatches build the graph they did (`STRIX_RAGGED=0`: the equal-length split) |

## Addendum 2026-09-27: 0.2.8

No file joins the delta (64: 60 modified, 4 added); replay 64 / 64, 67 patches. Measured in
`docs/results/prompt-alone-20260927.json`.

| patch | files | what |
|---|---|---|
| `apply_qsa3_wide_window_028` | `qsa.cu`, `qwen4exp.cpp` | the sparse prefill kernel (qsa3) numbers the window's blocks of 4 cells in 32 bits and takes a window of any size; with 16 bits it declined one past 262140 cells, and the generic flash-attention kernels that ran instead ignore the selected indices - with no mask (a maskless graph) every query attended densely to every cell of the window, other conversations and later positions included. 0.2.7's ragged ubatches spanned such windows (a prompt chunk next to three long answers in a 512K pool): ~20 ms a token, wrong result. `qwen4exp_qsa_maskless_ok` keeps a graph off the mask only where the kernel runs. Windows under 262141 cells: bitwise as before |
| `apply_prompt_alone_028` | `llama-batch.cpp` | `split_ragged` puts the short sequences of a batch together (decode tokens, MTP verify drafts, a prompt of a few tokens: at most `STRIX_MIX_MAX` = 32 tokens each and `STRIX_MIX_TOTAL` = 32 in all); a prompt chunk joins them only while the query-cells it adds - its queries over the others' contexts, theirs over its own, the causal score bounds a ubatch of several sequences loses - stay under `STRIX_MIX_CELLS` (6e7, ~0.2 s, about a pass), else it runs alone next. 0.2.7 let every chunk share (a 49K-token prompt next to an 88K-token answer: 793 t/s against 1012 alone); letting none share cost six agents' short tails a fifth more prompt time. `STRIX_MIX_MAX=0`: 0.2.7's ragged ubatches |
| `apply_disk_store_v4_028` | `server-task.cpp`, `server-context.cpp` | the disk tier's version 3 layout is numbered 4, so a store deletes what 0.2.7 and earlier wrote at startup: 0.2.7 could store a conversation whose prompt chunk had attended over other conversations' cells |

## Addendum 2026-09-27: 0.2.9

Four files join the delta (68: 64 modified, 4 added): `fattn.cu`, `norm.cu`, `norm-gated.cu`,
`norm-gated.cuh`; replay 68 / 68, 70 patches.

| patch | files | what |
|---|---|---|
| `apply_qsa_between_029` | `qwen4exp.cpp`, `fattn.cu` | a ubatch of 33-127 queries a sequence sits between the decode gather (at most `LLAMA_QSA_DECODE_GATHER_MAX_T` = 32) and the qsa3 prefill kernel (at least 128): its block selection went to the generic flash attention with the plain causal mask, which ignores the selection, so tool results and short follow-ups at depth attended densely to every visible cell. Such ubatches take the top-k and the top-k mask. `fattn.cu` aborts when a maskless op with selected indices would reach a generic kernel: 0.2.7's wide-window bug, and this one, would have failed loudly |
| `apply_gdn_r16_lds_029` | `gated_delta_net.cu` | the r16 recurrence loads each tile from global memory straight into LDS; staged through per-thread register arrays it kept them in scratch (10 `scratch_store_b128` and 10 `scratch_load_b128` a tile). GDN in a 64K-token prefill 3006 -> 2582 ms, bitwise |
| `apply_norm_rows_plain_029` | `norm-gated.cu`, `norm-gated.cuh`, `norm.cu` | `rms_rows_f32` (a wave a row, 8 rows a block) takes a variant without weights, and `rms_norm_f32_cuda` hands it rows of at most 256 floats when there are at least 4096 of them: the per-head q/k norms of a 64K-token prefill 449 -> 352 ms, bitwise |

## Addendum 2026-09-28: 0.3.0

No file joins the delta (68: 64 modified, 4 added); 71 patches.

| patch | files | what |
|---|---|---|
| `apply_kq_mask_seq_bits_030` | `llama-kv-cache.cpp` | each sequence's first KQ-mask row tested every cell's membership bitset (256 bits, 32 bytes a cell), so a ubatch of several conversations read the window's bitsets once per conversation; the bits of a ubatch's sequences (2-16 of them, one cell array, windows of at least 4096 cells) are gathered in one pass, two bytes a cell, and the rows test those - the same test on the same cells. Four conversations over ~112K cells: a step's inputs 2.33 -> 1.75 ms. `STRIX_KQ_MASK_CHECK=1` fills every mask the upstream way as well and compares the two byte for byte |

## Addendum 2026-09-28: 0.3.1

Three files join the delta (71: 67 modified, 4 added): `llama-context.h`, `llama-cparams.h`, `llama-ext.h`;
replay 71 / 71, 73 patches.

| patch | files | what |
|---|---|---|
| `apply_kb_pending_031` | `llama-memory-hybrid-idx.h`, `llama-memory-hybrid-idx.cpp`, `qwen4exp.cpp` | the QSA block-key cache refreshed a block's key only in a graph that runs the indexer, and a view of at most `indexer_top_k + ratio - 1` cells takes the dense shortcut (`build_qsa_store_k`) instead - so a new conversation in a used slot whose first prompt was under ~2K tokens kept the previous conversation's block keys for those cells, and once it grew past 2K the sparse selection scored its first blocks (system prompt, first message) with the wrong keys. A graph that writes indexer keys without the indexer now leaves its sequences pending from its first position (`kb_from`); the next indexer graph puts those blocks in its dirty list (up to `kb_pending_max` = 64 positions besides its own tokens) or rebuilds the sequence in full. Fresh-vs-used-slot probe (`tmp/qsa/kb_stale_probe.py`): the used slot's second turn differed (first-token top-5 logprob delta 2.9), with the fix both are bitwise identical and equal to the old fresh run |
| `apply_mtp_index_share_031` | `llama-memory-hybrid-idx.h`, `llama-memory-hybrid-idx.cpp`, `qwen4exp.cpp`, `models.h`, `llama-context.h`, `llama-context.cpp`, `llama-cparams.h`, `llama-ext.h`, `llama-graph.h`, `speculative.cpp` | IndexShare for the MTP draft (`LLAMA_MTP_INDEX_SHARE=1`): a draft step attended densely to the draft's whole cache; now it gathers the sparse selection a catch-up kept for its sequence plus the cells of every position since. A catch-up runs the layer's indexer only when the latest kept selection is more than `LLAMA_MTP_INDEX_SHARE_REFRESH` (32) positions behind, on views of `LLAMA_MTP_INDEX_SHARE_MIN_KV` (32768) cells or more; a draft loop reuses one at most 48 positions old; every reused cell is clamped to the graph's view. A draft ubatch without outputs (a prompt, the catch-up after a verify) stores K, V and the indexer keys and builds no query, attention or wo (`LLAMA_MTP_STORE_ONLY=0` restores them): bitwise-identical drafts. A speculative pass: 86K 79.75 -> 76.90 ms, 212K 93.70 -> 87.60 ms at the same acceptance |

## Addendum 2026-09-28: 0.3.2

Six files join the delta (77: 73 modified, 4 added): `common.h`, `common.cpp`, `sampling.h`, `sampling.cpp`,
`server-common.h`, `server-common.cpp`; replay 77 / 77, 74 patches.

| patch | files | what |
|---|---|---|
| `apply_ckpt_edges_032` | `common.h`, `common.cpp`, `sampling.h`, `sampling.cpp`, `server-common.h`, `server-common.cpp`, `server-context.cpp` | checkpoints of the recurrent state at an answer's edges instead of prompt batches cut short near their end: where the answer starts (the prompt's end, taken as its first token is sampled, with that token's logits) and where the answer before it ended (the slot's state as a continuing request arrives), neither costing a pass. The cuts 4 and 4 + n_ubatch tokens before a prompt's end go, and the one at the last user message when the answer-end checkpoint covers it. A regenerate is sampled again from the stored logits and processes nothing; a thinking prompt (ending in `<think>\n`, which a reasoning-dropping client re-renders as `<think>\n\n</think>`, one token earlier) runs its last token alone. Prompts are batched in the order their requests arrived. The disk tier writes a conversation's main line only: its prompts, and an answer once the next request continues it; an unconfirmed answer goes as the conversation leaves unless its client is known to drop answers - known from a hash of the answer's first tokens, kept in memory across the conversation's trips to the store. `common_sampler_sample_logits` and `get_token_probabilities` work from a stored row of logits. `STRIX_CKPT_EDGES=0`, `STRIX_PROMPT_ORDER=slot`, `STRIX_DISK_MAINLINE=0` restore 0.3.1 |

## Addendum 2026-09-28: 0.3.3

No file joins the delta (77: 73 modified, 4 added); replay 77 / 77, 75 patches.

| patch | files | what |
|---|---|---|
| `apply_pool_guard_033` | `server-context.cpp`, `server-task.h`, `server-queue.cpp` | the pool guard. Upstream fails every request of a step that does not fit in the unified KV pool with "Context size has been exceeded" (after halving the batch down to one token) and clears their caches; a user's eight agents, ~870K tokens in a 680K pool, lost seven requests at once. A prompt now starts only when it fits beside what the busy conversations will hold - their whole prompts and a reserve for each answer (8192 tokens, less when the request allows less; `STRIX_POOL_ANSWER_ROOM` for tests) - once idle conversations are on disk or purged (`make_room` counting busy slots at `pool_future`); until then it waits in the queue holding no slot (`wait_pool`), handed back by `update_slots` once it fits, and one that has waited a minute holds back those behind it. A prompt's next piece and an answer's next tokens go into a step only as far as there are cells, idle slots purged first; otherwise they wait. When every busy conversation waits three steps in a row, the one that arrived last gives way: a prompt holding cells with a 503 (`ERROR_TYPE_UNAVAILABLE`), else an answer ending with `stop_type` limit. `/slots` answers with `X-Strix-Waiting` (the deferred requests); deferred tasks that become ready return to the queue in the order they were deferred, and a released slot no longer pops one waiting for room. Eight ~12K-token conversations at once in a 64K pool: 8 of 8 failed with `STRIX_POOL_GUARD=0`, 0 of 8 with the guard; a pool with room: bitwise 0.3.2 |

## Addendum 2026-09-28: 0.3.4

No file joins the delta (77: 73 modified, 4 added); replay 77 / 77, 78 patches.

| patch | files | what |
|---|---|---|
| `apply_idx_score_dec_034` | `ggml-cuda.cu`, `idx-relu-sum.cu`, `idx-relu-sum.cuh` | a decode step's lightning-indexer block scores in one kernel. For each of the 12 sparse-attention layers the scores ran as GET_ROWS (every block key of the window, F16 out of the block-key cache into an F32 copy), MUL_MAT (the keys against the step's queries, four heads; with several conversations in a step the vector kernel re-read every key per column chunk), RELU, the head sum and the visibility (with several conversations, a 0/1 membership product): about 20 passes over the scores. `k_idx_score_dec` reads each key once from the cache and computes the same scores bit for bit: each dot product is the one `mul_mat_vec_f` takes for K = 128 (64 threads a row, `ggml_cuda_mad` from zero, a butterfly sum in each wave of 32, the two waves reduced again with zeros), spelled out by two threads a key. `ggml_cuda_try_fuse` matches the chain from its GET_ROWS (one form for a step of one conversation, one with the membership inputs), skips it and runs the kernel at the MUL_MAT; `graph_optimize` keeps the inputs alive to the chain's end. One conversation at 110K: 45.4 -> 43.1 ms a token; eight at 40K each, 512K q8_0 pool, MTP off: the step 131 -> 107.5 ms. `STRIX_IDX_SCORE_DEC=0` keeps the nodes; `=2` runs both and aborts on any difference (with `GGML_CUDA_DISABLE_GRAPHS=1`). Also: a graph the scheduler did not split again (same cgraph, same uid) keeps its HIP graph key and compatibility check, ~0.28 ms of host time a step |
| `apply_qsa_runs_034` | `llama-kv-cells.h`, `llama-memory-hybrid-idx.h`, `llama-memory-hybrid-idx.cpp`, `llama-graph.cpp` | the sparse-attention inputs without the per-cell scan. `set_input_qsa` tested every cell of the graph's window against a bitset per ubatch: 1.6 ms a step for one conversation at 110K cells, 10-14 ms for eight at 40K each (the window spans them all), the GPU idle meanwhile. `llama_kv_cells::seq_run` knows which sequences are one run - consecutive cells holding consecutive positions, one a position (text only: an image repeats a position) - kept by an append at the next cell and position and by removing the last cell, checked again in full after any other change. For a ubatch whose sequences are each one run sharing no cell, `set_input_qsa_run` derives every input from the runs' first cells and positions: a sequence's blocks are its whole position buckets, listed bucket by bucket with the sequences in reverse order of their first cell, as the scan's list-head insertion leaves them; the block-key refresh lists and the membership inputs follow. `qsa_scalar_visibility`'s window scan for image cells is skipped the same way. Eight at 40K: 10-14 -> 1.8 ms a step. `STRIX_QSA_RUN=0` scans; `=2` takes both ways and aborts on any difference. `STRIX_INPUT_TIMING=1` logs each graph input's set and reuse time |
| `apply_ckpt_budget_034` | `server-context.cpp` | the context checkpoints of every slot share one budget in RAM. Each slot kept up to `n_ctx_checkpoints` (8) recurrent-state checkpoints of ~113 MiB, so eight conversations grew by ~2.2 GB a turn (1.8 -> 8.3 GiB working set in three turns; a user saw ~10 GB). Over `STRIX_CKPT_BUDGET` (24 by default, ~2.7 GB) the slot holding the most in RAM (of equals the least recently used) gives one up - its least valuable by the per-slot eviction's own measure, never its first (the end of the system prompt), this task's last - so one conversation still keeps `n_ctx_checkpoints` and eight keep three each. Checkpoints the disk tier holds for an idle slot do not count. `STRIX_CKPT_BUDGET=0`: the per-slot limit alone |

## Addendum 2026-09-29: 0.3.5

No file joins the delta (77: 73 modified, 4 added); replay 77 / 77, 83 patches.

| patch | files | what |
|---|---|---|
| `apply_idx_score_skip_035` | `idx-relu-sum.cu` | the decode indexer score (0.3.4's `k_idx_score_dec`) tests a query's visibility before its dot products: with several conversations in a step a block is seen by its own conversation's queries only, and the kernel computed all eight queries' four head scores for every block before writing -inf for the others. The membership product (0/1 terms) becomes a test of two bit masks for up to 32 slots; a thread pair skips together, so its shuffle stays inside it. The scores computed are the same bits. Eight conversations at 40K: 105.8 -> 103.3 ms a step |
| `apply_qsa_run_buffers_035` | `llama-memory-hybrid-idx.h`, `.cpp` | the run-based sparse-attention inputs reuse their vectors from call to call (eight conversations at 40K: ~4 MB a step) and set each block's membership row at its run's slot, found once; byte for byte the same inputs (`STRIX_QSA_RUN=2`); host time of the step's QSA inputs 1.7-1.8 -> 0.8-1.05 ms |
| `apply_parallel_sampling_035` | `server-context.cpp` | the generating slots of a step that need a plain sample are sampled at once, one thread each, with `common_sampler_sample_logits` from the logits row read after the context synchronized; accepting, text, stop checks and responses stay in slot order. Eight conversations at 40K with temperature 0.8 / top-k 40 / top-p 0.95 / min-p 0.05 / repeat penalty: 109.3 -> 103.0 ms a step. `STRIX_PARALLEL_SAMPLING=0` samples one by one, `=2` also samples each slot the old way from a copy of its sampler and aborts on a difference. Debugging: `STRIX_STATE_DUMP=<dir>` (with `STRIX_STATE_HASH`) writes whole states as a conversation leaves a slot and as a task starts |
| `apply_skip_ops_shape_035` | `ggml-cuda.cu` | measurement only: `STRIX_SKIP_NOFILL=1` drops a skipped node without the zero fill (itself a dispatch), and `OP@ne0` selects an op's nodes by output width |
| `apply_restore_wait_035` | `llama-kv-cache.h`, `.cpp` | a restore waits for the queued zeroing of freed cells before it writes rows. `zero_cells` (0.1.17) queues the zeroing on the graphs' stream; `state_read_data` and the disk tier's `seq_rows_set` write with `ggml_backend_tensor_set` on another stream, so for a conversation restored into cells another had just left, the last-queued zeroing (the last attention layer's V, ~590 cells) could land after the restore and blank those rows. `wait_queued_copies` synchronizes the scheduler first; a conversation's whole state as it left and as it came back now hash the same |

## Addendum 2026-09-30: 0.3.6

No file joins the delta (77: 73 modified, 4 added); replay 77 / 77, 88 patches.

| patch | files | what |
|---|---|---|
| `apply_mmvq_i8_036` | `mmvq.cu` | the q8_0 and Q6_K vector products of 5-8 columns (a step of 5-8 conversations, or a verify pass) with the activations laid out by the lane that reads them: `quantize_q8_1_i8` / `quantize_q8_1_q6_K_i8` write each lane's share of every column contiguously, and `mul_mat_vec_q8_0_i8` / `mul_mat_vec_q6_K_i8` (one wave a row) keep the stock kernel's lane share, float order and reduction, so the products are the same bits. At 8 columns the stock kernel was bound by issuing its activation loads (24 of 29 loads an iteration); the q8_0 classes take 1.0-1.4 ms less a step and the Q6_K lm head 3.75 -> 2.9-3.1 ms. RDNA3.5 only; `STRIX_MMVQ_I8=0` turns it off, `=N` sets the fewest columns (5), `STRIX_MMVQ_I8_CHECK=1` runs both and aborts on a difference |
| `apply_idx_score_v3_036` | `idx-relu-sum.cu` | `k_idx_score_dec3`: each thread pair first marks which of the step's queries see its block, then computes only those, a lane its own query at a time. With several conversations the block list is interleaved, so 0.3.5's kernel ran every query's dot products for the whole wave (8 times at 8 conversations). The queries sit in LDS head by head with the rows spread over the banks; the arithmetic is `k_idx_score_dec`'s, so the scores are the same bits (`STRIX_IDXD_V=1` takes the old kernel; `STRIX_IDXD_BENCH=n` times both in the server and compares them) |
| `apply_kb_rows_036` | `llama-memory-hybrid-idx.h`, `.cpp` | the block-key cache laid out by residue class: cell c's row is `(c % r)*S + c/r`, r the block ratio (4) and `S = ceil((kv_size + 1)/r)`, where it was c. A block's key is in the row of its first cell and a conversation's blocks start every r cells, so their keys are adjacent rows instead of 1 KB apart - a stride on which this machine's DRAM channels alias: eight conversations' keys read at ~50 GB/s. Any bijection is right whatever the block layout; the inputs (`bid_rows`, the dirty lists, the scratch row) and the region moves (one row run per residue class) use it. Decode score at 8 x 20K: 205 -> 48 us a strip with the new kernel; one conversation at 110K: 33 -> 17 us |
| `apply_top_k_rows_036` | `top-k.cu` | the one-pass TOP_K row kernel (0.2.0) from one row up instead of 32: the sparse-attention decode has a row a query, and below 32 rows it took the parallel radix path, a dozen launches - ~340 us a strip at eight conversations of ~20K (a step 103 -> 97.6 ms); one conversation, at 110K or 210K, is unchanged. The same set; `STRIX_TOP_K_ROWS_MIN=32` restores the old bound |
| `apply_repeat_probe_036` | `ggml-cuda.cu` | measurement only: `STRIX_REPEAT=topk:N,fa:N,kvrows:N,argsort:N,idxdec:N` dispatches those ops N more times (each writes what it wrote before), so a step grows by N times their in-graph cost; `STRIX_SKIP_OPS` takes `OP@lo-hi` and `*@...` |

## Addendum 2026-10-01: 0.3.9

Three files join the delta (80: 76 modified, 4 added): `fattn-common.cuh`, `fattn-tile.cuh` and
`template-instances/fattn-tile-instance-dkq256-dv256.cu`; replay 80 / 80, 90 patches.

| patch | files | what |
|---|---|---|
| `apply_fa_gather_037` | `ggml.h`, `ggml.c`, `fattn-common.cuh`, `fattn-tile.cuh`, `fattn.cu`, `fattn-tile-instance-dkq256-dv256.cu`, `qwen4exp.cpp` | the sparse-attention decode reads its selected cells in place. `qwen4exp_gather_attn` copied each query's 2304 cells (2051 selected, the rest masked) out of the KV cache with `ggml_get_rows` into an f16 slab and ran the tile flash attention on the slabs; at eight conversations the copies cost as much as the attention (~3.0 + ~3.2 ms a step at 8 x 20K, `STRIX_REPEAT`). `ggml_flash_attn_ext_gather(q, k, v, mask, rows, scale)` is a `GGML_OP_FLASH_ATTN_EXT` node (op_params[5] `GGML_FLASH_ATTN_EXT_GATHER`, the cell list in src[8]) that the CUDA backend runs as the same tile kernel instance (`<256, 256, 1, 4>` for 12 query heads a KV head) with its K/V loader reading `rows[s*n_sel + i]` of the cache: f16 copied, q8_0 dequantized as `ggml_get_rows` does - `ggml_cuda_q8x4_to_h2`, the 0x6400 bias and one f16 multiply, equal to `float(q)*float(d)` cast to half for all 2^16 scales x 256 values (checked exhaustively) - and with the f16 kernel's occupancy, so the same parallel blocks, split and combination: the output is the gathered path's bit for bit. Standalone (12 layers): 8 x 20K q8_0 6.7 -> 2.2 ms, f16 6.8 -> 2.4, one query at 40K 0.94 -> 0.40. In the server, alternated with 0.3.6: eight conversations of ~40K in a 512K q8_0 pool, MTP off, 99.9 -> 93.6 ms a step (80.3 -> 85.6 tok/s); eight of ~20K, f16, 95.9 -> 88.4 ms; one conversation 38.7 -> 38.0 ms a token at 50K, unchanged at 3K; MTP on, three / four streams +2% / +3%. `STRIX_FA_GATHER=0` keeps the copy; `STRIX_FA_GATHER_CHECK=1` (with `GGML_CUDA_DISABLE_GRAPHS=1`) also runs the copy and the old attention for every call and aborts on a difference |
| `apply_mmvq_i8_unroll_037` | `mmvq.cu` | `mul_mat_vec_q8_0_i8` (the 5-8 column q8_0 GEMV, 0.3.6) unrolls its K loop by two, so two iterations' loads are in flight: eight columns of [10240 -> 320] 26.3 -> 22.1 us, 97 of them a step. Each lane's operations keep their order: the same bits (gemv_bench hashes, `STRIX_MMVQ_I8_CHECK=1`) |

## Addendum 2026-10-01: 0.4.0

Two files join the delta (82: 78 modified, 4 added): `mmvf.cu` and `topk-moe.cu`; replay 82 / 82, 96 patches.

| patch | files | what |
|---|---|---|
| `apply_bf16_twins_040` | `ggml.h`, `ggml-cuda.cu`, `llama-model.h`, `llama-model.cpp`, `llama-graph.cpp` | BF16 twins of the F32 weights whose every value is a bfloat16: 264 of them (the MoE routers, the shared experts' gates, the delta-net alpha / beta, the hyper-connection injects), 0.14 GiB. A batch of at most 8 tokens multiplies them with `mul_mat_vec_f`, whose bf16 loop is its f32 loop, and bf16 widens to f32 exactly: the same products from half the bytes (the router [2560 -> 512] 28 -> 17 us at one token, 36 -> 28 at eight). `GGML_BF16_TWIN_MAGIC` in op_params[0] keeps a twin on the vector kernel up to 8 columns, as its original (bf16 alone takes the WMMA kernel from 4 columns on RDNA3.5, with other sums). `STRIX_BF16_TWINS=0`: none |
| `apply_gpu_timeline_040` | `ggml-cuda.cu`, `server-context.cpp` | measurement only: `STRIX_GPU_TIMELINE=N` captures a `wall_clock64` timestamp kernel after every dispatch into the HIP graph and prints the GPU time per op class and graph shape (`STRIX_GPU_TIMELINE_CHAINS`, `STRIX_GPU_TIMELINE_ORDER`) - repeats read their weights from the 32 MB cache and the per-node event timer inflates small kernels, so this is the profile to trust; `STRIX_REPEAT_FILE=<path>` repeats op classes, re-read while the server runs; `STRIX_SPEC_TIMING=1` also times MTP-off steps |
| `apply_decode_small_kernels_040` | `strixllama-chain.cu`, `mmvf.cu`, `topk-moe.cu` | the latency-bound small kernels of a decode step. The fused elementwise chains read their operands by kind (direct, linear, a scalar) and keep their step loop rolled: the unrolled general indexing was code fetched at every launch (sigmoid-mul-add 8.4 -> 2.7 us, add-softplus-mul 8.2 -> 2.5, scale-silu 3.5 -> 1.4); `mul_mat_vec_f` unrolls its loop by four below 128 rows (the hyper-connection inject 6.0 -> 4.0 us); `topk_moe` takes its arg-max with DPP and a non-inlined `expf` (10.3 -> 7.8 us). The same arithmetic, the same bits. One conversation at 3K / 50K / 110K: -2.4 / -2.7 / -1.9 ms a token with the BF16 twins |
| `apply_ab_switch_041` | `ggml-cuda.cu`, `common.cuh` | measurement only: `STRIX_AB=n`, an in-run A/B. Every n decode graphs the side flips between A and B and every HIP graph is captured anew; a switch that reads `ggml_cuda_strix_ab()` (or `ggml_backend_cuda_strix_ab` through the backend registry) follows it, and the GPU timeline keeps the two sides' tables apart, leaving out the two graphs after a flip. Sequential runs on this machine drift 3-5% over minutes; inside one run the drift falls on both sides (a null A/B: +0.3%) |
| `apply_gdn_records_041` | `gated_delta_net.cu`, `llama-memory-recurrent.h`, `.cpp`, `ggml.h` | the delta net at decode. Two columns a warp (`STRIX_GDN_COLS=1`: one): a warp had one 512-byte column of state in flight, and eight conversations' 24 MB a layer were read at ~117 GB/s, at ~205 with two (207 -> 123 us standalone, `tmp/dec040/wb_probe2.hip`; in the server, a step at eight conversations 87.6 -> 84.4 ms). And the records accumulate (`STRIX_GDN_ACC=W`, default 8; 0 or 1 as before): a short batch - a decode step, now also without MTP, or a verify - appends its records (delta, key and gate, 33 KB a token) after the pending ones and leaves the state row as it is while the set holds W, and only then writes the replayed state back. Writing 3 MB a conversation a layer every token cost 5.85 ms of an 83.5 ms step at eight conversations (the write skipped, in-run); with W = 8 a step is 4.6 ms shorter. The replay loads its records in blocks of six, a wait a block (eight took 98 registers and lost occupancy). Records are addressed by index (`info` [3] and [4]) and -1 as the row written leaves the row alone; a rollback reaches back at most through its last batch (`rec_last`); a memory without recurrent layers (the MTP draft's) records nothing. Each column is summed by the same lanes in the same order and the replay is the plain net's fmaf chain: the same bits (78 op-level cases, one column against two, append included, `tmp/dec040/gdn_cols_test.cpp`; the probes; eight conversations in one request). The host's replay for state saves and `seq_cp` takes an AVX2 FMA path, checked against `fmaf` at first use: 262 -> 14 ms for eight records of a conversation |
| `apply_ple_wake_041` | `llama-lazy-reader.h`, `qwen4exp.cpp`, `llama-context.cpp` | the PLE rows are read unbuffered, and the drive and its link drop to a low-power state within ~1 ms of idle: the first read after it takes ~640 us instead of ~150 (`tmp/dec041/ssd_latency.cpp`; polling the completion instead of waiting changes nothing). A decode step's gather nearly always misses a row of the RAM cache - a new n-gram - so it paid the wake-up with the GPU idle. `llama_lazy_reader::wake()` reads one page ahead of it, fire and forget, when a small batch's work is synchronized and when the next small batch starts: the gather at eight conversations 0.80 -> 0.45 ms. `STRIX_PLE_WAKE=0`: off |

## Addendum 2026-10-01: 0.4.1

One file joins the delta (83: 79 modified, 4 added): `common/speculative.h`; replay 83 / 83, 97 patches.

| patch | files | what |
|---|---|---|
| `apply_spec_sampling_042` | `sampling.h`, `sampling.cpp`, `speculative.h`, `speculative.cpp`, `server-context.cpp` | speculative sampling for the MTP draft of a request that samples (temperature above 0). The draft was each step's top candidate and the verification an exact match against the target's own draw, so a draft token was accepted with probability p(x). The drafter now draws each token from q - its top ten candidates through the target's top-k, top-p and min-p (on their probabilities at temperature 1, as the target's chain applies them) and the target's temperature (`common_sampler_spec_draw_q`) - and records q's support; `common_sampler_sample_and_accept_n_spec` computes p (the chain's samplers but its final dist, then dist's softmax) and accepts a draft token with probability min(1, p(x) / q(x)), replacing a rejected one with a draw from max(0, p - q), and draws from p after a draft accepted whole (Leviathan et al. 2023, Chen et al. 2023): each token is distributed exactly as `common_sampler_sample` draws it, a grammar's rejection sampling included. The first position is accepted with probability sum min(p, q): on Chinese prose at Jan's sampling (temperature 0.7, top_k 20, top_p 0.8) 63.4 -> 66.9%, the second 58.0 -> 60.0%, on the same contexts. Greedy decoding, mirostat and backend sampling keep the exact match; a replay after a checkpoint restore accepts the tokens it holds. With p_min, a draft shorter than n_min is kept: its length depends on the tokens it drew, and dropping it biases them (the Monte Carlo's control stream). A sampled request's draft takes its own cap by generating slots, `STRIX_SPEC_DRAFT_BY_SLOTS_SAMPLED` (the manager sets 2,2,1,1,0). `tools/spec_sampling_mc.py` compiles the shipped functions alone and checks them by Monte Carlo. `STRIX_SPEC_SAMPLING=0`: the exact match as before; `STRIX_SPEC_SAMPLING_STATS=1`: the expected acceptance per draft temperature and of the greedy draft in the log; `STRIX_SPEC_QSCALE`: q's temperature over the target's (1) |

## Addendum 2026-10-03: 0.4.2

No new files (83: 79 modified, 4 added); replay 83 / 83, 98 patches.

| patch | files | what |
|---|---|---|
| `apply_mmb_kq_042` | `mmb.cu` | UD-Q4_K_XL's routed experts on the matrix-core GEMM. The tuned prefill kernels covered UD-IQ4_XS's expert types (IQ3_S gate/up through glu3, IQ4_NL down); UD-Q4_K_XL's (gate/up Q4_K, down Q5_1) took the stock MMQ path, and a reader's llama-bench on Linux read pp2048 804 t/s against 1254 for UD-IQ4_XS. glu3 now also dequantizes Q4_K (a quarter of a 144-byte superblock a step: one nibble of 16 bytes a thread, the sub-block's 6-bit scale and min decoded at load time) and the routed plain kernel Q5_1 (two 24-byte blocks a step, the fifth bits spread to their bytes with one multiply). Q4_K_XL prefill 875.9 -> 1126.7 t/s at 2K tokens, 982.6 -> 1234.3 at 9.8K, 850.4 -> 1054.1 at 95.6K. Not bitwise against the stock path (BF16 weights instead of Q8_1 activations): paired perplexity 40 chunks of 8192 +0.84% (t = 1.2, lower in 17 of 40), 80 chunks of 4096 -0.05% (t = -0.6, lower in 43 of 80). UD-IQ4_XS is untouched: the probes identical. `STRIX_MMB_KQ=0`: the stock path |

## Addendum 2026-10-04: 0.4.3

Two files join the delta (85: 81 modified, 4 added): `moe-weighted-reduction.cu` and `moe-weighted-reduction.cuh`; replay 85 / 85, 99 patches.

| patch | files | what |
|---|---|---|
| `apply_prefill_043` | `ggml-cuda.cu`, `hc-cn.cu`, `.cuh`, `mmb.cu`, `.cuh`, `mmid.cu`, `moe-weighted-reduction.cu`, `.cuh`, `norm-gated.cu`, `.cuh`, `llama.h`, `llama-context.cpp`, `llama-graph.h`, `.cpp`, `llama-lazy-reader.h`, `qwen4exp.cpp`, `server-context.cpp` | a prompt's pass, shorter; each piece gives the same bits (the three probes). The shared expert's tail: `build_moe_ffn` takes a hook (`moe_before_weighting`) that qwen4exp uses from 512 tokens to build the shared expert and its sigmoid gate after the routed down projection, before the weighting, so the weighted reduction, the MUL by the gate and the ADD are adjacent and one kernel (`STRIX_SHEXP_TAIL`, `STRIX_SHEXP_MOVE`), its multiply and add separate round-to-nearest instructions as the ops were: -60..-77 ms an 8K-token batch (the shared expert built first let the allocator give glu3's output the routed input's buffer). XRES (`STRIX_HC_XRES`): when the Q8_0 gate GEMM's fused mix is the only reader of the hyper-connection combine's F32 normalized streams, the combine stops writing them and stores a per-row scale; the mix forms them from the F32 residual, the scale and gamma, the same two products in the same order: the combine 588 -> 439 ms, the mix +33. `STRIX_NORM_SCALE`: an RMS norm followed by a scale (the delta net's q/k L2 norms) is one `rms_rows_f32` with a post scale and bias: ~-42 ms. `mmid.cu`: the sorted route from 64 tokens on (`STRIX_IDS_ROUTE_MIN`; it took over above 4096 only, and the single-wave helper cost 0.62-0.68 ms of a routed GEMM at 1984 tokens). `mmb.cu`: the routed down projection's tiles in three classes, 32 / 64 / 128 tokens (`mmb_build_desc3`; `STRIX_MMB_DOWN_TILES=2`: two): -5% at 2K. The PLE rows: the server announces a prompt's next chunk at the size the step really takes (`n_prompt_step`) from 64 tokens, the reader starts it once the current batch has taken its rows (`defer_prefetch` / `launch_deferred`), threads chained instead of joined; a prompt's first chunk is gathered at once (`llama_strix_pregather_now`, `STRIX_PLE_NOW=0`: off) into a pending entry that `take_pregathered` waits for; a batch whose rows contain a whole gathered entry or a run of 1024 of its rows (a chunk after other conversations' tokens) takes it and gathers the rest. The last chunk of a 110K-token prompt 256 -> 3 ms. Measurement only: `STRIX_GPU_TIMELINE_ROWS`, the rows the GPU timeline keeps. Fresh prompts 1K / 2K / 4K / 8K / 16K: 920 / 1072 / 1122 / 1267 / 1245 -> 964 / 1148 / 1227 / 1302 / 1301 t/s |

## Addendum 2026-10-05: the prefill gap round

No new files in the delta (85: 81 modified, 4 added); replay 85 / 85, 100 patches. Measurement only:
no kernel and no graph changes, so every number 0.4.3 measured still holds.

| patch | files | what |
|---|---|---|
| `apply_moe_glu_kq_perf` | `test-backend-ops.cpp` | `STRIX_MOE_GLU_PERF` gains UD-Q4_K_XL's expert types. 0.4.2 gave glu3 a Q4_K dequantization and the routed plain kernel Q5_1, but `make_test_cases_perf` still built only IQ3_S / IQ4_NL / IQ4_XS cases, so the kernel carrying ~37% of that file's prefill (gate/up 356 ms of a 2048-token batch) could not be timed on its own. `GGML_TYPE_Q4_K` joins the fused gate/up cases and `GGML_TYPE_Q5_1` the down cases, and `STRIX_MOE_GLU_TYPES` takes a list of names (`iq3_s` `iq4_nl` `iq4_xs` `q4_k` `q5_1` `q8_0`) where it only took `all` before; the default is unchanged. A routing dump is replayed only at a token count its `n_rows` divides, so `rows2k.txt` (19840) runs at 1984 and `rows.txt` (81560) at 8156. `tmp/qk/build_tbo.py` builds the binary against the release kernels, and `tmp/qk/kernel_lab.py` drives it. `bootstrap.py --build` configures with `-DLLAMA_BUILD_TESTS=OFF` and the separate `build-dev` tree is a full HIP recompile behind, so the test binary is built in the release dir instead; only `ggml-cuda.cu` comes back out of that (its source is newer than its object) and `mmb.cu.obj` is not recompiled at all, so the routed expert kernels the lab times are the release build's own object. The HIP build is not byte-reproducible in general - recompiling an unchanged `ggml-cuda.cu` twice gives two different `ggml-hip.dll` hashes - so the claim is about that object, not about the DLL. The lab reproduces the in-model profile: Q4_K gate/up 8.00 ms a layer at 1984 tokens against 7.75 ms read off the 2048-token GPU timeline (the two rounds bracket each other; the case moves a few percent run to run, so `tmp/qk/kernel_lab.py` pins one routing line with `STRIX_MOE_IDS_LINE` and reports the minimum of interleaved runs). `STRIX_NORM_PERF` adds the delta net's narrow-row norms, the shapes the GPU timeline charges 95.9 / 17.6 / 11.0 ms to in a 2048-token prompt for the same bytes an element; alone they are 478 / 87 / 522 us, so `rms_rows_f32` runs at 210.5 GB/s for [128,48,2048] - the machine's measured peak - and the graph's 5.6x is the node's context, not the kernel (`tmp/qk/norm_lab.py`) |

## Addendum 2026-10-10: 0.5.6

One new file in the delta (98): `common/parsers/qwen3-coder.cpp`; replay 98 / 98, 119 patches.

| patch | files | what |
|---|---|---|
| `apply_server_056` | `server-context.cpp`, `.h`, `server.cpp`, `server-task.cpp` | **Images in the own-tokens splice:** the prompt after the common prefix is walked as runs of text and images; a slot's image is kept where the prompt has the same image (chunk id, size) at the same place, with the prompt's data (the slot may hold a placeholder), and the prompt's images after the cut follow as they came. A follow-up with a screenshot after a split answer token: 3,595 of 3,595 slot tokens kept (0 in 0.5.5). **Issue #18:** a sampled request on one generating slot drafts the sampled table's length or one more, per block of 32 passes, by tokens a second: 1 + acc1 + acc2 against that plus acc2 * acc2 / acc1 (the third position's acceptance estimated from the first two, within 0.05 of its measurement on code, an agent and prose), the longer pass costing 1.14x at short contexts rising to 1.30x at 85K. Blocks because each switch rebuilds the verify graph (per-pass switching measured the longer draft ~30 ms dearer than it is); measured times and the third position's own acceptance went stale and are only logged. One slot, sampled, two rounds against fixed 2: an agent 23.1 -> 21.7 ms a token, code 18.4 -> 16.4, prose unchanged. `STRIX_SPEC_LEN_ADAPT=0` keeps the table. **Draft length without a reload:** `STRIX_SPEC_DRAFT_CAP` at start, `POST /strix/spec {"draft_max": n}` while running, below the starting `--spec-draft-n-max`. **Disk tier:** entries without draft rows (saved with MTP off) are kept when the server drafts - passed over by the lookup, used again with MTP off - instead of being deleted as an older format. |
| `apply_chat_056` | `server-common.cpp`, `server-chat.cpp` | **Issue #17:** `reasoning_effort: "none"` also sets the `enable_thinking` kwarg (which `--reasoning on` sets true and the template reads); `/v1/messages` `thinking.type: "disabled"` maps to it; `"minimal"` / `"max"` map to low / high; a template's refusal is HTTP 400 with its message. **Issue #12:** on "System message must be at the beginning." (templates that take one system message first, Swift-1.5's) the leading system and developer messages are joined and the request rendered again - `/v1/responses` `instructions` plus a developer message failed with HTTP 500. |
| `apply_tool_params_056` | `json-schema-to-grammar.cpp`, `parsers/qwen3-coder.cpp` | alternatives that each require a parameter but share none (Deep Rulith's OpenCase: caseType or caseId) flattened to no required parameter, and the Qwen3-Coder grammar allowed an empty call every alternative rejects (made twice at 260K tokens, 10-09). The flat schema carries `minProperties: 1` and the grammar takes at least one parameter. |

Not a patch: `tools/manager.py` places the experts by what loads really take. The estimate was 4.65 GiB short at a
768K pool, ubatch 16384 and image input (64 GB carve): the draft's compute buffer is not capped past 512K cells (2687 ->
3812 MiB), and the server holds ~10.9 KiB a cell beyond its buffer report past 2^19 cells. Refitted, seven measured
settings fall within +-1.2 GiB. The server's dedicated GPU memory is read while it runs and its peak kept per model and
settings (`placement.json`) for the next load; a load that fills the carve as it becomes ready is reloaded once, placed
by the measurement. The Q6_K trunk copies count when `trunk_decode_q6k` is on (2.6 GB, issue #18). A new draft length
goes to the running server (`apply_live`, the app skips the reload). `bootstrap.py` finds Visual Studio through
`vswhere`, the Build Tools under Program Files (x86) included (issue #17).

## Addendum 2026-10-09: 0.5.5

No new files in the delta (97); replay 97 / 97, 116 patches.

| patch | files | what |
|---|---|---|
| `apply_own_tokens_055` | `server-context.cpp`, `server-task.h` | a conversation's next request (a tool call's follow-up, an agent step, the next chat turn) sends the model's answer back as text, and tokenizing it again does not always give the tokens the model sampled (a word it produced in two pieces comes back as one): the cached prefix stopped inside the answer and everything after the answer's start was processed again. In the request logs of 10-08/09, 3 of 7 continuing requests lost their whole previous answer this way (2.8K, 5.7K and 0.7K tokens), the text identical in all three. `splice_own_tokens` keeps the slot's own remaining tokens when the prompt's text after the common prefix starts with their text and tokenizes only the rest (text only after the common prefix, an image before it stays; a cut on a character boundary; at least 16 tokens; message spans found again); whitespace the template trims off an answer's end is kept when the text continues with the end of the turn. `STRIX_SPLICE_OWN=0` turns it off. With MTP, an end of turn accepted as a draft token was followed into the cache by the tokens drafted after it, and the next turn parted from the slot before its state: the tokens after an end of turn are now dropped like a rejected draft. The 10-09 web-search follow-up replayed on one slot: 4,228 tokens processed in 3.7 s against 7,822 in 6.9 s; a third greedy chat turn 326 -> 26 tokens processed. |
| `apply_kb_uncached_f16_055` | `qwen4exp.cpp` | issue #16: with the block-key cache wired in, `build_qsa_top_k` scores blocks with the keys read back from the F16 cache (a rebuild graph too, since `apply_qsa_kb_rebuild_f16`); without it, with the pooled keys just computed, in F32. The cache is left out when switched off (`LLAMA_QSA_BLOCK_KEY_CACHE=0`), with several KV streams, for an image ubatch, and - through `kb_pos_dup` - for every text ubatch after the server has seen one image, so a server that had seen an image computed text above the sparse-attention threshold slightly differently from a fresh one and greedy output could change. The uncached keys are rounded to F16 and back (the reporter's change). An 11K-token prompt, greedy with top-5 log-probabilities: after an image it differed from token 83, as with the cache off; now before and after an image, cache on and off and the default without the change are bit-identical. The cached path is unchanged. |

Not a patch: `tools/manager.py` starts the server with `--reasoning off`, or `--reasoning on --reasoning-effort
<level>`, instead of the deprecated `--chat-template-kwargs` (issue #15). Both set the template defaults the kwarg did
(`enable_thinking`, `reasoning_effort`), the level keeps its mapping (high -> xhigh; the template accepts only low /
medium / xhigh), and a request's own `chat_template_kwargs` or `reasoning_effort` still overrides them.

## Addendum 2026-10-08: 0.5.4

No new files in the delta (97); replay 97 / 97, 114 patches.

| patch | files | what |
|---|---|---|
| `apply_image_text_compact_054` | `llama-memory-hybrid-idx.cpp` | issue #13: once a conversation held an image, every later ubatch of it took the dense sparse-attention inputs (a mask and a per-block bias over n_kv x n_tokens filled on the host, the ubatch held to 2^27 cell-token pairs since #2), and the text after one image prefilled ever slower: 1,465 t/s before it, 224 t/s at 140K. `qsa_scalar_visibility` (from upstream) refused the compact inputs whenever an image cell was in the window. For a text ubatch whose conversation's image cells all lie before its first position every image cell is visible by position alone, and `set_input_qsa` already ranks the cells and fills the compact metadata in that rank space (the MTP draft context's path after every image), so such a ubatch keeps the compact inputs; the 2^27 bound applies only to batches that still take the dense ones (an image, several conversations, text an image does not wholly precede), and no longer to the MTP draft's catch-up. `STRIX_IMAGE_TEXT_COMPACT=0` restores the dense inputs. 147K tokens with a 1920x1080 image after the first 45K: 404 -> 1,259 t/s (text alone 1,351); against the dense inputs, an image question 75K tokens back answered the same and a 300-token greedy continuation came out identical. |

Not a patch: `tools/manager.py` places experts by what the carve really leaves. The decode pauses of 0.4.7 - 0.5.2
(~0.3 s every ~5.3 s on the 64 GB carve) were the display driver moving a 1.5 GiB device buffer between system memory
and the carve: the server's shared GPU memory dropped by exactly that at every pause and came back right after
(`tmp/stall/vram_watch.py`), with no allocation by the runtime in between. Three drafts' buffers are ~1.8 GiB more than
one's, so with other programs holding 2.8 GiB of the carve and the estimate allowing the server 61.4 GiB less a 2.5 GiB
margin, one buffer did not fit; 0.5.3's one-draft cap hid that. The vision projector (1.1 GiB) was not counted at all.
The estimate now counts the projector, and the dedicated GPU memory other programs hold at load time (the "GPU Adapter
Memory" counters less any llama-server's) beyond 0.7 GiB, plus 0.5 GiB for them to grow, comes off the server's share;
one or two more layers' experts go to system memory, which reads as fast. Drafts are back to 3. With vision on, against
0.5.3 as shipped: code decode 42.2 -> 63.2 tok/s, prose 38.6 -> 42.4, no pause over 250 ms in either.

## Addendum 2026-10-08: 0.5.3

Eight new files in the delta (97): `common/json-schema-to-grammar.h`, `common/chat-auto-parser-generator.cpp`,
`common/parsers/deepseek.cpp`, `kimi-k3.cpp`, `minicpm5.cpp`, `minimax-m3.cpp`, `muse-glimmer.cpp` and
`tools/server/server-chat.cpp`; replay 97 / 97, 113 patches.

| patch | files | what |
|---|---|---|
| `apply_tool_schemas_053` | `json-schema-to-grammar.h/.cpp`, `parsers/parsers.cpp`, the per-parameter parsers (`deepseek`, `kimi-k3`, `minicpm5`, `minimax-m3`, `muse-glimmer`), `chat-auto-parser-generator.cpp` | every tool-call format that builds one grammar rule per parameter read the top-level `properties` only, so a tool whose parameters sat behind a top-level `$ref`, were split over `allOf` parts or given as `$ref` alternatives got no parameter and every call arrived as `{}` (0.5.2 covered plain `oneOf` in the Qwen3-Coder format only). `common_tool_parameters_flatten` spells the properties and the required list out for all of them: local `$ref`s resolved, `allOf` parts each requiring what they list, `oneOf` / `anyOf` alternatives offered together and required only where every alternative requires them, `{not: {required: [...]}}` leaving those out; a plain schema comes back unchanged, and what the model is shown is not affected. In json-schema-to-grammar, an `allOf` of objects required every property of a required part: Rulith's `ApplyBatch` atoms (`allOf: [$ref atom, {not: {required: [naf]}}]`) had `negated` and `naf` forced into every call and were rejected. Parts now require their own list, a `{not: {required}}` part removes those properties, nested `allOf` and `oneOf` parts contribute theirs, and `allOf` of a string `$ref` and a constraint is a string. Five schema shapes: 1 / 5 called as asked with 0.5.2, 5 / 5 now. |
| `apply_agent_compat_053` | `server-chat.cpp`, `server-common.cpp`, `server-context.cpp`, `server-task.h/.cpp` | coding agents on `/v1/responses` and `/v1/messages` (issue #12). A system or developer message after the conversation started (Codex sends developer messages between turns; Claude Code 2.1 puts system messages inside `messages`) made templates that only allow a leading one raise "System message must be at the beginning." and the request return HTTP 500: with such a template it is passed on as a user message saying whose it is. Responses `custom` tools and `namespace` tools were dropped: a custom tool is offered as a function taking one string `input` (a grammar format is described to the model), the tools of a namespace as functions of their own (`NAMESPACE__NAME` only when a name repeats), tools in `additional_tools` input items too; `custom_tool_call` / `custom_tool_call_output` items and function calls carrying a `namespace` map onto those functions. The map travels as `strix_resp_tools` into `task_params::resp_tools`, and the response side gives calls back as `custom_tool_call` items (input unwrapped) or `function_call` items naming their namespace, non-stream and streamed (a custom call's events are held and sent whole at the end, with `custom_tool_call_input.delta` / `.done`). Hosted tools (`web_search`, ...) are named in an `X-Rulith-Unsupported-Tools` response header; reasoning items with only `encrypted_content` and items of unknown types are left out instead of failing the request; tool outputs with non-text parts say what was left out; a `tool_choice` object maps to `required` or the `allowed_tools` mode. Twelve Codex- and Claude Code-shaped requests: 0.5.2 failed 8 of the first 9, now 12 / 12; the Codex CLI 0.160 (plain and with its custom `exec` tool) and Claude Code 2.1.247 finish an edit-and-run task. |

Not a patch: `tools/manager.py` passes `--spec-draft-n-max 1` when experts are placed in host memory (the 64 GB
carve). With host experts and two or more draft steps, decoding paused ~400 ms every ~5.3 s of wall clock (since
0.4.7); one draft step has no pause.

## Addendum 2026-10-08: 0.5.2

Two new files in the delta (89): `common/json-schema-to-grammar.cpp` and `common/parsers/parsers.cpp`; replay 89 / 89, 111 patches.

| patch | files | what |
|---|---|---|
| `apply_request_log_052` | `server-queue.h/.cpp`, `server-context.cpp` | `STRIX_REQUEST_LOG=<file>`: every completion request, one JSON line per task when it ends or when the client goes away first - the HTTP body, the parameters the server derived from it (grammar, triggers, parser; not the rendered prompt), the raw generated text, the parsed message with its tool calls, the stop reason and token counts. Long values that repeat between requests (each message, the tools, the grammar) are written in full once per file and referenced by hash afterwards. Files roll over at `STRIX_REQUEST_LOG_MAX_MB` (1024) to `<file>.2`, `.3`, ...; nothing is deleted. The app's manager points it beside the server's console log unless `RULITH_REQUEST_LOG=0`. |
| `apply_tool_schemas_052` | `json-schema-to-grammar.cpp`, `parsers/parsers.cpp` | the Qwen3-Coder XML tool grammar took parameters from the top-level `properties` only: a tool given as `oneOf` / `anyOf` alternatives (Rulith's `OpenCase`) had none, and every call arrived as `{}`. `foreach_parameter` offers the union of the alternatives' properties, each required only where every alternative requires it. `allOf` over a non-object type (an array `$ref` plus `minItems` / `items`, `QueryBoard`'s `selector.roots`) was merged as an object and forced to `{}`; its keywords now merge into one schema that is visited (objects and enum / const intersections keep the old path). Verified end to end with Deep Rulith's schemas. |

## Addendum 2026-10-08: 0.5.1

No new files in the delta (87); replay 87 / 87, 109 patches.

| patch | files | what |
|---|---|---|
| `apply_prefill_051` | `gated_delta_net.cu/.cuh`, `ggml-cuda.cu`, `mmb.cu/.cuh`, `llama-memory-recurrent.cpp/.h`, `llama-context.cpp/.h`, `llama-ext.h`, `speculative.cpp`, `qwen4exp.cpp`, `test-backend-ops.cpp` | prefill. The delta net's pending records of a cell are replayed on the GPU by the graph's own kernel (`ggml_backend_cuda_gdn_replay`, found through the backend registry's proc address) instead of on the host: a turn start's checkpoint 108.8 -> ~3 ms (`STRIX_GDN_DEV_REPLAY=0` restores the host replay). Fusions, bitwise: the router's F32 product as a two-term F16 tile kernel, the delta net's beta and alpha products in one pass over F16 copies of their weights, the attention gate GEMM fused with the gated RMS norm (`STRIX_MMB_F32_TILE2`, `STRIX_MMB_NARROW_PAIR`, `STRIX_MMB_F16W`, `STRIX_GNORM_FUSE`). The routed expert kernels wait on LDS only at their tile barriers (`__syncthreads` drained every load), keep raw weight fields in whole registers and decode Q4_K scales at dequantization. The Q8_0 down projection of five layers and the IQ4_XS gate/up of layer 2 leave MMQ for the routed kernels (12.1 -> 5.7 and 11.3 -> 7.1 ms at 2K tokens; bf16 like the other expert layers, so those six layers are not bitwise; 40-chunk paired perplexity 2.8134 -> 2.8070, `STRIX_MMB_Q8=0` restores MMQ). MTP's catch-up reads the target's rows in place and a long prompt chunk keeps only its last verify row (`STRIX_MTP_INPLACE=0`). test-backend-ops' routing replay wraps at the end of its file. |

## Addendum 2026-10-06: 0.4.9

No new files in the delta (87); replay 87 / 87, 108 patches.

| patch | files | what |
|---|---|---|
| `apply_poll_sync_049` | `ggml-cuda.cu` | the backend's synchronize polls a sequence number that a kernel at the end of the stream writes into fine-grained host memory, after an event query that submits what the launch left batched. hipStreamSynchronize returned ~0.65 ms after a 2000-node graph had finished (ROCm/TheRock#8786). Streams nothing went into since they were last seen done (marked by the graph, the async copies, event waits) are not waited for; past 100 ms the runtime's sync takes over. STRIX_POLL_SYNC=0 restores the runtime's sync, `ab` alternates with STRIX_AB=n, and STRIX_AB logs per-side step medians. |

## Addendum 2026-10-06: 0.4.8

No new files in the delta (87); replay 87 / 87, 107 patches.

| patch | files | what |
|---|---|---|
| `apply_split_no_free_048` | `llama-model.cpp` | an even layer split when every device reports no free memory. An integrated GPU counts pinned host buffers against itself: with 37 GiB of experts in ROCm_Host, HIP reported 0 MiB free when the MTP draft loaded, the split by free memory became 0/0, and the draft failed with "invalid vector subscript". |
| `apply_ckpt_turn_cuts_048` | `server-context.cpp` | a prompt is cut only at its last user message, when no checkpoint stands there. Its other checkpoints come where batches end anyway, spaced as the anchors were. The eviction keeps the checkpoint at the current task's last user message (an edit of the message forks there). STRIX_CKPT_ANCHORS=1 restores the cuts at the first user message and at the spaced turn starts. |

## Addendum 2026-10-06: 0.4.7

No new files in the delta (87); replay 87 / 87, 105 patches.

| patch | files | what |
|---|---|---|
| `apply_host_buft_padding` | `ggml-cuda.cu` | the host buffer type (ROCm_Host) pads quantized rows to MATRIX_ROW_PADDING and zeroes the padding, as device buffers do. An integrated GPU computes on it in place, and MMQ reads each matrix's last row that far: with experts there, the 640-wide down experts read the next tensor's bytes as scales and every MMQ batch (17 tokens up) came out NaN. |
| `apply_dense_ubatch_own_cells` | `llama-memory-hybrid-idx.cpp` | issue #10, fix by @obsidience: the dense-input ubatch bound (#2) sizes n_kv by the cells the graph views for the batch's own conversations (get_kv_window), not the pool's last used cell, so an image's split no longer depends on other slots. Reporter's repro: A1 vs A2 differed from token 6 on 0.4.6, identical text and logprobs now. |

## Addendum 2026-10-06: 0.4.6

Two files join the delta (87: 83 modified, 4 added): `src/llama-arch.h` and `src/llama-arch.cpp`, for the
draft head's two tensors. Replay 87 / 87, 103 patches.

| patch | files | what |
|---|---|---|
| `apply_mtp_lowrank_head` | `llama-arch.h`, `llama-arch.cpp`, `llama-model.h`, `qwen4exp.cpp` | the MTP draft's head through a low-rank pre-score: with `blk.N.nextn.lr_proj` (F16, 2560 -> 512) and `lr_scores` (Q8_0, 512 -> vocab) in the draft file, a step picks `STRIX_MTP_LR_K` (512) candidates (`top_k` of the pre-score), gathers their rows of the head (`get_rows`, the target's Q6_K) and multiplies only those; the other logits are -inf (`fill` + `set_rows`). Without the tensors, or with `STRIX_MTP_LR_K=0`, the full head as before. |
| `apply_ot_host_buft` | `arg.cpp` | `--override-tensor` also lists each device's host buffer type (`ROCm_Host`), which an integrated GPU computes on in place: weights can live in pinned system memory outside the carve. Six layers of experts there (8 GB) left prefill and decode unchanged at the 96 GB carve. |

## Addendum 2026-10-05: 0.4.4

No new files in the delta (85: 81 modified, 4 added); replay 85 / 85, 101 patches.

| patch | files | what |
|---|---|---|
| `apply_xres_fix_044` | `ggml-cuda.cu` | issue #7. XRES (0.4.3) let the HC gate mix form the normalized streams from the combine's F32 residual output, a read the graph does not record. Every layer's residual is read again by the next combine and stays allocated; the last layer's is read by nothing after the head's mix, so the allocator could place that mix's own down projection output ([T, 320] F32: exactly 64 rows of the residual) over it. With the head run on many rows in one ubatch - llama-perplexity, -b 4096 at -ub 512 or 2048 - its first 64 rows were read overwritten: PPL 39516 / 17.48 against 2.8755 / 2.8511 (4 chunks of 4096, the docs text). The plan now forms the streams only when the residual is read again after the mix or is a graph output; otherwise the combine writes the F32 streams as before 0.4.3. Perplexity at -ub 512 / 1024 / 2048 / 4096 with XRES on equals XRES off, and the probes give 0.4.2's bits: the server's head works on its output rows only (one in a prompt), below XRES's 512-row floor, and every layer mix keeps the shortcut |
