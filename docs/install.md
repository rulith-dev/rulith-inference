# Installing and running it

**The short way** is the installer from the [releases page](https://github.com/rulith-dev/rulith-inference/releases):
`Rulith-Inference_<version>_x64-setup.exe` carries the desktop app and, under `runtime/`, everything it
runs — `llama-server` built from the pinned fork, the eight ROCm DLLs it imports plus the gfx1151
kernel libraries, the Visual C++ and OpenMP runtimes, the manager and an embedded Python. On a
machine set up as in section 1, with the model files from section 5 already in LM Studio's folder,
it needs nothing else: install, open, load the model from the welcome screen. That path, including
exactly which five files to download and where to put them, is written out step by step in
[getting-started.md](getting-started.md). The bundle is made by
`tools/make_runtime_bundle.py` from a finished build; `BUNDLE.json` inside it records what went in
and the upstream commit it came from.

The rest of this page is the long way, for building it yourself. Four things have to be in place:
the machine set up correctly, a toolchain, a build, and the model files. The build is the easy part.

## 1. The machine

| | |
|---|---|
| GPU | Radeon 8060S / **gfx1151** (Ryzen AI Max+ 395). Other RDNA 3.5 parts are untested; the kernels are compiled for this target only. |
| Memory | 128 GB unified |
| **GPU carve** | **96 GB**, set in the BIOS |
| OS | Windows 11 |
| Disk | ~110 GB for the model, ~15 GB for the toolchain, ~10 GB for the build |

**Set the carve before anything else.** At 64 GB the 93.7 GB model does not fit: about 9.8 GB spills
into shared memory and the driver pages during decode. Measured, 64 GB against 96 GB: prefill 839 vs
942 t/s, decode 49.6 vs 38.6 ms/token, multi-turn 17-19 vs 26-30 tok/s — and the 64 GB arm had the
*higher* power limit. No software setting recovers this. It leaves Windows 31.6 GB, which is enough.

Raising the carve lowers the commit limit; on this machine that is better solved with a fixed large
page file than by shrinking the carve back.

## 2. Software prerequisites

- **Python 3.12+**
- **CMake** and **git** on `PATH`
- **Visual Studio 2022** with the C++ workload. The ROCm SDK's clang still links against the MSVC
  runtime and uses the Windows SDK headers, so `vcvars64.bat` has to exist — `bootstrap.py` finds it
  and merges its environment, so you do not need a developer prompt.

## 3. The toolchain

```bash
python bootstrap/bootstrap.py --toolchain
```

This creates `toolchain/rocm-venv`, installs the ROCm SDK into it as wheels, expands the devel SDK
(`rocm-sdk init` — the wheel is a tarball until then, and a bootstrap that skipped this had no
compiler and no device bitcode), and downloads ninja. About 5 GB.

**Use TheRock ROCm, not the system ROCm 7.1.** Same source, same flags: 458 → 735 t/s, +60% (TheRock 10.1).
This is the single largest factor in the prefill numbers and it is not a code change. Windows wheels
are the only form AMD ships — the native tarballs are Linux-only.

### The pin will expire, and you should plan for it

`ROCM_VERSION` is pinned to `10.2.0a20260925`. It comes from a **nightly index with a rolling window
of about 27 days**, so:

```bash
python bootstrap/bootstrap.py --toolchain --rocm-version <a version the index still has>
```

`--toolchain` prints the versions the index currently offers when the pin fails to resolve.

Up to 0.2.2 the pin was `10.1.0a20260910`, the last nightly of the 10.1 series, and the numbers up to
then were measured on it. On the same source 10.2 gives the same perplexity (2.6814 at 8K context), the
same output bit for bit on an 18.6K-token prompt, the same prefill and single-conversation decode, and
3-4% more with three or four conversations decoding (`docs/results/multi-stream-20260926.json`).

**Anything other than the pinned version is untested here.** A different SDK can change numerics, and
this project has twice shipped a regression that a short perplexity run could not see. If you move
off the pin, re-run the gates in [measuring.md](measuring.md) before trusting the build —
`tools/truncation_test.py --fill` at minimum.

If you have the pinned version working, **archive the wheels now**, because the index will not have
them in a month:

```bash
toolchain/rocm-venv/Scripts/python.exe -m pip download --pre \
  --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,devel,device-gfx1151]==10.2.0a20260925" -d toolchain/rocm-wheels
```

and install from that directory afterwards with `--no-index --find-links toolchain/rocm-wheels`.

## 4. Build

```bash
python bootstrap/bootstrap.py --fetch --patch --build
```

- `--fetch` clones `pwilkin/llama.cpp` at the pinned revision into `src/llama.cpp`.
  The result of `--fetch --patch` is what the `strixllama` branch of [rulith-dev/llama.cpp](https://github.com/rulith-dev/llama.cpp/tree/strixllama) holds as a single commit, if you would rather read the delta than replay it.
- `--patch` overlays `patches/iq3s-kernel/` and runs the scripts in order.
- `--build` configures and compiles, then copies the runtime into `bin/hip/` (`bin/hip-rocm101/`
  up to 0.2.4, named after the ROCm 10.1 it was first built with; the manager still runs a tree that
  has only that folder).

`--build` re-runs cmake every time on purpose: cmake resolves its `*.cu` glob at configure time, so a
file the patch set adds or renames is invisible to an existing `build.ninja`.

Check it against the recipe:

```bash
python tools/replay_bootstrap.py     # clean upstream + patch set == the 30 files. Expect 30 / 30.
```

## 5. Model files

Three files, none of them in this repository.

**The model** — Unsloth's `Qwen3.8-Flash-Next-GGUF`, quant `UD-IQ4_XS`, three shards, 93.7 GB. Point
the download at the first shard's siblings; the manager groups shards by name and refuses to start if
one is missing.

**The vision projector** — `mmproj-F16.gguf`, 904 MB, shipped in the same repository. Put it *beside*
the model: the manager looks for it there. Without it the server has no multimodal capability at all
and rejects any request carrying an image. It costs 904 MB of GPU memory and nothing else.

**The MTP draft** — Unsloth's `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (or `-shared-Q8_0`, under
1% apart) next to the model works as it is: the manager picks it up and every figure in this
repository except the last 5–9% of decode was measured with a draft like it. To get that last part,
give the draft its own output projection:

```bash
python tools/make_draft_head.py --base <model directory>/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
```

The result is written beside the base as `…-shared-Q4_K_M-head-iq4_xs.gguf` (the target shard holding
`output.weight` is found in the same directory; `--target` overrides). Keep it there: the
configuration page only lists drafts found under the registered model directories, so a copy kept
elsewhere never appears in it. Press *Rescan* under Model › Library and the `*-head-*` file is
preferred automatically for every profile that has not chosen a draft by hand.

`--head-only` writes just the quantised tensor instead, `mtp-<family>-head-<type>.gguf` (349 MB):
that is the release asset. The manager's `merge_draft_head()` splices such a file with the shared
draft in the same directory at rescan, in pure Python, copying the base's key-value and tensor-info
bytes verbatim and appending one tensor — byte-for-byte the same tensors and metadata as this
script's output, verified against it. So an installed copy needs neither numpy nor `llama-quantize`.

Why it helps: the `shared-*` drafts borrow the target's Q6_K head, which streams 521 MB on every
draft step — 2.4 ms of a 4.3 ms step. Every drafted token is verified by the target, so a coarser
head can only cost acceptance, and at IQ4_XS acceptance did not move: +5% on English, +9% on Chinese.

### Already have them in LM Studio? Nothing to download

If LM Studio is installed, its download folder is picked up automatically: the manager reads
`downloadsFolder` from `~/.lmstudio/settings.json` and adds it as a model root. LM Studio lays
models out as `<root>/<publisher>/<repo>/<file>.gguf`, which is exactly the shape the catalog walks,
so everything already on disk shows up — **the same files, read in place, no import step and no
second 93.7 GB copy.** Both applications can keep using them; nothing is moved or rewritten.

If your models are somewhere else, add the directory to `roots` in `config/jan/settings.json`. Paths
outside a registered root are refused — that check is deliberate, since the manager runs model files
chosen over IPC.

## 6. Run

```bash
python tools/manager.py <<< '{"op":"catalog","data":{"refresh":true}}'   # find models, get their ids
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
python tools/manager.py <<< '{"op":"status"}'
python tools/manager.py <<< '{"op":"stop"}'
```

The server is then an ordinary OpenAI-compatible endpoint on `http://127.0.0.1:8080/v1`.

**The defaults are the measured configuration.** A first load runs with context 262144, batch and
ubatch 8192, flash attention on, sparse attention and MTP on for this model, one slot — the
settings every number in [results.md](results.md) was measured with — so nothing has to be set to
get the claimed performance. On a carve this does not fit, the display driver puts what is left in
shared GPU memory (see *Shared GPU memory* below), which is slower; a smaller context is the setting
to change. Sparse attention has to stay on above ~64K: dense attention runs out of memory there, and it
is slower below it.

### Conversations come back from disk

The server's RAM prompt cache (`--cache-ram`, on by default) keeps a short finished conversation's
tokens, state and recurrent checkpoints so it can be resumed on a later request. `prompt_cache_disk`
(the *Keep conversations on disk* switch) adds a disk tier under `config/jan/prompt-cache`, up to
the size in the profile, oldest first, that survives restarts. It is off by default since 0.1.13:
nothing is written to the SSD unless you turn it on, and a conversation pushed out of its slot is then
processed again when it returns.

What it writes, since 0.1.13: a conversation's attention rows - about 29 KB per token on this model,
the MTP draft's included - once each, 4096 positions at a time as they are computed; and its recurrent
state, the part that changes with every token, only when the conversation leaves memory, because
another one needs its cells or the server is stopped (the manager's unload has it write first). That is
the state at its end and its last prompt's two latest checkpoints, 0.33 GB. A 173K-token conversation
reads back in 1.6 s and a 33K-token one in 0.4 s, and the next turn processes its own tokens only; a
restored conversation answers with the same tokens and probabilities as it would have kept resident
(`docs/results/disk-tier-v3-20260924.json`). If the server dies rather than being stopped, the
conversations it held come back only as far as the last time they left memory, and the rest is
processed again. The directory holds only what this version writes: entries of an older format are
deleted when the server starts, so after an upgrade they are processed once more. Conversations with
images are not persisted. The slot save/restore endpoints are not a substitute: they carry the state
but not the checkpoints, and without one a hybrid model re-processes the whole prompt.

### Several long conversations at once

With more than one slot the slots share one KV pool, and by default it is the context: 262144 tokens
for all of them together, so a second long conversation pushes the first out: processed again when
it returns (~3 minutes at 173K tokens), or read back from the disk tier in ~1.6 s when that is on.
`kv_pool` - *Shared context pool* under Model › Configuration › Concurrency, which takes effect
with several slots - makes the pool larger while every conversation stays capped at the context: any
size, one allocation, rounded up to 256 cells. Each token beyond the context costs this model about
39 KB of GPU memory and about 24 KB of Windows commit (RAM plus page file). With a q8_0 cache it is
about 26 KB of GPU memory: 21 KB of cache, and about 5 KB for the MTP draft, whose attention mask
spans the whole pool.

On this machine commit is the limit, not the carve. With `kv_pool` 393216, conversations of 173K and
155K tokens both stay resident and each answers its next question in under a second, with 2.6 GB of
commit left at the lowest point (0.1.12's disk tier, which gathered a long conversation into a 5 GB
buffer, took it to 0.04 GB); 524288 failed with "bad allocation" although the carve had room for it
(about 86 GiB allocated). On the default pool, swapping the same two through the disk tier keeps
9.6 GB free. So for long conversations, enlarge the page file first - it costs disk space only. A
loaded server with less than 8 GiB of commit left gets a warning on the page.

With q8_0 the carve is the limit instead. With four slots, a micro-batch of 8192 and MTP, a 512K pool
loads with 84 of the 96 GB in use; at 768K the MTP draft's compute buffer (3.9 GB, the mask over the
whole pool) no longer fits. With MTP off there is no draft context, and room for a larger pool. Measured in
`docs/results/kv-pool-20260924.json` and `docs/results/disk-tier-v3-20260924.json`.

### KV cache: f16 or q8_0

*KV cache type* under Context (`kv` in a profile) stores the attention keys and values as f16, the
default, or q8_0. q8_0 halves this model's K/V cache: at 262144 tokens 6 GiB becomes 3.2 GiB, and a
token costs about 21 KB of GPU memory instead of 32.5 (the sparse-attention indexer, its block keys
and the draft's cache stay f16). The disk tier stores 18 KB a token instead of 29. The output
differs from f16's about as much as f16's own does at another batch size: mean KL divergence 0.0139
+/- 0.0007 against f16 over 12K tokens of text, where f16 at two batch sizes gives 0.0116, and the
same top token 96.6% of the time; perplexity 3.154 against 3.117 (+/- 0.087). Prefill is 1-2% slower
at 86K tokens (the sparse-attention kernel reads f16, so each batch dequantizes the cache into its
layout); decode is the same within noise (median 29.3 against 29.7 ms/token over three runs, MTP
acceptance 63% against 65%).

Keys are rotated before they are quantized, as upstream does for quantized caches; values are not.
The sparse-attention prefill kernel sums probabilities times values on the matrix cores, and those
sums move in the last bit with the values of keys they weight by zero - the free cells after a
conversation's last one. Until 0.1.17 those held whatever an earlier conversation, or a rejected
draft, left there; f16 carried that bit too, and the next layer's rounding absorbed it, but the
inverse rotation of a rotated V spread it over 64 dimensions, and a conversation read back from disk
then parted from the same one kept in memory. Since 0.1.17 a cell is zeroed as it is freed (see
below), so a rotated V would now be safe; it stays unrotated because the disk tier's q8_0
conversations are stored that way, in rows of the same size, and the quality is the same within
error (mean KL divergence 0.0137 with V rotated too).

Switching the type discards the conversations the disk tier holds of the other type when the model
next loads. Measured in `docs/results/kv-q8-20260924.json`.

### Several conversations decoding at once

Speculative decoding drafts tokens that the next step verifies. For one conversation that is worth
+74% (38.7 against 22.3 tok/s at 20K tokens). With several at once it costs more than it returns: every
verified token makes the step read the weights of about ten more of this model's 512 experts, and
conversations decoding together cannot share them - four conversations with three draft tokens verify
16 tokens a step, which touch ~138 experts where 4 tokens touch ~39. The expert kernels ran at the
memory's bandwidth in the cases 0.1.17 measured (251-261 GB/s), so the lever was to verify fewer tokens.
The server therefore drafts `draft_max` tokens while one conversation generates, at most 2 while two to four do,
and none from five on (`STRIX_SPEC_DRAFT_BY_SLOTS`, set by the manager when MTP is on and there is more
than one slot; up to 0.2.2 none from four on). The draft model keeps processing every step, so drafting
resumes as soon as fewer conversations generate. 0.1.17 measured, summed over the conversations, ~20K
tokens each, q8_0 K/V:

| generating | 3 draft tokens | 2 | 1 | none | the cap |
|---|---|---|---|---|---|
| 1 | 38.7 | | | 22.3 | 3 |
| 2 | 35.7 | 39.9 | 40.6 | 34.7 | 2 |
| 3 | 40.5 | 44.7 | 42.3 | 43.6 | 2 |
| 4 | 37.9 | 42.7 | 41.1 | 49.4 | 0 (measured 47.4 with the draft model loaded) |

With speculation the random samples decide how many drafts are accepted, so single runs spread by
+/-10%; each entry is the median of three. Measured in `docs/results/concurrency-20260924.json`.

Since 0.2.3 a verify step of several conversations costs less, and drafting pays at four. Two of its
products fell off the vector kernel onto tiles they barely filled once the step carried more than a few
tokens: the MoE router, an F32 [2560 × 512], took the MMB kernel's 128-row tiles from 9 columns - four
workgroups for the whole GPU, ~250 µs a product - and the routed experts past the vector kernel's limit
(4 tokens for IQ3_S, 6 for IQ4_NL) took MMQ, which stages a tile per expert for the one or two tokens an
expert gets. Both now run the vector kernel over chunks of the batch: F32 weights up to 32 columns
(`STRIX_F32_VEC_CHUNK_MAX`), the experts in chunks of 4 tokens (`STRIX_MOE_VEC_CHUNK`, `=0` off) up to 16
(`STRIX_MOE_VEC_CHUNK_MAX_T`). A nine-token verify step went from 120-123 to 103-104 ms. At ~4K tokens
each, four conversations drafting 2 then got 62.6 tok/s summed against 55.7 without drafts (at ~20K
tokens each 56.1 against 49.8-51.4), while six and eight still lose with drafts (60.7 against 64.4, 64.0
against 70.7). Against 0.2.2, with the server's default sampling: three conversations 47.7 → 55.7 tok/s,
four 55.4 → 60.4, one unchanged at ~38. Measured in `docs/results/multi-stream-20260926.json`.

Since 0.2.4 three more of the step's products leave kernels that ran a few workgroups for the whole GPU.
The GDN conv input's concat of 2-31 transposed tokens takes the tiled transpose (`STRIX_CONCAT_T_MIN`;
from 32 before), and at 9-32 columns BF16 weights (the indexer's k projection) and quantized weights of
at most 1024 output rows (the hyper-connection down projection, the attention k and v, the shared
expert) run the vector kernel over chunks of 8 columns (`STRIX_BF16_VEC_CHUNK_MAX`,
`STRIX_Q_VEC_CHUNK_MAX`, `STRIX_Q_VEC_CHUNK_ROWS`). The server's own timing of a nine-token verify step
went from 106.4 to 103.0 ms, a twelve-token one from 129.4 to 123.5. With the server's default sampling
a round's tok/s follows the draft acceptance of the text it happens to sample (54-86% over these rounds),
so the builds are compared at the same acceptance: three conversations 53.8 → 55.0 tok/s summed (+2%),
four 58.1 → 60.7 (+4.5%). The same fixed seed per conversation made 0.2.3's gain over 0.2.2 read +17%
at three conversations; at the same acceptance it was +13%. Measured in
`docs/results/verify-small-products-20260926.json`.

Since 0.2.1 the drafts of one step are also the same length for every conversation
(`STRIX_SPEC_EVEN_DRAFTS`; `=0` drafts as before). The drafter stops a conversation's draft at its first
unconfident token, so two conversations could draft 2 and 1 tokens, and the model's hybrid memory runs a
batch as ubatches with the same number of tokens for every sequence: the verify became two or three
passes of the whole model, each a shape the HIP graph cache had not seen - 200-440 ms where an even
batch takes ~120. Now a conversation that turns unconfident keeps drafting while another is still
confident, and all are cut to the longest confident run. Three conversations at 4K tokens each went
from 34.3 tok/s summed with drafting (43.9 without) to 45.7; at 20K from 39.6 to 45.7; two
conversations, which had swung between 26 and 41, give 39-42. Measured in
`docs/results/multi-stream-20260925.json`.

### Freed cache cells are zeroed

A cell of the KV cache that a conversation gives up - evicted to disk, rewound, a draft token the next
step rejected, a run moved by the pool's re-layout - is zeroed as it is freed, by copies queued on the
stream the graphs run on (`STRIX_KV_ZERO_FREED=0` leaves it as it was). The attention of the next
batch reads past a conversation's last cell up to its block boundary with a weight of exactly zero, but
the matrix cores' sums still moved in the last bit with what those cells held, so a result depended on
who had used the cells before. It costs about 0.5% of decode with MTP (greedy A/B, one conversation);
with MTP off a fresh conversation frees nothing, and its output is bit for bit 0.1.16's.

### The draft head's carried row

The MTP draft head reads each token together with the target's hidden state at the position before
it, and the last of a batch's states is carried over to the next batch. After a rewind to a checkpoint
(a regenerate, a template that drops the previous turn's thinking), a conversation brought back from the
prompt cache or the disk tier, or another conversation taking the slot, the carried state was the one
the slot had computed last - another position, or another conversation - and the draft cache's first row
after it was computed from that. It never changed what the target generates, only which tokens were
drafted, but the same prompt sent twice drafted differently the second time, the verify batches took
different shapes, and greedy output parted where two tokens were nearly tied (token 133 of 400, at 85K
tokens). Since 0.1.17 the state is kept with its position, stored with checkpoints and cached
conversations, and zeros when it does not belong; the same prompt twice gives the same 400 tokens and
the same acceptance. Conversations the disk tier stored before 0.1.17 have no such state: they come
back as before, and the first draft after them starts from zeros.

### Output against 0.1.17

0.2.0's prefill kernels add in a different order in four places: the hyper-connection inject's partial
sums, the gated delta net's decay-scaled state, and the small F32 products (two kernels). The accuracy is
the same - perplexity at 8K context 2.6811 against 0.1.17's 2.6841 - but this model amplifies a last-bit
difference, so an answer can part from 0.1.17's where two tokens were nearly tied. Setting
`STRIX_HC_INJECT_FUSE=0 STRIX_GDN_R16=0 STRIX_SKINNY_F32=0 STRIX_MMB_F32_MIN_T=512` in the server's
environment gives 0.1.17's output bit for bit, at 0.1.17's prefill speed for those kernels. Measured in
`docs/results/prefill-kernels-20260925.json`.

### Output against 0.2.1

0.2.2 takes two kinds of small product off hipBLAS, whose GEMM kernels load from disk the first time
it meets a shape: the sparse-attention indexer's BF16 projections in batches of 9 to 511 tokens now run
on the MMB kernel, and Q6_K weights (the draft head's, and the output projection's) on MMQ at every
width. Such a batch - the new part of most chat turns - is summed in another order, so an answer can
part from 0.2.1's where two tokens were nearly tied; a prompt that arrives as one batch of 512 tokens
or more gives 0.2.1's output bit for bit. The perplexity tool asks the output projection for 8192 rows at
a time, which now take MMQ with its quantized activations: perplexity at 8K context 2.6814 against
2.6811 (the server asks for at most 16 rows, which MMQ already took). `STRIX_MMB_BF16_MIN_T=512
STRIX_MMQ_Q6K_ANY=0` in the server's environment gives 0.2.1's routing back.

### Output against 0.2.2

0.2.3 runs the MoE router and the routed experts of a batch of 5-32 tokens on the vector kernel, over
chunks of the batch. Such a batch - the verify step of several conversations, a short chat turn, the
last few tokens of a prompt - is summed in another order, so an answer can part from 0.2.2's where two
tokens were nearly tied. One conversation's decode (at most four tokens a step) and batches of more than
32 tokens are unchanged: an 18.6K-token prompt gives 0.2.2's output bit for bit, and perplexity at 8K
context is 2.6814 on both. `STRIX_F32_VEC_CHUNK_MAX=0 STRIX_MOE_VEC_CHUNK=0` in the server's
environment gives 0.2.2's routing back.

### Output against 0.2.3

0.2.4 runs BF16 weights, and quantized weights of at most 1024 output rows, on the vector kernel in
batches of 9-32 tokens. Such a batch - the verify step of three or four conversations, a short chat
turn, the last few tokens of a prompt - is summed in another order, so an answer can part from 0.2.3's
where two tokens were nearly tied. The decode of one or two conversations (at most eight tokens a step)
and batches of more than 32 tokens are unchanged, the new concat path being a copy either way: an
18.6K-token prompt gives 0.2.3's output bit for bit, and perplexity at 8K context is 2.6814 on both.
`STRIX_BF16_VEC_CHUNK_MAX=0 STRIX_Q_VEC_CHUNK_MAX=0` in the server's environment gives 0.2.3's
routing back, and `STRIX_CONCAT_T_MIN=32` its concat kernel.

### Shared GPU memory is the display driver's decision

Nothing here sets it. Up to 0.1.14 the manager set `GGML_HIP_ENABLE_UNIFIED_MEMORY` and, after a
load that ran out of memory, loaded again with it on "in shared memory". No code in the runtime
reads that variable - ggml reads only `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, and takes any value of it,
`0` included, as on; the HIP runtime reads neither (checked in the source and in every DLL the
release ships) - so the second load was the first one again. 0.1.15 removes both, and a profile
that still has `"shared_vram"` loads as if it did not.

Where an allocation lands is the driver's choice: in the carve while it has room, otherwise in
shared GPU memory, which is system RAM (Windows lets the GPU use up to half of it). At a 64 GB carve
~9.8 GB of the model goes there and decode is 28% slower ([results.md](results.md)). The driver can
also put allocations there while the carve still shows free space, and then they take system RAM
directly: GitHub issue #1 saw shared memory climb to its 15.8 GB limit on 0.1.8, where on this
machine the same growth stayed in the carve. Either way every allocation counts against commit,
the limit described above. The dedicated size the page shows is read from the display driver's
registry entry, so it costs nothing and needs no GPU context.

### Thinking depth

`thinking` takes `off`, `low`, `medium` or `high`, and Model › Configuration offers the same four.
They are not invented here — this model's chat template implements them, and the manager maps them
onto what it accepts:

| setting | sent to the template | what the model is told |
|---|---|---|
| `off` | `enable_thinking: false` | no reasoning step at all |
| `low` | `reasoning_effort: low` | keep thinking brief, go straight to the conclusion |
| `medium` | `reasoning_effort: medium` | nothing is injected: the model's own default |
| `high` | `reasoning_effort: xhigh` | think carefully, validate assumptions, weigh alternatives |

The template accepts **only** `low`, `medium` and `xhigh`, raising on anything else, and folds
`high` into `xhigh` itself. So four is the number of levels this model really has: a fifth would be
a second name for one of these. A single request can still override the profile with
`chat_template_kwargs`.

Optional — the desktop app: Jan v0.8.4 with the model pages (library, configuration, logs), a
welcome screen in place of Jan's setup, our name and no engine of its own:

```bash
python integrations/jan/apply.py <path to a Jan v0.8.4 checkout>
```

If the checkout was moved after `yarn install` (yarn links workspace packages with absolute
junctions), run `python tools/relink_jan_tree.py` first. Then, in that checkout, `yarn workspace @janhq/assistant-extension build`, `npx vite build` inside
`web-app/` once (it regenerates the route tree that `tsc` checks, which does not yet know the new
pages), and `yarn build:web && yarn tauri build`. If `dist/runtime` exists (section "The short
way"), apply.py stages it as a resource and the installer carries it.

## 7. Check it works

```bash
python tools/vision_probe.py                      # one image, end to end
python tools/vision_stress.py                     # 7 turns + 2 branches, the shape of three aborts
python tools/truncation_test.py --fill 70000      # 30 sections at depth; anything under 30/30 is a bug
python tools/decode_lab.py --keep --words 700,28800 --n 64
```

`vision_probe` and `vision_stress` generate their own image, so they need nothing but a running
server with a projector loaded.

## Known traps

- **`LLAMA_MMB_HC16` must be 0.** With it on, this compiler and SDK flood the output with `/`. The
  manager sets it; if you launch `llama-server` yourself, you need it too.
- **Do not measure with `llama-bench`.** It has no `--load-mode none`, so at this carve it maps
  93.7 GB through a 31.6 GB page cache and thrashes the page file. Measure through the server.
- **Do not measure decode on a server that has served an image** — it runs ~7% slower until
  restarted. See [measuring.md](measuring.md), which lists the rest.
- **GPU "Dedicated Usage" is not an allocation figure** under the flags the manager passes. Weights
  page in on demand, so it tracks how much context has been processed.
