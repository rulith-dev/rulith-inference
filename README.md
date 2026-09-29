# Strix Llama

Serving a 125B mixture-of-experts model fast on one AMD Strix Halo machine, on Windows.

Target: **Ryzen AI Max+ 395** (Radeon 8060S, gfx1151, 128 GB unified memory) running
**Qwen3.8-Flash-Next** (125B-A6B, Unsloth UD-IQ4_XS, 93.7 GB) through a patched llama.cpp.

*[中文说明](README.zh.md)*

## Where it stands

Measured on the target machine with 0.3.1 (2026-09-28), speculative decoding on, at the app's default
settings (eight conversation slots, MTP, images on): prefill over 95.6K tokens of real text; 400 tokens
decoded after 86K tokens of that text, and after a one-line question; and at four slots, several
conversations of ~4K tokens each decoding together, with a new sampling seed every round:

| | |
| --- | --- |
| prefill | **1228 t/s** |
| decode, 86K context | **27.3 ms/token** (36.7 tok/s at 63% draft acceptance) |
| decode, short context | **22.5 ms/token** (44.4 tok/s at 67% acceptance) |
| decode, 3 / 4 conversations at once | **55.4 / 62.5 tok/s** summed (1.40× / 1.57× one conversation measured the same way, 39.7; per step the same as 0.3.0, the sum moves ±10% with the sampled rounds' acceptance) |
| decode with MTP off, 8 conversations of ~40K tokens at once | **77.9 tok/s** summed in a 512K-token q8_0 pool (0.3.5; 0.3.4: 75.8 in the same session, 0.3.3: 58.8), 77.7 with typical sampling settings, and one conversation at 110K 24.6 tok/s (0.3.3: 21.6) |
| image input | supported (Qwen3-VL projector) |

Every number in this repository comes with the command that produced it, in
[docs/results.md](docs/results.md). Where a change could not be resolved above the noise floor, it
says so rather than claiming a win — and decode figures carry their draft acceptance, because with
speculation on, throughput without acceptance describes the prompt rather than the runtime.

## Just want to run it?

**[docs/getting-started.md](docs/getting-started.md)** — the installer from the
[releases page](https://github.com/rulith-dev/strixllama/releases), the five model files to download
and where to put them, and what to press. No Python, ROCm or build tools involved.

## Model files

Every number here was measured with these files. The links are pinned to the Hugging Face revision
they were checked against (`38bb39e`: the sha256 of each file compared with the repository's on
2026-09-26):

| | File | Size |
| --- | --- | --- |
| **Model**, required | Unsloth's Qwen3.8-Flash-Next-GGUF, [`UD-IQ4_XS`, three shards](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS) | 93.7 GB |
| **MTP draft**: speculative decoding, ~+60% decode | [`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf) | 1.9 GB |
| **Vision projector**: image input | [`mmproj-F16.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf) | 904 MB |
| **Draft head**, ours: the draft's own IQ4_XS output projection, +5-9% decode | [`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`](https://github.com/rulith-dev/strixllama/releases/download/v0.1.2/mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf) | 349 MB |

All four go flat into one folder: the app finds the draft, the projector and the head by name next to
the model's first shard. [docs/getting-started.md](docs/getting-started.md) has the download commands.

## Quick start (from source)

```bash
python bootstrap/bootstrap.py --toolchain          # ROCm SDK + ninja into toolchain/, ~5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` clones `pwilkin/llama.cpp` at a pinned revision, applies the patch set, and builds
against the ROCm SDK. The same 53-file delta is also published as one commit on a fork, so it can be
read as a plain diff: [rulith-dev/llama.cpp, branch `strixllama`](https://github.com/rulith-dev/llama.cpp/tree/strixllama). `tools/manager.py` is a JSON-on-stdin process manager: it owns the launch
flags, the environment gates and the runtime, so a configuration is reproducible rather than
remembered.

**Read [docs/install.md](docs/install.md) first.** Three things there are not optional and not
obvious: the GPU carve must be 96 GB (at 64 GB the model does not fit and decode is 28% slower, which
no software setting recovers), the ROCm SDK must be TheRock 10.2 rather than the system 7.1 (+60%
prefill on identical source), and the SDK version this was measured on comes from a nightly index
with a ~27-day window — so the pin will stop resolving, and that page says what to do about it. The
model files are a separate download of about 95 GB.

Optional: `integrations/jan/apply.py` overlays the model pages (library, configuration, logs), a
welcome screen in place of Jan's setup screen, the model's state in the sidebar and above the chat,
and Rulith's design language onto a [Jan](https://github.com/menloresearch/jan) checkout, in English
and Chinese.

## What is actually in here

The whole delta against upstream llama.cpp is **53 files** — 49 modified, 4 added, out of 3610. The
substantial pieces:

| | |
| --- | --- |
| **IQ3_S matrix-core path** | 52% of this model's body is stored as IQ3_S, which the MMB dequant GEMM did not accept, so half the weights never reached the matrix cores. +5.6% prefill. |
| **Expert kernels priced for this GPU** | On gfx1151 the vector units and the matrix cores never run at the same time, so every instruction a dequantizing GEMM spends unpacking weights is matrix time lost. The routed IQ3_S gate/up, prefill's largest kernel, rebuilt around that: 35.9 → 19.9 ms a layer, prefill +10%, bitwise the same output. |
| **The rest of a prefill batch** | With the experts at their limit, a quarter of a batch's GPU time went to work that should have been cheap: a 4-output projection through a 64-row GEMM tile, activation rows all on the same memory channels, a many-pass indexer score, two full copies of the cache per layer for sparse attention. Each is now fused or rewritten: GPU time 1860 → 1573 ms for a 2K-token batch, 2276 → 1788 ms at 64K depth; a 95.6K-token prompt 992 → 1187 t/s. The same perplexity, no longer bit for bit. |
| **Sparse attention at decode** | The sparse kernel needs a packed layout only prefill builds, so a decoded token attended densely over the whole cache. Gathering the selected cells instead: 97K decode 59.0 → 49.3 ms/token, and the context slope drops from 0.188 to 0.067 ms per 1000 tokens. |
| **A Q8_0 K/V cache, optional** | The sparse-attention kernels read f16 only, so a Q8_0 cache is dequantized into their layout each batch: at 262144 tokens the K/V cache takes 3.2 GB instead of 6, decode is unchanged and prefill 1-2% slower. f16 stays the default. |
| **MTP speculation, tuned** | Draft head with its own IQ4_XS output projection, three draft tokens, n-gram drafting off. 85K decode 21.5 → 34.9 tok/s. |
| **Drafts sized by concurrency** | In this mixture of experts every verified draft token reads the weights of ~10 more experts, which conversations decoding together cannot share. The draft shrinks as more conversations generate (3, then 2 up to four conversations, none from five on). The drafts of one step are the same length: the hybrid memory ran a verify of uneven drafts as several passes of the whole model, so three conversations at 4K tokens got 34 tok/s with drafting and 46 without; now 46 with it. And a verify step of several conversations (5-16 tokens) runs its small products on the vector kernel, not on tiles they barely filled: since 0.2.3 the MoE router and the routed experts (a nine-token step 120-123 → 103-104 ms; three conversations +13% at the same draft acceptance, four, which now draft, +8%), since 0.2.4 three more (106 → 103 ms; +2% and +4.5%). |
| **Output that does not depend on what ran before** | Freed KV cells are zeroed: the sparse attention's matrix-core sums moved in the last bit with what an earlier conversation or a rejected draft had left in them. The MTP drafter's carried state follows a conversation through rewinds and the prompt cache: the same prompt sent twice drafts, and answers, the same. |
| **A prefix cache that follows agents** | The recurrent state can be resumed only where a checkpoint was kept, so checkpoints now sit where later prompts branch: where the system prompt ends, at each user message, at the end of each prompt. A new session on a system prompt another slot already computed copies it on the GPU instead of computing it again, and sessions that start together wait for the first one's copy. An agent workload (10K-token system prompt, three sessions at once, three sub-agents): 73K tokens processed → 29K, prompt time 172 → 60 s. |
| **Several agents at once** | Agent tools run a session per sub-agent on one long system prompt, so every slot looked 97% similar to every new session, and a new session took whichever slot scored highest, cutting another agent's history. A slot now takes a request for what it holds only when the request continues it, and eight conversations stay loaded by default. Conversations that are answering share one pass whatever their lengths, a prompt gets a pass of its own, and while conversations answer a step takes at most 2048 prompt tokens. Six agents, 0.2.6 at its four slots against 0.2.7 at its eight: 75K tokens processed → 45K, median time to the first token 10.1 → 3.4 s, the longest pause of a streaming agent 6.2 → 2.7 s; a 19K-token prompt pauses a streaming conversation for 2.2 s at a time instead of 7. |
| **Rollback without snapshots** | A speculative verify saved the recurrent state after every checked token, 3 MB a layer, in case a draft was rejected. It now records only what it needs to recompute a rejected tail (33 KB a token) and replays it in the next step: a verify step at three / four conversations 2.8% / 4.0% shorter, output identical bit for bit. |
| **Three correctness fixes** | Speculative verification batches ran dense attention with no causal mask, so long answers drifted and stopped early. Image input aborted the server three separate ways in the QSA block machinery, and a fourth when a second loaded conversation got an image. |
| **Measurement instrumentation** | Per-graph, per-dispatch and per-phase timing, all off unless an environment variable is set. |

`patches/MANIFEST.md` lists every patch, what it touches, and the order they must run in.

## How the build is verified

A patch set that cannot be replayed is a patch set that quietly rots. Two tools keep this honest:

```bash
python bootstrap/bootstrap.py --record    # hash what the patch set produces
python bootstrap/bootstrap.py --verify    # is this tree still exactly that?
```

`--verify` answers the weak question — is this tree what the recipe produced? It passes just as
happily if the tree was edited by hand and re-recorded afterwards. The strong question is whether the
recipe still rebuilds the tree from nothing, and that has its own tool:

```bash
python tools/replay_bootstrap.py          # clean upstream + patch set == the 53 files, byte for byte
```

It restores the 47 modified files to upstream from the clone's own git objects, addressed by the blob
hashes in `bootstrap/UPSTREAM.json` — so clean upstream is reconstructed rather than trusted — then
replays the snapshot and every script and compares. It reports **53 / 53**.

It was not always so. Five of the 24 were owned by no script at all - including the largest measured
win in the project, which a clean rebuild would have silently dropped - and three scripts had drifted
until they no longer applied. `tools/make_patch_script.py` generates a patch script from two trees,
growing each anchor until it is unique, which is how those were closed. Hand-written anchors are what
rotted in the first place.

## Scope and limits

- **Windows only.** The manager uses Win32 process APIs; the build targets gfx1151.
- **One model family.** The attention work is specific to `qwen4exp`. The kernel work (IQ3_S MMB,
  MMVQ rows-per-block) applies more broadly.
- **Not a llama.cpp fork.** This repository holds patches, tools and measurements. No model weights,
  no upstream source, no binaries.

## Layout

```
bootstrap/   fetch, patch and build the runtime; UPSTREAM.json is the delta's hash record
patches/     the patch set, its manifest, and two whole-file drops too large to anchor
tools/       manager, measurement harnesses, verification
integrations/jan/   optional management pages for a Jan checkout
docs/        install.md · results.md · measuring.md · decode-budget.md · dead-ends.md
```

[docs/measuring.md](docs/measuring.md) is the one to read before trusting any number here, including
the ones above — it is a list of the ways this project measured itself wrong.

## Licence

MIT, see [LICENSE](LICENSE). Third-party attribution in [NOTICE.md](NOTICE.md) — in particular this
patches [pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp) (MIT) and the Jan overlay is meant
for a [Jan](https://github.com/menloresearch/jan) checkout (Apache-2.0).

---

Strix Llama is made by [Rulith](https://rulith.ai): verifiable execution infrastructure for AI agents.
