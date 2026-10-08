# Rulith Inference

Serving a 125B mixture-of-experts model fast on one AMD Strix Halo machine, on Windows.

Target: **Ryzen AI Max+ 395** (Radeon 8060S, gfx1151, 128 GB unified memory) running
**Qwen3.8-Flash-Next** (125B-A6B) through a patched llama.cpp: Unsloth's UD-IQ4_XS (93.7 GB) is the reference file,
and UD-Q4_K_XL (111.3 GB) is tuned too.

*[中文说明](README.zh.md)*

*Formerly **Strix Llama**, renamed in 0.3.6 so as not to be confused with the community fork
[halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp), which had the name first. Same app,
same data: an installed Strix Llama updates in place.*

## Where it stands

Measured on the target machine with 0.5.1 (2026-10-08, 64 GB VRAM carve), speculative decoding on, at the
app's default settings (eight conversation slots, MTP, images on): prefill over 95.6K tokens of real text;
400 tokens decoded after 86K tokens of that text, and after a one-line question; and at four slots, several
conversations of ~4K tokens each decoding together, with a new sampling seed every round. The last row is
MTP off. In brackets, 0.4.0 measured the same way (2026-10-01, 96 GB carve):

| | |
| --- | --- |
| prefill | **1328 t/s** (0.4.0: 1237) |
| decode, 86K context | **25.9 ms/token** (38.6 tok/s at 60% draft acceptance; 0.4.0: 23.9 at 65%, the same ~70 ms a pass) |
| decode, short context | **20.7 ms/token** (48.2 tok/s at 67% acceptance; 0.4.0: 21.7 at 65%) |
| decode, 3 / 4 conversations at once | **66.9 / 75.1 tok/s** summed (1.59× / 1.78× one conversation measured the same way, 42.1; 0.4.0: 59.8 / 69.1 and 42.1; the sum moves ±10% with the sampled rounds' acceptance) |
| decode with MTP off, 8 conversations of ~40K tokens at once | **95.0 tok/s** summed in a 512K-token q8_0 pool (0.4.0: 91.0); eight of ~20K with the f16 cache 97.7 (0.4.0: 95.6); one conversation 27.8 / 27.5 / 27.0 tok/s at 3K / 50K / 110K (0.4.0: 27.5 / 27.0 / 26.5) |
| image input | supported (Qwen3-VL projector) |

Every number in this repository comes with the command that produced it, in
[docs/results.md](docs/results.md). Where a change could not be resolved above the noise floor, it
says so rather than claiming a win — and decode figures carry their draft acceptance, because with
speculation on, throughput without acceptance describes the prompt rather than the runtime.

**On Linux.** There is no Linux installer, but the patched llama.cpp builds there. A user built the
patched branch of [rulith-dev/llama.cpp](https://github.com/rulith-dev/llama.cpp) (then `strixllama`, `rulith` since 0.4.3) with ROCm on Linux
and measured it with `llama-bench` (`-b 8192 -ub 8192 -fa 1`, with the app's switches exported):
UD-IQ4_XS 1254 t/s prefill (pp2048) and 27.8 tok/s decode (tg128, MTP off), the same as on Windows;
UD-Q4_K_XL 804 and 26.0 ([thread](https://www.reddit.com/r/StrixHalo/comments/1wv2x6c/comment/pdgodkl/)): the tuned expert kernels covered only the quant types inside
UD-IQ4_XS. Since 0.4.2 they cover UD-Q4_K_XL's as well: its prefill here went 876 → 1127 t/s on a 2K-token prompt and
850 → 1054 on 95.6K; its decode was already at what its 8.5% larger reads allow. Other quantizations of the model
load and answer, but run on llama.cpp's stock kernels and prefill correspondingly slower: the tuned kernels target
these two files ([#8](https://github.com/rulith-dev/rulith-inference/issues/8)). The switches are
`HIP_GATES` and `HIP_QSA_GATES` in [tools/manager.py](tools/manager.py); without them the server runs
close to stock llama.cpp.

## Just want to run it?

**[docs/getting-started.md](docs/getting-started.md)** — the installer from the
[releases page](https://github.com/rulith-dev/rulith-inference/releases), the five model files to download
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
| **Draft head**, ours: picks 512 candidates with a rank-512 pre-score and computes only their rows of the model's own output projection; +3.6-4.9% decode over the IQ4_XS head of 0.1.2 | [`mtp-Qwen3.8-Flash-Next-head-lr512.gguf`](https://github.com/rulith-dev/rulith-inference/releases/download/v0.4.6/mtp-Qwen3.8-Flash-Next-head-lr512.gguf) | 149 MB |

All four go flat into one folder: the app finds the draft, the projector and the head by name next to
the model's first shard. [docs/getting-started.md](docs/getting-started.md) has the download commands.

## Quick start (from source)

```bash
python bootstrap/bootstrap.py --toolchain          # ROCm SDK + ninja into toolchain/, ~5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` clones `pwilkin/llama.cpp` at a pinned revision, applies the patch set, and builds
against the ROCm SDK. The same 85-file delta is also published on a fork, a commit per release, so it can be
read as a plain diff: [rulith-dev/llama.cpp, branch `rulith`](https://github.com/rulith-dev/llama.cpp/tree/rulith) (`strixllama` until 0.4.3, still kept in
step for a few releases). `tools/manager.py` is a JSON-on-stdin process manager: it owns the launch
flags, the environment gates and the runtime, so a configuration is reproducible rather than
remembered.

The server it starts is an OpenAI-compatible endpoint, by default at `http://127.0.0.1:8080/v1`. In the
app, Model › Configuration › Network changes the port, lets other devices on the local network use the
server (it then listens on 0.0.0.0, and Windows may ask the first time whether to let llama-server through
the firewall) and sets an API key that requests must send as `Authorization: Bearer <key>`. The setting
is kept in `config/jan/settings.json` (`"network": {"port": 8080, "lan": false, "api_key": ""}`), which
updates leave in place; `RULITH_PORT`, `RULITH_HOST` (an address to bind, such as `0.0.0.0`) and
`RULITH_API_KEY` in the environment override it. Requests name the model as its file does, without the shard
suffix (`"model": "Qwen3.8-Flash-Next-UD-IQ4_XS"`, the name `/v1/models` lists and the Logs page shows);
with one model loaded, the server answers any name.

**Read [docs/install.md](docs/install.md) first.** Three things there are not optional and not
obvious: the GPU carve should be 96 GB or, since 0.4.7, 64 GB (the experts that do not fit it are read
from system memory at full speed; 96 GB leaves the KV pool more room), the ROCm SDK must be TheRock 10.2 rather than the system 7.1 (+60%
prefill on identical source), and the SDK version this was measured on comes from a nightly index
with a ~27-day window — so the pin will stop resolving, and that page says what to do about it. The
model files are a separate download of about 95 GB.

Optional: `integrations/jan/apply.py` overlays the model pages (library, configuration, logs), a
welcome screen in place of Jan's setup screen, the model's state in the sidebar and above the chat,
and Rulith's design language onto a [Jan](https://github.com/menloresearch/jan) checkout, in English
and Chinese.

## What is actually in here

The whole delta against upstream llama.cpp is **85 files** — 81 modified, 4 added, out of 3610. The
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
| **The delta net's state, written once in nine tokens** | Three layers in four keep a 3 MB state per conversation, read and rewritten for every token: with eight conversations, 24 MB written a layer a token, which cost more than reading it. A token now writes only its 33 KB record, and the state goes back to memory once eight records are pending, from the same records and the same arithmetic; a warp also reads two of its columns at once. Eight conversations of ~20K tokens, MTP off: 89.4 → 97.1 tok/s (0.4.0); output identical bit for bit. |
| **Shorter prompt passes** | A prompt's pass seen on one continuous GPU timeline put the routed experts at ~40% of a 2K-token batch, and around them some work that needed no pass of its own. The shared expert's gated output is now added inside the routed experts' weighted sum; the hyper-connection streams are formed where the next projection reads them instead of being written out in F32; the delta net's q/k norms take one kernel; the experts' row lists are sorted from 64 tokens on, not only above 4096; and the per-layer embedding rows a prompt chunk needs are read from the SSD before its pass asks for them. Fresh prompts of 1K / 2K / 4K / 8K / 16K tokens: 920 / 1072 / 1122 / 1267 / 1245 → 964 / 1148 / 1227 / 1302 / 1301 t/s (0.4.3); output identical bit for bit. |
| **UD-Q4_K_XL's experts on the matrix cores** | The tuned prefill kernels were written for the quant types inside UD-IQ4_XS, so UD-Q4_K_XL's experts (Q4_K, Q5_1) fell back to llama.cpp's generic path. They have their own now: Q4_K_XL prefill 876 → 1127 t/s on 2K tokens, 850 → 1054 on 95.6K (0.4.2). Perplexity unchanged within noise; UD-IQ4_XS untouched. |
| **Small decode kernels and BF16 copies** | A profile from timestamps captured inside the HIP graph showed ~3 ms a token in small kernels bound by latency: fused elementwise chains that fetched unrolled general indexing code at every launch (8.4 → 2.7 us), a top-k without DPP. And 264 F32 weights - routers, gates, injects - hold only bfloat16 values; BF16 copies give the same products from half the bytes. One conversation, MTP off, 25.9 → 27.3 tok/s at 3K (0.4.0), the same output. |
| **rocBLAS's own kernels in the runtime** | Issue #11: the runtime carried rocBLAS's Tensile kernels but not the kernel pack (`bin/.kpack/blas_lib_gfx1151.kpack`, 5 MB) that its own kernels come from in this ROCm SDK. Every hipBLAS product of one column with f32 accumulation (f32 in, or bf16 / f16 in with f32 compute) therefore failed with `hipErrorInvalidKernelFile` and stopped the server. Graphs built from the Unsloth files never send such a product to hipBLAS; a GSQ-RCO model with a BF16 projector did. The bundle now ships every pack its DLLs name. On the bundled runtime, ggml's MUL_MAT tests used to stop at the first such product and now pass, and output is bitwise unchanged (0.5.0). |
| **Waiting for the GPU without the runtime's lag** | On Windows, hipStreamSynchronize returns ~0.65 ms after a 2,000-kernel graph has finished, and the last batch of a launch is only submitted once the host synchronizes ([ROCm/TheRock#8786](https://github.com/ROCm/TheRock/issues/8786), with a standalone repro). Now the end of the stream writes a number into host memory, an event query submits the work, and the host polls the number. Past 100 ms (prefill) the runtime's own wait takes over. One conversation: -0.6 ms a token without MTP (-1.7%), +1.9% with MTP; eight: -1.0 to -1.7 ms a step. Output is bitwise unchanged. The poll costs ~0.4 of a CPU core while decoding; `STRIX_POLL_SYNC=0` turns it off (0.4.9). |
| **Prefill after a long conversation** | A new turn first saved a checkpoint of the delta net's state, and that meant replaying the recent tokens' pending updates on the CPU: ~109 ms at every turn start. The GPU replays them now with the kernel the forward pass uses (~3 ms). Three fusions (the router product, the delta net's beta / alpha pair, the attention gate with its norm), expert kernels whose barriers no longer wait for every load, and the six expert layers of UD-IQ4_XS that still ran on llama.cpp's generic path (five Q8_0 down projections, one IQ4_XS gate/up) moved to the tuned kernels. In the app (UD-IQ4_XS, MTP off, two slots) a 156K-token prompt takes ~115 s, 1,359 t/s. Against 0.5.0 a message is processed ~3% faster at the start of a conversation and ~7% faster after 4K-32K tokens of history. Those six layers now compute in bf16 like the other 90, so output is no longer bit for bit 0.5.0's; perplexity over 40 chunks 2.8134 → 2.8070. `STRIX_MMB_Q8=0` restores them (0.5.1). |
| **Prompts cut only where a turn begins** | A conversation read fresh (after a restart, or evicted with the disk tier off) was cut into batches at its first user message and at turn starts spaced through its history: ten short ubatches on 151K tokens of 35 turns. Its history is one piece of context now. Only the last user message, where an edit of it forks, starts a batch of its own; the other checkpoints are taken where a batch ends anyway, spaced as before. The cuts had cost nothing measurable (122.6 s for that prompt with or without them). What a user notices is the eviction fix that came with this: the checkpoint at the last user message no longer goes first when a long prompt fills the list, so an edit of that message computes 1,881 tokens instead of 5,895 (0.4.8). |
| **An expert placement that counts the whole load** | 0.4.7's estimate missed the part of the compute buffer that grows with the KV pool (~6.7 KiB a cell), counted the MTP draft's K/V at the target's type (they are f16) and took the whole carve as usable (the driver keeps ~4%). On a 512K q8_0 pool with MTP at the 64 GB carve it kept 15 layers' experts in system memory where 18 were needed, ~2 GB spilled, and a 156K-token prompt ran at 1,089 t/s; it runs at 1,237 now. With 32 layers' experts (37 GiB) there, the GPU reports no free memory and llama.cpp's layer split by free memory divided 0 by 0, so the MTP draft failed to load; an all-zero split is even now. At 32 layers prefill holds (1,233.5 t/s) and MTP decode is ~2% slower than at 18 (0.4.8). |
| **Experts in system memory** | A carve smaller than the load (the 64 GB setting, or a large KV pool at 96 GB) left the overflow to the display driver, which put whatever was allocated last in shared memory: the KV cache and compute buffers, read in small pieces every token. 64 GB was 28% slower; a user's 96 GB pool with 8 GB spilled lost 10-12% at long context. The app now sums what a load needs and keeps the experts of the last layers that do not fit in pinned system memory, which the GPU reads as fast as the carve. At 64 GB with 16 layers' experts there: prefill 1343-1368 t/s and decode 28.5 tok/s without MTP, 43.7 / 43.1 tok/s greedy / sampled with it, the same as 96 GB; the f16 / MTP / q8_0 probes give 0.4.4's bits (0.4.7). |
| **Image batches independent of other conversations** | Issue #10: the bound that keeps image batches' dense inputs within memory counted the whole KV pool, so another slot's long conversation split this one's image differently and changed its greedy answer. It counts the conversation's own cells now (fix by @obsidience; 0.4.7). |
| **A cheaper draft head** | A draft step read a whole output projection to pick one token: 338 MB with the IQ4_XS head, whose first choice matched the model's own 96.5-97% of the time. A rank-512 pre-score of the vocabulary (135 MB) now picks 512 candidates, and only their rows of the model's own projection are multiplied; the model's first choice is among them 99.9-100% of the time, on English, Chinese and conversations the pre-score was not fitted on. Decode +3.6% greedy and +4.9% sampled (acceptance 0.687 → 0.702) over the IQ4_XS head, greedy output unchanged bit for bit (0.4.6). |
| **MTP when sampling** | With a temperature above 0, a draft token counted only when the model drew that very token. The draft head now draws its proposals from its own distribution, shaped by the request's sampling settings, and speculative sampling accepts each with the probability that keeps every token distributed exactly as the model alone samples it: at temperature 0.7 the first proposal is kept 67% of the time instead of 63%. Proposals kept less often pay for fewer a step, so sampled answers draft 2 tokens for one or two conversations, 1 for three or four. At Jan's settings, one conversation 33.6 → 36.2 tok/s, three 61.3 → 67.1 (0.4.1). Greedy output unchanged bit for bit. |
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
python tools/replay_bootstrap.py          # clean upstream + patch set == the 85 files, byte for byte
```

It restores the 81 modified files to upstream from the clone's own git objects, addressed by the blob
hashes in `bootstrap/UPSTREAM.json` — so clean upstream is reconstructed rather than trusted — then
replays the snapshot and every script and compares. It reports **85 / 85**.

It was not always so. Five of the 24 were owned by no script at all - including the largest measured
win in the project, which a clean rebuild would have silently dropped - and three scripts had drifted
until they no longer applied. `tools/make_patch_script.py` generates a patch script from two trees,
growing each anchor until it is unique, which is how those were closed. Hand-written anchors are what
rotted in the first place.

## Scope and limits

- **Windows only.** The manager uses Win32 process APIs; the build targets gfx1151. The patched server
  alone also builds on Linux; that was measured by a user, not here (see "On Linux" above).
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

If these patches are useful in your project, please credit Rulith Inference with a link to
[this repository](https://github.com/rulith-dev/rulith-inference). That is a request, not a condition of the licence.

---

Rulith Inference is made by [Rulith](https://rulith.ai): verifiable execution infrastructure for AI agents.
