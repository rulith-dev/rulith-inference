# Rulith Inference

A 125B mixture-of-experts model, served fast on one AMD Strix Halo PC, on Windows.

Rulith Inference is a desktop app and a local server for **Qwen3.8-Flash-Next** (125B parameters, 6B active per token)
on the **Ryzen AI Max+ 395** (Radeon 8060S, gfx1151, 128 GB unified memory). One installer carries everything; the
app downloads the model itself and serves it to its own chat window and to any OpenAI- or Anthropic-compatible
client, coding agents included. Underneath is llama.cpp with 181 files changed for this GPU and this model.

*[中文说明](README.zh.md)* · *formerly Strix Llama*

## Speed

Rulith Inference 0.6.9 on the target machine at the app's default settings (64 GB GPU memory setting, eight
conversation slots, MTP speculative decoding), with the recommended EXL3 4.05 bpw file:

| | |
| --- | --- |
| reading a prompt | **1,300-1,330 t/s** on 16.9K-86K tokens |
| one conversation, MTP on | **58 tok/s** explaining (draft acceptance 66%), **70 tok/s** writing code (84%), **46 tok/s** prose (45%) |
| one conversation, MTP off | **33 tok/s** |
| after 86K tokens of context | **46 tok/s** (MTP on, acceptance 59%) |
| eight conversations at once | **85 tok/s** summed (91 with MTP off) |

With speculation on, speed depends on how predictable the text is, so every MTP figure carries its draft
acceptance (accepted draft tokens / proposed). Each number in this repository comes with the command that produced
it ([docs/results.md](docs/results.md)); [docs/measuring.md](docs/measuring.md) lists the ways this project once
measured itself wrong.

## Get started

1. **Install.** Download `Rulith-Inference_<version>_x64-setup.exe` from the
   [releases page](https://github.com/rulith-dev/rulith-inference/releases) and run it. No Python, ROCm or build tools:
   the runtime is inside. The app updates itself from then on, and only installs an update whose signature verifies.
2. **Set the GPU memory** in the BIOS to 64 GB or 96 GB (*UMA frame buffer size* or similar). At 64 GB the experts
   that do not fit are kept in system memory and read at the same speed.
3. **Download the model.** Open **Model › Download** and pick a file under *Recommended*; EXL3 4.05 bpw comes first.
   The app fetches the model, its MTP draft and its vision projector from Hugging Face (or its mirror), checks every
   file's SHA-256 and adds them to the Library. Files you already have, an LM Studio folder for
   example, are read in place: add their folder under **Model › Library › Model folders**.
4. **Load and chat.** Press **Load model**. The defaults are the configuration the numbers above were measured with;
   nothing has to be set first. Attach a picture to ask about it.

[docs/getting-started.md](docs/getting-started.md) walks through it with what to expect at each step and what to do
when a step fails.

## Models

Tuned and measured: Qwen3.8-Flash-Next in four files. Quality is the KL divergence to the BF16 model (lower is
closer) and top-1 agreement over six kinds of text; speed is one conversation writing code with MTP on (draft
acceptance in brackets), and reading a 16.9K-token prompt (0.6.8, 64 GB setting; one conversation runs the same code in
0.6.9).

| File | Size | KLD | top-1 | Code, MTP on | Prompt reading |
| --- | ---: | ---: | ---: | ---: | ---: |
| **EXL3 4.05 bpw** (recommended) | 104.3 GB | **0.022** | **96.0%** | 70.9 tok/s (84%) | 1,332 t/s |
| EXL3 3.05 bpw | 82.4 GB | 0.044 | 94.5% | 75.9 tok/s (81%) | 1,327 t/s |
| UD-Q4_K_XL | 111.3 GB | 0.024 | 95.9% | 58.8 tok/s (91%) | 1,379 t/s |
| UD-IQ4_XS | 93.7 GB | 0.048 | 94.4% | 63.6 tok/s (88%) | 1,414 t/s |

The EXL3 files are turboderp's quantization repacked for this engine:
[Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF](https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF), with
their own MTP draft; they load only in Rulith Inference. The UD files are
[Unsloth's](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). NVIDIA Nemotron 3.5 Super (120B-A12B, Q4_K_M)
loads too, with image input and its own MTP layer. Other quantizations of Qwen3.8-Flash-Next load and run on
llama.cpp's stock kernels.

## Use it from other programs

While a model is loaded the server answers at `http://127.0.0.1:8080/v1`: OpenAI's Chat Completions and Responses
APIs and Anthropic's Messages API, streaming and tool calls included. **Model › Configuration › Network** changes the
port, opens the server to the local network and sets an API key (`Authorization: Bearer <key>`). Requests name the
model as its file does without the shard suffix (`Qwen3.8-Flash-Next-EXL3-4.05bpw`); with one model loaded, any name
is answered.

Eight conversations stay loaded at once, and a slot takes a request only when the request continues what it holds, so
agent tools that run several sessions on one long system prompt do not cut each other's history. A system prompt
another session already read is copied on the GPU, not read again.

**Codex**, in `~/.codex/config.toml`:

```toml
model = "Qwen3.8-Flash-Next-EXL3-4.05bpw"
model_provider = "rulith"

[model_providers.rulith]
name = "Rulith Inference"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
# env_key = "RULITH_API_KEY"   # when the server has an API key
```

**Claude Code**: set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_AUTH_TOKEN` to the API key (any text when
there is none), and `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL` to the model's name. Hosted tools such as
`web_search` are not run; a response names them in an `X-Rulith-Unsupported-Tools` header.

## What is in the engine

The changes against upstream llama.cpp, by what they do. [docs/engine.md](docs/engine.md) has each one with its
measurement, release and switch.

- **Prompt reading.** Expert kernels written for how gfx1151 runs them (its vector units and matrix cores never work
  at the same time, so a kernel that spends instructions unpacking weights loses matrix time): IQ3_S, Q4_K and EXL3
  experts on the matrix cores, the delta net in 16-token chunks on the matrix cores, and the small work around the
  experts fused away.
- **Generation.** Decode reads the weights at the memory's speed: EXL3 weights stream through a ring in the GPU's local
  memory, the routed experts of a step run as one launch, sparse attention gathers only the selected cache cells, and
  the host waits for the GPU without the runtime's ~0.65 ms lag.
- **Speculative decoding.** The model's MTP layer drafts with Rulith's low-rank draft head; drafts are sized by
  acceptance and by how many conversations generate, and under sampling every token is distributed exactly as the
  model alone samples it.
- **Many conversations and agents.** One pass for every conversation that is answering whatever its length, prompt
  checkpoints where later prompts branch, a prompt cache that can be kept on disk, and a delta-net state written once
  in nine tokens.
- **Memory.** The app sums what a load needs and keeps the experts that do not fit the GPU setting in pinned system
  memory, read at full speed; a Q8_0 K/V cache is optional.
- **The same output for the same input.** Freed cache cells are zeroed and the draft's state follows a conversation
  through rewinds and the cache, so an answer does not depend on what ran before it.

## Build from source

```bash
python bootstrap/bootstrap.py --toolchain          # ROCm SDK + ninja into toolchain/, ~5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` clones upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at a pinned revision,
applies the patch set and builds against the ROCm SDK. The same delta is a plain diff on the fork:
[rulith-dev/llama.cpp, branch `rulith`](https://github.com/rulith-dev/llama.cpp/tree/rulith). `tools/manager.py` owns
the launch flags, the environment and the runtime, so a configuration is reproducible rather than remembered. Read
[docs/install.md](docs/install.md) first: the ROCm SDK must be TheRock's (+60% prefill on identical source), and its
nightly index keeps a version for only about four weeks.

`python tools/replay_bootstrap.py` rebuilds the 181 changed files from clean upstream and the recipe and compares them
byte for byte; it reports **181 / 181**. `patches/MANIFEST.md` records every change with its measurement, and
`integrations/jan/apply.py` builds the app's pages onto a [Jan](https://github.com/menloresearch/jan) checkout.

The patched server also builds on Linux with ROCm; a user measured it there at the same speed as on Windows
(UD-IQ4_XS). There is no Linux installer.

## Scope and limits

- **Windows 11 and gfx1151 only.** The bundled kernels are compiled for the Radeon 8060S; on another GPU the model does
  not load.
- **One model family tuned.** The attention and expert work is specific to Qwen3.8-Flash-Next (`qwen4exp`); Nemotron
  3.5 Super has its own smaller set.
- **No weights or binaries here.** This repository holds the patches, the app's pages, the tools and the measurements.
  The installer is on the releases page and the model files on Hugging Face.

## Layout

```
bootstrap/          fetch, patch and build the runtime; UPSTREAM.json is the delta's hash record
patches/            the patch set (a script per area + whole files), its manifest, the recipe's history
tools/              manager, downloader, measurement harnesses, verification
integrations/jan/   the app's pages on a Jan checkout
docs/               getting-started · install · engine · results · measuring · decode-budget · dead-ends
```

## Licence

MIT, see [LICENSE](LICENSE). Third-party attribution in [NOTICE.md](NOTICE.md): this patches
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), parts of the decode kernels adapt Apache-2.0 code whose
licence ships in the install's `runtime\licenses`, and the app is built on [Jan](https://github.com/menloresearch/jan)
(Apache-2.0).

If this is useful in your project, please credit Rulith Inference with a link to
[this repository](https://github.com/rulith-dev/rulith-inference). That is a request, not a condition of the licence.

---

Rulith Inference is made by [Rulith](https://rulith.ai): verifiable execution infrastructure for AI agents.
Questions, ideas and help: the [Rulith community discussions](https://github.com/rulith-dev/community/discussions).
