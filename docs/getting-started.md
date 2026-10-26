# Getting started with the installer

*[中文](getting-started.zh.md)*

This page is for running Rulith Inference from the release installer. Nothing on it needs Python, ROCm, Visual Studio
or a source checkout: the installer carries the whole runtime. Building it yourself is a different page:
[install.md](install.md).

The whole path is: install the app, download a model from inside it, press *Load model*, chat.

## What you need

| | |
|---|---|
| Machine | **Ryzen AI Max+ 395**: Radeon 8060S, `gfx1151`, 128 GB unified memory. The bundled kernels are compiled for gfx1151 only; on any other GPU the model will not load. |
| GPU memory | **64 GB** or **96 GB**, set in the BIOS (the option is usually called *UMA frame buffer size* or *dedicated graphics memory*). At 64 GB the app keeps the experts that do not fit in system memory, which the GPU reads at the same speed; 96 GB leaves the conversation cache more room. |
| Windows | 11 |
| Disk | 83-112 GB for the model files, depending on the file (104 GB for the recommended one), 0.5 GB for the app |

## 1. Install the app

Download `Rulith-Inference_<version>_x64-setup.exe` from the
[releases page](https://github.com/rulith-dev/rulith-inference/releases) and run it. That is the whole installation:
the app, `llama-server` built from our fork, the ROCm libraries it needs, the process manager and its own Python are
all inside.

The app updates itself. It checks this project's GitHub releases when it starts, and when there is a newer one it
offers it in a prompt, and as **New version** at the foot of the sidebar. The setup it downloads is installed only if
its signature verifies against the key the app was built with. The model is unloaded first, because the update
replaces the runtime; chats and settings are kept. To upgrade by hand, run the newer installer over the old one.

## 2. Download a model

Open **Model › Download**. *Recommended* lists the files Rulith Inference is tuned for and every published number was
measured with:

| File | Size | | |
|---|---:|---|---|
| **Qwen3.8-Flash-Next EXL3 4.05 bpw** | 104.3 GB | recommended: closest to the full model | [Rulith](https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF) |
| Qwen3.8-Flash-Next EXL3 3.05 bpw | 82.4 GB | smaller and faster, twice the KL divergence | [Rulith](https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF) |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 111.3 GB | | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| Qwen3.8-Flash-Next UD-IQ4_XS | 93.7 GB | | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |

Pick one; *Download options* switches between the versions of a repository. *Files that go with it* are the MTP draft
(speculative decoding: about 1.5-2x faster answers), the vision projector (image input) and, for the Unsloth files,
Rulith's draft head (faster drafting); leave them on. **Download** asks for the source (Hugging Face, or its mirror in
China) and the folder, and shows the free space there. Every file is checked against its SHA-256 when it is complete; a
download can be paused and continues where it stopped, also after a restart. When it is done, the model is in the
Library and **Load now** loads it.

The search box finds other GGUF repositories on Hugging Face. Files Rulith Inference has its own kernels for are marked
*Tuned*; other models are not tested by Rulith and run on llama.cpp's stock kernels.

### Files you already have

The app reads model files where they are; it never copies them. If LM Studio is installed, its download folder is
registered automatically. Otherwise add the parent folder (for example `D:\models`) under **Model › Library › Model
folders** with **Browse…** or by pasting its path, and press **Save and scan**; folders are scanned recursively. Keep
a model's draft, projector and head in the same folder as its first shard, under their original names: the app pairs
them by name.

To download by hand instead, with the Hugging Face CLI (`pip install huggingface_hub`; set
`HF_ENDPOINT=https://hf-mirror.com` first for the mirror):

```bash
hf download Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF --local-dir D:\models\rulith --include "EXL3-4.05bpw/*" "mtp-*.gguf" "mmproj-F16.gguf"
```

The CLI keeps the repository's folders, so afterwards move `mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf` and
`mmproj-F16.gguf` into `EXL3-4.05bpw\`, next to the shards.

## 3. Load it and chat

Press **Load model** on the welcome screen, on the model's row under **Model › Library**, or in the model server panel
at the foot of the sidebar. There is nothing to set first: the defaults are the configuration every published number
was measured with: context 262144, eight conversation slots, flash attention, sparse attention, the MTP draft and image
input on, thinking depth *medium*. **Model › Logs** shows the load progressing. The first load reads the whole file
from disk; later ones are faster.

When it is ready the chat opens by itself, with the model selected: just type. Attach a picture to ask about it. The
app does not load a model on its own when it starts unless **Model › Configuration › Startup** says to.

What to expect with EXL3 4.05 bpw (0.6.9, 64 GB setting):

| | |
|---|---|
| one conversation | 58 tok/s for an explanation, 70 writing code, 46 for prose (with MTP; speed depends on how predictable the text is) |
| after 86K tokens of context | 46 tok/s; a long prompt is read at about 1,330 tokens a second |
| eight conversations at once | about 85 tok/s summed. Agent tools need one slot for each session they run at once; *Concurrent conversations* is under Model › Configuration › Concurrency |

With *Keep conversations on disk* on (Model › Configuration; off by default), a conversation you come back to later is
not processed again: its state is kept on disk (about 30 KB a token, up to 16 GB) and read back in about a second.

If the load fails with out of memory, this configuration does not fit: raise the GPU memory in the BIOS, lower the
context length under Model › Configuration, or give Windows a larger page file (RAM plus page file is the limit that
runs out first; see [install.md](install.md)).

While a model is loaded the server is also an OpenAI- and Anthropic-compatible endpoint at `http://127.0.0.1:8080/v1`
for other clients, coding agents included; the [README](../README.md#use-it-from-other-programs) has the settings for
Codex and Claude Code. Model › Configuration › Network changes its port, lets other devices on your network use it and
sets an API key for them; Model › Logs shows the addresses in use.

## If something goes wrong

- **The model library is empty.** Check *Model folders*: only `.gguf` files under a registered folder are listed, and a
  split model is listed by its `-00001-of-` shard. Press *Rescan*.
- **MTP or image input is off and Model › Configuration says it did not find the file.** The draft or the projector is
  not in the model's folder (or is named differently). Add it and *Rescan*; the switch turns itself on. A load only
  refuses to start over these when a saved profile has the switch on by hand and the file has since gone.
- **The load starts, then dies.** Read Model › Logs. The usual causes are a GPU other than gfx1151, or GPU memory or a
  page file too small for the context length.
- **Port 8080 is in use.** A `llama-server` from an earlier instance of this app is still running; the model pages
  adopt it and can unload it. Anything else on that port has to be stopped by hand, or the server moved to another
  port under Model › Configuration › Network.
- **Where things are.** The app is `%LOCALAPPDATA%\Rulith Inference\rulith-inference.exe`. Up to 0.6.1 an install
  made before the rename stayed in `%LOCALAPPDATA%\Strix Llama` as `strixllama.exe`; the 0.6.2 setup renames that
  folder in place (it stays where it was if a program holds a file in it) and re-points the shortcuts and a taskbar
  pin; if a pin still shows a blank icon, pin the app again. The manager's settings, model catalog and prompt cache are
  in `runtime\config\jan\` under it, the server's logs in `runtime\logs\` (the newest ten runs and at most 4 GB are
  kept; past 1 GB a run's console log is cut back to the end of the load, and its request log keeps its newest two 1 GB
  files); chats are in `%APPDATA%\strixllama\data`. Uninstalling removes the app; the settings and the prompt cache go
  too when you tick *Delete the application data*, and chats stay either way.
