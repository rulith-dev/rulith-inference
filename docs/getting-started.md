# Getting started with the installer

*[中文](getting-started.zh.md)*

This page is for running Strix Llama from the release installer. Nothing on it needs Python, ROCm,
Visual Studio or a source checkout — the installer carries the whole runtime. Building it yourself
is a different page: [install.md](install.md).

The whole path is: install the app, download five files into one folder, point the app at the
folder, press *Load model*, chat.

## What you need

| | |
|---|---|
| Machine | **Ryzen AI Max+ 395** — Radeon 8060S, `gfx1151`, 128 GB unified memory. The bundled kernels are compiled for gfx1151 only; on any other GPU the model will not load. |
| GPU carve | **96 GB**, set in the BIOS (the option is usually called *UMA frame buffer size* or *dedicated graphics memory*). At 64 GB the 93.7 GB model spills into shared memory and decode is 28% slower; no software setting recovers that. |
| Windows | 11 |
| Disk | about 100 GB for the model files, 0.5 GB for the app |
| Download | five files, about 97 GB, from Hugging Face |

## 1. Install the app

Download `Strix-Llama_<version>_x64-setup.exe` from the
[releases page](https://github.com/rulith-dev/strixllama/releases) and run it. That is the whole
installation: the app, `llama-server` built from our fork, the ROCm libraries it needs, the process
manager and its own Python are all inside. Nothing else has to be installed.

From 0.2.5 on, the app updates itself. It checks this project's GitHub releases when it starts, and when
there is a newer one it offers it in a prompt, and as **New version** at the foot of the sidebar. The
setup it downloads is installed only if its signature verifies against the key the app was built
with. The model is unloaded first, because the update replaces the runtime. Chats and settings are kept.
To upgrade by hand, or from 0.2.4 or earlier, run the newer installer over the old one; where GitHub
cannot be reached, download the setup yourself.

## 2. Download the model files — all into one folder

The app loads three kinds of file, and it finds the second and third **by name, next to the first**,
so all of them must sit in the same folder. Keep the file names exactly as they are.

The folder we recommend is the one LM Studio would use:

```
D:\models\unsloth\Qwen3.8-Flash-Next-GGUF\
```

If LM Studio has already downloaded this model, you have most of the files and can skip to the
next section — the app reads LM Studio's folder in place, nothing is copied.

All five come from one Hugging Face repository,
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). The links and
commands below are pinned to its revision `38bb39e`, the one every file was checked against (sha256,
2026-09-26):

| # | Path in the repository | Size | What it is |
|---|---|---|---|
| 1 | [`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf) | 11 MB | the model, shard 1 of 3 (metadata) |
| 2 | [`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf) | 49.8 GB | the model, shard 2 |
| 3 | [`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf) | 43.8 GB | the model, shard 3 |
| 4 | [`MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf) | 1.9 GB | the MTP draft: speculative decoding, roughly +60% decode speed |
| 5 | [`mmproj-F16.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf) | 904 MB | the vision projector: image input |

**Why exactly these.** Every number this project publishes was measured on the `UD-IQ4_XS` quant;
other quants of the same model load, but nothing here is tuned or verified for them. Files 1–3 are
the model and are required. Files 4 and 5 are optional: the draft and the projector are switched on
automatically when they are in the folder and left off when they are not, and the Configuration
page says which of the two it did not find. Without the draft, decode is about 40% slower.
(`mtp-…-shared-Q8_0.gguf` works in place of file 4; the two are within 1% of each other.)

**One more, optional: the draft head** — [`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`](https://github.com/rulith-dev/strixllama/releases/download/v0.1.2/mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf), 349 MB,
from this project's [releases page](https://github.com/rulith-dev/strixllama/releases). Put it in the
same folder as file 4. On the next rescan the app combines the two into
`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf` (a few seconds, once) and prefers it: decode
is 5% faster on English and 9% on Chinese with the same acceptance rate, which is the configuration
the numbers below were measured with. See [section 5](#5-optional-the-faster-draft-head-5-9-decode).

### Downloading

With [aria2](https://aria2.github.io/) in PowerShell — this puts the five files flat into one folder,
resumes if interrupted, and is the fastest way we know of for files this size:

```powershell
$base = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/38bb39ee97821de2c9009abb7e93950eec396e66"
$dir  = "D:\models\unsloth\Qwen3.8-Flash-Next-GGUF"
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($f in
    "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
    "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
    "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf",
    "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf",
    "mmproj-F16.gguf") {
  aria2c -x 8 -s 8 -c -d $dir -o (Split-Path $f -Leaf) "$base/$f"
}
# optional: the draft head (section 5)
aria2c -x 8 -s 8 -c -d $dir "https://github.com/rulith-dev/strixllama/releases/download/v0.1.2/mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf"
```

If Hugging Face is slow or unreachable where you are, use the mirror: replace `huggingface.co` in
`$base` with `hf-mirror.com`.

With the Hugging Face CLI instead (`pip install huggingface_hub`):

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --local-dir D:\models\unsloth\Qwen3.8-Flash-Next-GGUF \
  --include "UD-IQ4_XS/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" "mmproj-F16.gguf"
```

This one keeps the repository's sub-folders (`UD-IQ4_XS\`, `MTP\`), so afterwards **move the four
files up** into the folder beside `mmproj-F16.gguf`. For the mirror, set `HF_ENDPOINT=https://hf-mirror.com`
before running it. (Older installs call the command `huggingface-cli download`.)

A browser works too: open each of the five paths on the repository page, download, and save all of
them into the same folder.

## 3. Tell the app where the folder is

Open the app. The welcome screen looks for the model files.

- If LM Studio is installed, its download folder is registered automatically and the model is
  already found.
- Otherwise press **Model folders**, add the parent folder — `D:\models` — with **Browse…** or by
  pasting its path, and press **Save and scan**. Folders are scanned recursively, so the parent is
  enough.

The same list is under **Model › Library** in the sidebar: the model shows up as one row, *IQ4_XS ·
93.7 GB · 3 shards*. *Incomplete* means a shard is missing. The list is cached: after adding or
moving files, press **Rescan**. Switch the file-type filter to *MTP drafts* or *Vision projectors*
to check that files 4 and 5 were seen.

## 4. Load it and chat

Press **Load model** on the welcome screen (or on the model's row under **Model › Library**). There
is nothing to set first: the defaults are the configuration every published number was measured
with — context 262144, batch and ubatch 8192, flash attention, sparse attention, the MTP draft and
image input all on, thinking depth *medium*. **Model › Logs** shows the load progressing. The first
load reads 93.7 GB from disk; later ones are faster.

When it is ready the chat opens by itself, with the model selected — just type. Image input works by
attaching a picture to the message. The app does not load a model on its own when it starts: press
**Load model** in the strip above the chat or in the model server panel, or turn on **Model ›
Configuration › Startup** to have it load the last used model. The strip also says when a model is
loading. The model server panel at the foot of the sidebar
shows the same state, with the same button, on every page.

What to expect, measured on the reference machine:

| | |
|---|---|
| short context | about 41 tok/s |
| 86K tokens of context | about 33 tok/s; prefill about 1180 t/s |
| three / four conversations at once | about 54 / 61 tok/s summed; *Concurrent conversations* is 4 by default (Model › Configuration › Concurrency), and each keeps its conversation resident |

With *Keep conversations on disk* on (Model › Configuration; off by default), a conversation you
come back to later is not processed again: its state is kept on disk (about 30 KB per token, up to
16 GB) and read back in about a second.

If the load fails with out of memory, this configuration does not fit: raise the carve in the BIOS,
lower the context length under Model › Configuration, or give Windows a larger page file (RAM plus
page file is the limit that runs out first; see [install.md](install.md)).

The server is also an ordinary OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1` while the
model is loaded, for any other client.

## 5. Optional: the faster draft head (+5–9% decode)

The decode figures above were measured with a draft file that carries its own IQ4_XS output
projection instead of borrowing the model's; it is 5% faster on English and 9% on Chinese, with the
same acceptance rate. Everything else on this page works without it.

Unsloth's `shared-*` drafts have no output projection of their own: every draft step streams the
model's 521 MB one. The projection is a single tensor, so it is shipped on its own —
`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`, 349 MB, on the
[releases page](https://github.com/rulith-dev/strixllama/releases). Download it into the model's
folder, next to file 4, and press **Rescan** under Model › Library. The app writes
`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf` beside them (a byte-for-byte splice of the
two files, a few seconds, done once; it is the only file the app ever writes into a model folder),
reports it in a notice, and every profile that has not chosen a draft by hand uses it from then on.
The draft list under Model › Configuration shows all three.

Building the head yourself instead needs a source checkout with the toolchain:
[install.md, section 5](install.md#5-model-files).

## If something goes wrong

- **The model library is empty.** Check *Model folders*: only `.gguf` files under a registered
  folder are listed, and a split model is listed by its `-00001-of-` shard. Press *Rescan*.
- **MTP or image input is off and Model › Configuration says it did not find the file.** File 4 or
  5 is not in the model's folder (or is named differently). Add it and *Rescan*; the switch turns
  itself on. A load only refuses to start over these when a saved profile has the switch on by hand
  and the file has since gone.
- **The load starts, then dies.** Read Model › Logs. The usual causes are a GPU other than
  gfx1151, or a carve or page file too small for the context length.
- **Port 8080 is in use.** A `llama-server` from an earlier instance of this app is still running;
  the model pages adopt it and can unload it. Anything else on that port has to be stopped by hand.
- **Where things are.** The app is in `%LOCALAPPDATA%\Strix Llama`; the manager's settings, model
  catalog and last server log are in `runtime\config\jan\` under it; chats are in
  `%APPDATA%\strixllama\data`. Uninstalling removes the app folder including those settings; chats stay.
