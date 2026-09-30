# Jan overlay

The model pages inside a [Jan](https://github.com/menloresearch/jan) checkout — a library of the
model files found, the load configuration, the server log — plus a welcome screen in place of Jan's
setup screen, the model's state at the foot of the sidebar and above the chat, and Rulith's design
language for the whole app. Optional — everything works without it; `tools/manager.py` accepts JSON
on stdin and the server is a plain OpenAI-compatible endpoint.

This is an **overlay, not a fork**. It patches a Jan checkout you install yourself, so the running
app is Jan under Jan's Apache-2.0 licence. See [NOTICE.md](../../NOTICE.md).

Base: Jan **v0.8.4**, commit `5f30aee467f08941964a83f946e2663e7ae0e01f`. `apply.py` fails rather
than guessing if an upstream anchor has moved.

## Apply

```bash
git clone --depth 1 --branch v0.8.4 https://github.com/janhq/jan src/jan
python integrations/jan/apply.py src/jan     # then build Jan as its own README describes
```

Re-running it is safe: every edit is anchored and becomes a no-op once applied.

## What it installs

| File | |
| --- | --- |
| `StrixLlamaPage.tsx` | the frame of the model pages: the page's name in a strip across the top, one centred column, and the alerts that persist (a failed load, little Windows commit left) above the page |
| `ModelsView.tsx` | Model › Library: the files found, filtered by kind, with load (or switch) and configure actions, and the model-folder dialog |
| `ConfigurationView.tsx` | Model › Configuration: the launch profile in labelled sections, a one-line description per row and the long explanation behind an (i); unsaved edits get a bar with *Discard*, *Save* and *Save and reload* |
| `LogsView.tsx` | Model › Logs: the server's state, model (with its API id to copy), runtime (ROCm release, GPU target) and endpoint, what each conversation slot is doing, the launch parameters, and a live log with search, level filter, pause, copy and export |
| `Welcome.tsx` | shown until the chat has a model: finds the model files or asks for their folder, and loads one |
| `ModelState.tsx` | the strip above the chat when the model cannot answer (loading, not loaded, failed), with the action that fixes it |
| `Sidebar.tsx` | the sidebar's brand block, the Model group (Library, Configuration, Logs), and the model server panel at its foot: state, load / unload, the model with a menu to run another, Settings and the guide |
| `StrixLlamaSync.tsx` | the app-wide status poll, the provider registration, the optional load at startup, and the load-finished and out-of-memory toasts |
| `store.ts`, `status.ts`, `parts.tsx` | shared state (catalog, profile being edited), the manager request and types, the small components the views share |
| `attachments.ts` | documents in the chat: a dropped file's text read by the app (below), and a document the model's context cannot hold refused before it is sent |
| `compact.ts` | a conversation that outgrows the model's context folded into a summary (below) |
| `downloads.ts` | files the page saves (a code block's download button, a table's CSV, the Logs page's log) written into Downloads by the app (below) |
| `fixtures.ts` | canned manager answers for the browser preview (below); development builds only |
| `strixllama.css` | a bounded log viewport that pauses auto-follow when you scroll up |
| `rulith-theme.css` | the palette, type and corners for the whole app (below) |
| `strixllama.rs` | one Tauri command that pipes a bounded JSON request to `tools/manager.py` — no shell, no arbitrary executable — one that reads a dropped document's bytes with Jan's own document parser (`tauri-plugin-rag`) through a temporary file it removes again, and one that writes a file the page saves into the Downloads folder, numbering the name when it is taken |
| `tests/compact.test.ts` | the unit tests of `compact.ts`, run with the web app's vitest beside the module (not copied by `apply.py`) |
| `locales/{en,zh-CN}/strixllama.json` | English and Chinese |
| `icons/` | the application icon, drawn by `make_icons.py` with no image library |

Language follows Jan's own setting: Jan discovers i18n namespaces with `import.meta.glob`, so the
locale files only have to be dropped in. 281 keys, identical key sets in both languages.

## Design

The app follows Rulith's design language (Rulith Console and the Rulith desktop app): neutral greys,
a white primary button on dark and a black one on light, hairline borders, 6-8 px corners, the
system's own type (Segoe UI Variable, Microsoft YaHei UI for Chinese), small uppercase labels over
bordered cards, and one brand colour — Rulith green — for state. `rulith-theme.css` overrides Jan's
theme variables rather than its components, so the chat, Jan's settings and the model pages all
follow; the few structural changes are patched by `apply.py`:

- the sidebar is flush instead of a floating card, with the model pages as a group of their own and
  Settings in the model server panel at its foot;
- Jan's settings cards take the same label-over-card layout and type sizes as the model pages;
- "Colored user message bubble" tints your messages Rulith blue; green stays reserved for state;
- the accent colour picker is gone from Settings › Appearance: the palette is fixed, so the picker would
  change nothing.

## Looking at the pages without the app

`yarn dev:web` in the Jan checkout serves the web app alone, with no Tauri and no manager behind it.
The pages then take their answers from `fixtures.ts` - a catalog of the Qwen3.8 Flash Next files, a
running server, a log - so they can be looked at and screenshotted on any machine. `?state=` picks
the starting state (`ready`, `loading`, `stopped`, `failed`, `commit` for the low-commit warning, `memory` for answers stopped by it,
`none` for no model files); loading and unloading then behave as the real server would. The module
is imported behind `import.meta.env.DEV`, so a production build does not contain it.

## What it renames

The built application is strixllama's, not Jan's: `productName`, the window title (which lives in
`tauri.windows.conf.json`, not in `index.html`), the icon, and the bundle identifier — the last of
which gives it **its own application-data directory** instead of sharing the one a real Jan install
uses. Pass `--keep-data-dir` to leave the identifier alone.

Renaming a build of Jan is what Apache-2.0 allows; [NOTICE.md](../../NOTICE.md) states plainly what
this is. Do not present it as endorsed by Jan.

`apply.py` also drops `yarn build:icon` from `build:tauri`. That step regenerates every icon size
from `icon.png` by plain downscaling, which would overwrite the 16 px icon that `make_icons.py`
deliberately draws differently — at that size the ring is under a pixel wide, so it is dropped and
the eyes grow instead.

## Building behind a GitHub block

Jan's `yarn download:bin` pulls bun and uv from GitHub releases. Where that is unreachable, fetch
the two archives through a mirror and unpack them yourself:

```bash
curl -L -o scripts/dist/bun-windows-x64.zip <mirror>/https://github.com/oven-sh/bun/releases/latest/download/bun-windows-x64.zip
curl -L -o scripts/dist/uv-x86_64-pc-windows-msvc.zip <mirror>/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
```

**Unpack them by hand as well.** `download-bin.mjs` puts the download *and* the decompress inside
one `if (!exists)`, so an archive that is already there makes it skip both, and the build then fails
on a missing `uv.exe` rather than on a missing download. The files it wants in
`src-tauri/resources/bin/` are `bun.exe`, `bun-x86_64-pc-windows-msvc.exe`, `uv.exe` and
`uv-x86_64-pc-windows-msvc.exe`. sqlite-vec failing is fine — the script marks it non-fatal.

The Jan checkout itself may also need a mirror: a `--filter=blob:none` clone cannot restore a
deleted file without reaching the origin, which is a bad surprise to discover later. Prefer
`--depth 1 --branch v0.8.4`, which is a complete tree for that one revision.

## What it removes from Jan

`apply.py` also calls `converge_settings()`, which strips settings surfaces this build has no path
through — Jan's own llama.cpp engine controls, the Hub download token, and Jan's Resources and
Community cards. They point at upstream Jan rather than at this build.

## Documents in the chat

Jan reads an attached document either into the message or into a vector store, through an embedding
model its own llama.cpp engine serves. That engine is not in this build, so `documents()` in
`apply.py` sends every document into the message whole: *Add documents or files* works whatever the
model (Jan offers it only to a model with the `tools` capability; text in the message needs nothing of
the model, and in 0.3.6 the item stayed grey after an update until a model had loaded), a document dropped on
the chat box is read as it lands (Jan's chat box takes only images, audio and video there), and a
file that gives no text, or that the loaded context cannot hold, is refused with the reason instead of
falling back to embeddings. Settings › Attachments keeps the switch and the size limit; the chunking
and retrieval settings belonged to the engine. Images need the model loaded with *Accept images*:
then the model carries the `vision` capability.

## Web search

With `tools`, Jan's own web search works too: the model gets `web_search` and `web_fetch`, which
Jan's websearch plugin answers from Exa's keyless endpoint (Tavily or a SearXNG instance can be set in
Settings › Web Search). Jan has it on by default; `web_tools()` in `apply.py` starts it off - a stored
setting from before included - so no chat sends the tools, or a search query to a third party,
until the globe in the chat box turns it on.

## Long conversations

A conversation that outgrows the model's context used to fail with "Model ran out of context size", and every later
message with it. `compact.ts` (hooked into Jan's chat transport by `compaction()` in `apply.py`, for the local provider)
folds it instead: when the next request would pass 80% of the context the server was loaded with - counted from the
server's own usage figures up to the last answer, plus an estimate that takes a CJK character as a token - the model
first summarizes the conversation so far, and the request goes out as the summary (in the system prompt) plus the new
message. The summary request is the previous request plus an instruction, same system prompt, same conversion, same
tool definitions (without their execute), thinking off, so the server answers it from its cache and only writes the
summary; later requests keep the same head and reuse the cache again. The thread keeps the summary and the id of the
last message it covers (`metadata.strix_compact`); editing or deleting that message drops it. When no summary can be
made, the oldest messages are left out, with a note for the model. Jan's own trimmer (`lib/context-manager.ts`) is not
used: it counts 3.5 characters a token, a third of the truth for Chinese, and re-trims or re-summarizes on every
request, so the prompt's head changes each turn and the server re-reads the whole window every time.

## Several conversations at once

The thread route is one component reused across threads, and each thread's AI SDK `Chat` keeps the callbacks it was
created with, so the tool loop's `AbortController` - a single `useRef` - was shared by every conversation: two answers
that called tools at the same time overwrote and cleared each other's, and the one whose results came back last never
sent them back to the model. `concurrent_tools()` in `apply.py` gives each thread its own controller; a thread you
leave while its tools run keeps going, and only a tool call waiting for your approval stops when you leave, as before.

## Downloads

The code blocks in an answer have a download button (Jan's `streamdown`), tables an export, the Logs page a log
download; all of them click a `download` link to a blob URL, which Tauri's webview saved nowhere visible (issue #5).
`downloads.ts` takes those clicks: it keeps each blob by its URL as it is made (the page's `fetch` is the HTTP plugin's
and cannot read a blob URL), holds back the revoke until the bytes are read, and hands them to
`strixllama_save_download`, which writes them into the Downloads folder; a notice says where, with a button to show
the file.

## Updates

Jan's updater stays, pointed at this project instead of Jan. `apply.py` replaces Jan's endpoint and
signing key with this repository's latest release (`releases/latest/download/latest.json`) and
`updater.pub`, and takes the update prompt's release notes from this repository too. Before an
update is installed the app unloads the model, because the update replaces the runtime; the
setup's pre-install hook stops a server that is still running from the install directory.

A build signs its setup only when `TAURI_SIGNING_PRIVATE_KEY` is set (`apply.py` then turns on
`createUpdaterArtifacts`). `tools/make_update_manifest.py` writes `latest.json` from the signed setup
and its `.sig`; both go up with each release. A build without the key still checks for, verifies and
installs the official releases.

**Removing the upstream project's own links is a deliberate choice, not an oversight.** Comment out
`converge_settings()` to keep them.

## Configuration surface

The page shows what is actually worth turning: the model file and whether it is the one loaded (with
*Load*, *Switch* or *Unload*), thinking depth, context length, the KV cache type, keeping
conversations on disk (and how much disk), concurrent conversations and the context pool they share,
MTP (draft model, length, threshold), image input, and loading the model when the app starts. GPU
layers, batch and micro-batch, CPU threads and n-gram drafting are behind **Advanced** — the defaults
are the measured best for this machine. Each row says in one line what it does; the longer
explanation, with the measured costs, is behind its (i).

Flash Attention and sparse attention are stated rather than offered: turning either off runs out of
memory above 64K context on this model.
