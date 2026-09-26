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
| `fixtures.ts` | canned manager answers for the browser preview (below); development builds only |
| `strixllama.css` | a bounded log viewport that pauses auto-follow when you scroll up |
| `rulith-theme.css` | the palette, type and corners for the whole app (below) |
| `strixllama.rs` | one Tauri command that pipes a bounded JSON request to `tools/manager.py` — no shell, no arbitrary executable |
| `locales/{en,zh-CN}/strixllama.json` | English and Chinese |
| `icons/` | the application icon, drawn by `make_icons.py` with no image library |

Language follows Jan's own setting: Jan discovers i18n namespaces with `import.meta.glob`, so the
locale files only have to be dropped in. 266 keys, identical key sets in both languages.

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
