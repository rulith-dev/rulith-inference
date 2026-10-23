# Third-party code and licenses / 第三方代码与许可

*English below, 中文在后。*

---

## English

This repository contains only the scripts, patches, UI overlay and measurement data written for this
project. It contains no model weights, no third-party source trees and no prebuilt binaries: the
`src/`, `bin/`, `toolchain/` and `models/` directories are excluded by `.gitignore`;
`bootstrap/bootstrap.py` assembles the first two.

The repository's own code is released under the [MIT License](LICENSE).

### llama.cpp — MIT, Copyright (c) 2023-2026 The ggml authors

Every script under `patches/` is a modification of llama.cpp. Context lines quoted inside a patch
belong to the original authors and remain under llama.cpp's MIT license. Obtain the corresponding
source yourself, then apply the patches to it.

Upstream: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) at
`19ab5a5e65736a35437e6b1c0ccd6562fed01b1b`, pinned in `bootstrap/UPSTREAM.json` (up to 0.6.4:
`pwilkin/llama.cpp` at `f5daaa3cfa6358e5dd398911ec741813745a5440`, the base of `patches/history/`).
Every script in `patches/` applies to that revision, and `tools/replay_bootstrap.py` proves the set
reproduces the build exactly. The files under `patches/files/` that also exist upstream are
llama.cpp's files with this project's changes, under llama.cpp's MIT license.

`bootstrap/UPSTREAM.json` records the provenance of the 176 files this project changes: for each of
the 119 it modifies, the Git blob SHA-1 and SHA-256 the file has *upstream*, and for all 176 the
SHA-256 the patch set produces. The replay restores the upstream side by blob hash, so what it
rebuilds from is upstream's content by construction rather than by assertion.

### hipfire — Apache License 2.0, Copyright the hipfire authors

The gated delta net prefill kernel `ggml/src/ggml-cuda/gdn-wy.cuh` of the fork (shipped in
`patches/files/`) is adapted from hipfire's `kernels/src/gated_delta_chunk_wmma.gfx1151.hip`
([`warpfront/hipfire`](https://github.com/warpfront/hipfire), v0.4.2.1-13-gbfb90b9), by way of the
copy in the ExLlamaV3 port below; the file states what was changed. It remains under the Apache
License 2.0: the license text and hipfire's NOTICE are `licenses/LICENSE-hipfire-APACHE` and
`licenses/NOTICE-hipfire` in the fork, and the app ships both in `runtime\licenses`.

### ExLlamaV3 — MIT, Copyright (c) 2025 Turboderp; its gfx11.5 port — MIT, Copyright (c) 2026 Yamz Labs

The grouped EXL3 expert GEMM, the trellis decode and the WMMA helpers under
`ggml/src/ggml-cuda/vendors/kyojin/` of the fork come from the gfx11.5 port of ExLlamaV3
([`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)) published as
[`Yamz-Labs/kyojin`](https://github.com/Yamz-Labs/kyojin), MIT. `vendors/kyojin/README.md` lists what
was taken and changed; the license text is `vendors/kyojin/LICENSE`, shipped as
`runtime\licenses\LICENSE-exllamav3`.

### Jan — Apache License 2.0, Copyright 2025 Menlo Research (https://menlo.ai)

`integrations/jan/` is an overlay applied to a Jan checkout, not a fork of it. Base: Jan **v0.8.4**,
commit `5f30aee467f08941964a83f946e2663e7ae0e01f`.

`StrixLlamaPage.tsx`, `strixllama.css`, `strixllama.rs`, `locales/*/strixllama.json` and `apply.py` were written
for this repository and are MIT like the rest of it. They are not standalone: they compile only
inside a Jan checkout, against Jan's own UI components, provider store and i18n runtime, and
`apply.py` registers them into Jan's router and Tauri command table. Using them therefore means
using Jan, under Jan's Apache-2.0 license.

`apply.py` also removes settings surfaces this build has no path through, including Jan's own
Resources and Community cards - its documentation, release notes, GitHub and Discord links - which
point at upstream Jan rather than at this build. Stated here so the removal is disclosed rather than
silent; comment out `converge_settings()` to keep them.

### ROCm SDK

`toolchain/` is assembled locally from the ROCm SDK wheels (TheRock) plus `ninja` and `aria2`, and
is not redistributed here. Each component keeps its own license; the wheels used are
`rocm 10.2.0a20260925` and its `rocm_sdk_*` companions (`rocm 10.1.0a20260910` up to 0.2.2).

### Model weights — not in this repository

**Qwen3.8-Flash-Next**, GGUF quantisations by Unsloth, and the accompanying `mmproj-F16.gguf` vision
projector. Obtain them from their official channels and observe their own licenses.

### Local-only build inputs

- `corpus/` holds the prose the measurement tools read (`tools/decode_lab.py --gen-prefix`,
  `tools/vision_stress.py`). Nothing in it is ours: it was English and Chinese WikiText plus a
  code mix. Any equivalent long text works; the numbers in this repository were taken with that one.

---

## 中文

本仓库只包含为本项目编写的脚本、补丁、界面覆盖层和实测数据，不含任何模型权重、第三方源码树或预编译二进制：
`src/`、`bin/`、`toolchain/`、`models/` 均由 `.gitignore` 排除；前两者由 `bootstrap/bootstrap.py` 生成。

本仓库自身的代码按 [MIT](LICENSE) 分发。

### llama.cpp —— MIT，Copyright (c) 2023-2026 The ggml authors

`patches/` 下的每个脚本都是对 llama.cpp 的修改。补丁中引用的上下文行属于原作者，仍按 llama.cpp 的 MIT
许可分发。请自行获取对应版本的源码后再打补丁。

上游：[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)，版本
`19ab5a5e65736a35437e6b1c0ccd6562fed01b1b`，固定在 `bootstrap/UPSTREAM.json` 里（0.6.4 及以前是
`pwilkin/llama.cpp` 的 `f5daaa3cfa6358e5dd398911ec741813745a5440`，即 `patches/history/` 的基线）。
`patches/` 下的每个脚本都针对该版本；`tools/replay_bootstrap.py` 可证明整套补丁精确重建该构建。
`patches/files/` 下在上游也存在的文件，是带有本项目改动的 llama.cpp 文件，按 llama.cpp 的 MIT 许可分发。

`bootstrap/UPSTREAM.json` 记录本项目改动的那 176 个文件的来源：被修改的 119 个各记其**上游**版本的
Git blob SHA-1 与 SHA-256，176 个全部记补丁集产出的 SHA-256。重放时按 blob 哈希取回上游那一侧，
所以它据以重建的内容是上游的——这是构造保证，不是声明。

### hipfire —— Apache License 2.0，Copyright the hipfire authors

fork 中的 gated delta net 预填充内核 `ggml/src/ggml-cuda/gdn-wy.cuh`（随 `patches/files/` 提供）改编自
hipfire 的 `kernels/src/gated_delta_chunk_wmma.gfx1151.hip`
（[`warpfront/hipfire`](https://github.com/warpfront/hipfire)，v0.4.2.1-13-gbfb90b9），经由下面 ExLlamaV3 移植中的副本；
文件开头写明了改动。它仍按 Apache License 2.0 分发：许可全文与 hipfire 的 NOTICE 位于 fork 的
`licenses/LICENSE-hipfire-APACHE` 和 `licenses/NOTICE-hipfire`，应用把两者一起装在 `runtime\licenses` 里。

### ExLlamaV3 —— MIT，Copyright (c) 2025 Turboderp；其 gfx11.5 移植 —— MIT，Copyright (c) 2026 Yamz Labs

fork 中 `ggml/src/ggml-cuda/vendors/kyojin/` 下的 EXL3 专家分组矩阵乘法、trellis 解码与 WMMA 辅助代码，来自
ExLlamaV3（[`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)）的 gfx11.5 移植，
发布于 [`Yamz-Labs/kyojin`](https://github.com/Yamz-Labs/kyojin)，MIT 许可。`vendors/kyojin/README.md` 列出取用和改动的内容；
许可全文为 `vendors/kyojin/LICENSE`，随应用装在 `runtime\licenses\LICENSE-exllamav3`。

### Jan —— Apache License 2.0，Copyright 2025 Menlo Research（https://menlo.ai）

`integrations/jan/` 是作用于 Jan 源码树的**覆盖层**，不是对 Jan 的分叉。基线：Jan **v0.8.4**，
commit `5f30aee467f08941964a83f946e2663e7ae0e01f`。

`StrixLlamaPage.tsx`、`strixllama.css`、`strixllama.rs`、`locales/*/strixllama.json` 和 `apply.py` 由本仓库编写，
与其余部分同为 MIT。但它们不能独立存在：只有在 Jan 的源码树内、依赖 Jan 自身的 UI 组件、provider store
和 i18n 运行时才能编译，`apply.py` 还会把它们注册进 Jan 的路由和 Tauri 命令表。因此使用它们即意味着使用
Jan，受 Jan 的 Apache-2.0 许可约束。

`apply.py` 还会移除本构建没有路径可达的设置界面，其中包括 Jan 自己的 Resources 与 Community 卡片
（其文档、发布说明、GitHub 和 Discord 链接）——它们指向上游 Jan 而非本构建。在此写明，使这一移除是
公开披露而非悄悄进行；注释掉 `converge_settings()` 即可保留。

### ROCm SDK

`toolchain/` 由 ROCm SDK 的 wheel 包（TheRock）加 `ninja`、`aria2` 在本地拼成，本仓库不再分发。
各组件保留各自的许可；实际使用的是 `rocm 10.2.0a20260925` 及其 `rocm_sdk_*` 配套包（0.2.2 及以前为 `rocm 10.1.0a20260910`）。

### 模型权重 —— 不在本仓库内

**Qwen3.8-Flash-Next**（GGUF 由 Unsloth 量化），以及配套的视觉投影模型 `mmproj-F16.gguf`。
请从官方渠道获取并遵守其自身许可。

### 仅限本地生成的构建输入

- `corpus/` 存放测量工具读取的长文本（`tools/decode_lab.py --gen-prefix`、
  `tools/vision_stress.py`）。其中没有本项目的内容：原本是英文与中文 WikiText 加一份代码混合。
  任何等价的长文本都可以替代；本仓库的数字是用那一份取得的。
