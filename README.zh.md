# Rulith Inference

在一台 AMD Strix Halo 电脑上、在 Windows 下，把 125B 的 MoE 模型跑快。

Rulith Inference 是一个桌面应用加本地服务器，在 **Ryzen AI Max+ 395**（Radeon 8060S，gfx1151，128 GB 统一内存）上运行
**Qwen3.8-Flash-Next**（总参数 125B，每个 token 激活 6B）。一个安装包包含全部运行环境；应用自己下载模型，供它自带的聊天窗口
和任何兼容 OpenAI 或 Anthropic 接口的客户端使用，包括编程智能体。底层是 llama.cpp，为这块 GPU 和这个模型改了 181 个文件。

*[English](README.md)* · *原名 Strix Llama*

## 速度

Rulith Inference 0.6.9，目标机器，应用默认设置（GPU 显存 64 GB 档、8 个对话槽、MTP 推测解码），推荐的 EXL3 4.05 bpw 文件：

| | |
| --- | --- |
| 读提示词 | 16.9K-86K token **1,300-1,330 t/s** |
| 单个对话，MTP 开 | 解释说明 **58 tok/s**（草稿接受率 66%），写代码 **70 tok/s**（84%），散文 **46 tok/s**（45%） |
| 单个对话，MTP 关 | **33 tok/s** |
| 86K token 上下文之后 | **46 tok/s**（MTP 开，接受率 59%） |
| 8 个对话同时 | 合计 **85 tok/s**（MTP 关 91） |

开着推测解码时，速度取决于文本有多好预测，所以每个 MTP 数字都带上草稿接受率（接受的草稿 token / 提出的草稿 token）。
本仓库里每个数字都附带产生它的命令（[docs/results.md](docs/results.md)）；[docs/measuring.md](docs/measuring.md)
列出了这个项目曾经测错自己的各种方式。

## 开始使用

1. **安装。** 从 [Releases 页面](https://github.com/rulith-dev/rulith-inference/releases) 下载
   `Rulith-Inference_<版本>_x64-setup.exe` 并运行。不需要 Python、ROCm 或编译工具：运行环境都在里面。之后应用会自动更新，
   只安装签名校验通过的更新。
2. **设置 GPU 显存**：在 BIOS 里设为 64 GB 或 96 GB（选项通常叫 *UMA frame buffer size* 之类）。64 GB 时放不下的专家
   留在系统内存里，读取速度相同。
3. **下载模型。** 打开 **模型 › 下载**，在 *推荐* 里选一个文件，排第一的是 EXL3 4.05 bpw。应用会从 Hugging Face
   （或其镜像）下载模型、它的 MTP 草稿和视觉投影模型，逐个校验 SHA-256 后加入模型库。已经有的文件（比如 LM Studio
   的目录）原地读取：在 **模型 › 模型库 › 模型文件夹** 里加上它们的文件夹即可。
4. **加载并聊天。** 点 **加载模型**。默认设置就是上面那些数字的测量配置，不需要先调任何东西。附上一张图片即可针对图片提问。

[docs/getting-started.zh.md](docs/getting-started.zh.md) 逐步说明每一步该看到什么、某一步失败时怎么办。

## 模型

调优并实测过的是 Qwen3.8-Flash-Next 的四个文件。质量用与 BF16 模型的 KL 散度（越低越接近）和六类文本上的 top-1
一致率表示；速度是单个对话开 MTP 写代码（括号内为草稿接受率），以及读一个 16.9K token 的提示（0.6.8，64 GB 档；0.6.9 的单对话路径与之相同）。

| 文件 | 大小 | KLD | top-1 | 代码，MTP 开 | 读提示词 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **EXL3 4.05 bpw**（推荐） | 104.3 GB | **0.022** | **96.0%** | 70.9 tok/s（84%） | 1,332 t/s |
| EXL3 3.05 bpw | 82.4 GB | 0.044 | 94.5% | 75.9 tok/s（81%） | 1,327 t/s |
| UD-Q4_K_XL | 111.3 GB | 0.024 | 95.9% | 58.8 tok/s（91%） | 1,379 t/s |
| UD-IQ4_XS | 93.7 GB | 0.048 | 94.4% | 63.6 tok/s（88%） | 1,414 t/s |

EXL3 文件是 turboderp 的量化，为本引擎重新打包：
[Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF](https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF)，
带自己的 MTP 草稿，只能在 Rulith Inference 里加载。UD 文件来自
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)。NVIDIA Nemotron 3.5 Super（120B-A12B，Q4_K_M）也能加载，
支持图像输入和它自己的 MTP 层。该模型的其他量化版本也能加载，走 llama.cpp 原版内核。

## 给其他程序用

模型加载后，服务器在 `http://127.0.0.1:8080/v1` 提供 OpenAI 的 Chat Completions 与 Responses 接口和 Anthropic 的
Messages 接口，支持流式输出和工具调用。**模型 › 配置 › 网络** 可以改端口、开放给局域网、设置 API 密钥
（`Authorization: Bearer <密钥>`）。请求里的模型名就是文件名去掉分片后缀（`Qwen3.8-Flash-Next-EXL3-4.05bpw`）；只加载了
一个模型时，任何名字都会应答。

同时保留 8 个对话；一个槽只接收接续它已有内容的请求，所以在同一段长系统提示上跑多个会话的智能体工具不会互相截断历史。
别的会话已经读过的系统提示在 GPU 上直接复制，不再重读。

**Codex**，在 `~/.codex/config.toml` 里：

```toml
model = "Qwen3.8-Flash-Next-EXL3-4.05bpw"
model_provider = "rulith"

[model_providers.rulith]
name = "Rulith Inference"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
# env_key = "RULITH_API_KEY"   # 服务器设了 API 密钥时
```

**Claude Code**：设置 `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`，`ANTHROPIC_AUTH_TOKEN` 设为 API 密钥（没有密钥时
任意文本），`ANTHROPIC_MODEL` 和 `ANTHROPIC_SMALL_FAST_MODEL` 设为模型名。`web_search` 这类托管工具不会执行；响应会在
`X-Rulith-Unsupported-Tools` 头里列出它们。

## 引擎里有什么

相对上游 llama.cpp 的改动，按用途归类。[docs/engine.zh.md](docs/engine.zh.md) 逐项列出测量数字、所在版本和开关。

- **读提示词。** 专家内核按 gfx1151 的执行方式重写（它的向量单元和矩阵核心从不同时工作，内核每花一条指令解包权重，
  就损失一份矩阵时间）：IQ3_S、Q4_K 和 EXL3 专家走矩阵核心，delta net 以 16 token 为块在矩阵核心上计算，专家周围的
  零碎工作融合掉。
- **生成。** 解码按内存带宽的速度读权重：EXL3 权重经 GPU 本地内存里的环形缓冲流入，一步里所有被路由的专家在一次
  启动里完成，稀疏注意力只收集选中的缓存单元，主机等待 GPU 时绕开运行时约 0.65 ms 的延迟。
- **推测解码。** 模型自带的 MTP 层配合 Rulith 的低秩草稿头起草；草稿长度按接受率和同时生成的对话数调整，采样时每个
  token 的分布与模型单独采样完全相同。
- **多对话与智能体。** 所有正在回答的对话不论长短共用一次前向，在后续提示会分叉的位置保存检查点，提示缓存可以保存到
  磁盘，delta net 状态每九个 token 才写回一次。
- **内存。** 应用先算出一次加载需要多少内存，把 GPU 显存档放不下的专家放进锁页系统内存，读取速度不变；Q8_0 K/V 缓存
  可选。
- **同样的输入，同样的输出。** 释放的缓存单元会清零，草稿的状态跟随对话经过回退和缓存，所以回答不受之前运行过什么的影响。

## 从源码构建

```bash
python bootstrap/bootstrap.py --toolchain          # ROCm SDK + ninja 装进 toolchain/，约 5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` 在固定版本克隆上游 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，打上补丁集，用 ROCm SDK
编译。同一份改动在 fork 上可以直接看 diff：[rulith-dev/llama.cpp 的 `rulith` 分支](https://github.com/rulith-dev/llama.cpp/tree/rulith)。
`tools/manager.py` 掌管启动参数、环境变量和运行环境，所以配置是可复现的，而不是靠记忆。先读
[docs/install.md](docs/install.md)：ROCm SDK 必须用 TheRock 的（同一份源码预填充快 60%），而它的 nightly 索引只保留每个
版本约四周。

`python tools/replay_bootstrap.py` 从干净的上游加配方重建 181 个改动文件并逐字节比较，结果是 **181 / 181**。
`patches/MANIFEST.md` 记录每项改动及其测量；`integrations/jan/apply.py` 把应用的页面构建到一份
[Jan](https://github.com/menloresearch/jan) 源码上。

打过补丁的服务器也能在 Linux 上用 ROCm 编译，有用户在 Linux 上测得与 Windows 相同的速度（UD-IQ4_XS）。没有 Linux 安装包。

## 适用范围与限制

- **只支持 Windows 11 和 gfx1151。** 随附的内核只为 Radeon 8060S 编译；换别的 GPU 模型加载不了。
- **只针对一个模型系列调优。** 注意力和专家方面的工作专门针对 Qwen3.8-Flash-Next（`qwen4exp`）；Nemotron 3.5 Super
  有它自己较小的一套。
- **这里没有权重和二进制文件。** 本仓库存放补丁、应用页面、工具和测量结果。安装包在 Releases 页面，模型文件在
  Hugging Face。

## 目录结构

```
bootstrap/          获取、打补丁、编译运行环境；UPSTREAM.json 是改动的哈希记录
patches/            补丁集（每个领域一个脚本 + 整文件）、清单、配方历史
tools/              进程管理器、下载器、测量工具、验证
integrations/jan/   应用页面（构建在 Jan 源码上）
docs/               getting-started · install · engine · results · measuring · decode-budget · dead-ends
```

## 许可

MIT，见 [LICENSE](LICENSE)。第三方署名见 [NOTICE.md](NOTICE.md)：本项目为 [llama.cpp](https://github.com/ggml-org/llama.cpp)
（MIT）打补丁，部分解码内核改编自 Apache-2.0 代码（其许可随安装附在 `runtime\licenses`），应用构建在
[Jan](https://github.com/menloresearch/jan)（Apache-2.0）之上。

如果这些补丁对你的项目有用，请在致谢中注明 Rulith Inference 并附上[本仓库](https://github.com/rulith-dev/rulith-inference)
的链接。这是请求，不是许可条件。

---

Rulith Inference 由 [Rulith](https://rulith.ai) 制作：面向 AI 智能体的可验证执行基础设施。
问题、想法与帮助：[Rulith 社区讨论区](https://github.com/rulith-dev/community/discussions)。
