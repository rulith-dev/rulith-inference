# Rulith Inference

在一台 AMD Strix Halo 机器上、在 Windows 下，把 125B 的 MoE 模型跑快。

目标平台：**Ryzen AI Max+ 395**（Radeon 8060S，gfx1151，128 GB 统一内存），模型
**Qwen3.8-Flash-Next**（125B-A6B，Unsloth UD-IQ4_XS，93.7 GB），经打过补丁的 llama.cpp 服务。

*[English](README.md)*

*原名 **Strix Llama**，0.3.6 起改为现名，以免和更早用这个名字的社区分支
[halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) 混淆。应用和数据都不变：
已安装的 Strix Llama 会原地更新过来。*

## 当前水平

本机用 0.4.0 实测（2026-10-01），推测解码开启，应用的默认设置（8 个对话槽、MTP、图像输入开启）：预填充为 95.6K
token 真实文本；解码为在同一文本的 86K token 之后、以及只有一句提问之后各生成 400 token；多路为 4 个槽下几个各约 4K
上下文的对话同时解码，每轮换一个采样种子。最后一行关闭 MTP，其中 0.3.9 的数字是当天与 0.4.0 交替测得的：

| | |
| --- | --- |
| 预填充 | **1237 t/s** |
| 解码，86K 上下文 | **23.9 ms/token**（41.8 tok/s，草稿接受率 65%；0.3.1：27.3，接受率 63%） |
| 解码，短上下文 | **21.7 ms/token**（46.1 tok/s，接受率 65%；0.3.1：22.5，接受率 67%） |
| 解码，3 / 4 路同时 | **59.8 / 69.1 tok/s** 合计（同样测法下单路 42.1 的 1.42 / 1.64 倍；0.3.1 为 55.4 / 62.5 与 39.7；合计随各轮采样的接受率浮动 ±10%） |
| 关闭 MTP 解码，8 个约 40K token 的对话同时 | 512K token q8_0 池中合计 **91.0 tok/s**（同一天 0.3.9 为 85.7，0.3.3 为 58.8）；8 个约 20K 的对话、f16 缓存 95.6（0.3.9：89.4）；单个对话 3K / 50K / 110K 时 27.5 / 27.0 / 26.5 tok/s（0.3.9：25.9 / 25.5 / 25.0） |
| 图像输入 | 支持（Qwen3-VL 投影模型） |

本仓库里的每个数字都附带产生它的命令，见 [docs/results.md](docs/results.md)。凡是改动无法在噪声
之上分辨的，就直接写明分辨不出，而不是算作收益；解码数字一律带上草稿接受率——开着推测解码时，脱离
接受率谈吞吐，描述的是那段提示词，不是这套运行时。

## 只想跑起来？

**[docs/getting-started.zh.md](docs/getting-started.zh.md)**——从
[Releases 页面](https://github.com/rulith-dev/rulith-inference/releases)装安装包、要下载的五个模型文件和
放在哪、点什么。不涉及 Python、ROCm 或任何编译工具。

## 模型文件

这里的每个数字都是用下面这些文件测的。链接固定在核对过的 Hugging Face 版本上（`38bb39e`：
2026-09-26 逐个文件比对过 sha256）：

| | 文件 | 大小 |
| --- | --- | --- |
| **模型**，必需 | Unsloth 的 Qwen3.8-Flash-Next-GGUF，[`UD-IQ4_XS`，三个分片](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS) | 93.7 GB |
| **MTP 草稿**：推测解码，解码约 +60% | [`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf) | 1.9 GB |
| **视觉投影**：图像输入 | [`mmproj-F16.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf) | 904 MB |
| **草稿头**，本项目提供：草稿自带的 IQ4_XS 输出投影，解码 +5–9% | [`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`](https://github.com/rulith-dev/rulith-inference/releases/download/v0.1.2/mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf) | 349 MB |

四样都平铺放进同一个文件夹：应用按文件名在模型第一个分片旁边找草稿、投影和草稿头。下载命令见
[docs/getting-started.zh.md](docs/getting-started.zh.md)。

## 快速开始（从源码）

```bash
python bootstrap/bootstrap.py --toolchain          # 把 ROCm SDK 和 ninja 装进 toolchain/，约 5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` 按固定版本号克隆 `pwilkin/llama.cpp`、应用补丁集、对着 ROCm SDK 构建。
`tools/manager.py` 是一个从 stdin 读 JSON 的进程管理器：启动参数、环境开关和运行时都由它掌握，
所以一套配置是可复现的，而不是靠记忆。

**先读 [docs/install.md](docs/install.md)。** 有三件事不是可选的，也不显然：显存划分必须是 96 GB
（64 GB 下模型装不下，解码慢 28%，没有任何软件设置能补回来）；ROCm SDK 必须用 TheRock 10.2 而不是
系统的 7.1（同样的源码，预填充 +60%）；而实测所用的 SDK 版本来自一个约 27 天滚动窗口的 nightly
索引——也就是说这个钉死的版本号迟早会失效，那篇文档写了届时怎么办。模型文件是另外约 95 GB 的下载。

可选：`integrations/jan/apply.py` 把模型页面（模型库、配置、日志）、替换 Jan 引导页的欢迎页、侧栏和
聊天页上方的模型状态，以及 Rulith 的设计风格叠加进 [Jan](https://github.com/menloresearch/jan) 的源码树，
中英双语。

## 里面到底有什么

相对上游 llama.cpp 的全部改动是 **83 个文件**——3610 个里改了 79 个、新增 4 个。主要几项：

| | |
| --- | --- |
| **IQ3_S 走矩阵核** | 这个模型 52% 的主体以 IQ3_S 存储，而 MMB 反量化 GEMM 不接受这种格式，于是一半权重从未到达矩阵核。预填充 +5.6%。 |
| **按这块 GPU 定价的专家内核** | gfx1151 上向量单元和矩阵核从不同时运行，反量化 GEMM 用来解包权重的每条指令都是矩阵核少算的时间。据此重写了路由专家的 IQ3_S gate/up（预填充里最大的内核）：每层 35.9 → 19.9 ms，预填充 +10%，输出逐位不变。 |
| **预填充批次的其余部分** | 专家内核到了极限之后，一个批次仍有约四分之一的 GPU 时间花在本应很便宜的工作上：4 个输出的投影走 64 行宽的 GEMM 分块、激活行全落在同一组内存通道上、索引器打分拆成多遍、稀疏注意力每层把缓存完整复制两遍。这些现在逐一融合或重写：2K token 批次 GPU 时间 1860 → 1573 ms，64K 深度 2276 → 1788 ms；95.6K token 提示词 992 → 1187 t/s。困惑度相同，但不再逐位相同。 |
| **解码期稀疏注意力** | 稀疏内核需要只有预填充才构建的打包布局，所以解码一个 token 要在整个缓存上做稠密注意力。改为聚集被选中的单元：97K 解码 59.0 → 49.3 ms/token，上下文斜率从每千 token 0.188 降到 0.067 ms。 |
| **可选的 Q8_0 K/V 缓存** | 稀疏注意力内核只读 f16，所以 Q8_0 缓存每个批次先反量化成它的布局：262144 token 时 K/V 缓存从 6 GB 降到 3.2 GB，解码不变，预填充慢 1–2%。默认仍是 f16。 |
| **MTP 推测解码调优** | 草稿头配自己的 IQ4_XS 输出投影，三个草稿 token，n-gram 草稿关闭。85K 解码 21.5 → 34.9 tok/s。 |
| **按并发调整草稿长度** | 这个 MoE 里每个待验证的草稿 token 都要多读约 10 个专家的权重，同时解码的几个对话无法共享。同时生成的对话越多，草稿越短（3，然后到四路为止是 2，五路起不起草）。同一步里各路草稿补齐到相同长度：长短不一的草稿会让混合记忆把一次验证拆成整个模型的好几次前向，三路 4K 上下文开草稿只有 34 tok/s、不开反而 46；现在开草稿也是 46。几路一起验证的一步（5-16 个 token）里，小乘积改走向量 kernel，不再用填不满的分块 kernel：0.2.3 起是 MoE 路由和被路由到的专家（9 token 一步 120-123 → 103-104 ms；同样草稿接受率下三路 +13%，四路现在也起草，+8%），0.2.4 起又加了三个（106 → 103 ms；三路 +2%，四路 +4.5%）。 |
| **输出不再依赖之前跑过什么** | 释放的 KV 单元会被清零：稀疏注意力的矩阵核求和曾在最末一位受先前对话或被拒草稿留在其中的数据影响。MTP 起草器携带的状态会随对话经过回退和 prompt 缓存：同一 prompt 发两次，草稿和回答都相同。 |
| **跟得上智能体的前缀缓存** | 递归状态只能从保留了检查点的位置接续，所以检查点现在放在后续提示会分叉的地方：系统提示结束处、每条用户消息处、每个提示的末尾。新会话用的系统提示如果别的槽已经算过，就在 GPU 上复制过来，不再重算；同时开始的几个会话会等第一个算完再复制。一个 agent 负载（10K token 系统提示、3 个会话同时开始、3 个子智能体）：处理量 73K → 29K token，提示耗时 172 → 60 秒。 |
| **多个智能体同时跑** | 智能体工具每个子智能体一个会话，共用一个很长的系统提示，于是每个槽看起来都和新会话 97% 相似，新会话会抢走得分最高的槽，截掉另一个智能体的历史。现在只有请求确实是某个槽里对话的延续时才用那个槽，默认同时保留 8 个对话。正在回答的对话不论长短共用一遍前向，新提示单独一遍；对话在回答时，每步最多处理 2048 个提示 token。六个智能体，0.2.6（默认 4 槽）对比 0.2.7（默认 8 槽）：处理量 75K → 45K token，首 token 时间中位数 10.1 → 3.4 秒，流式输出中最长的停顿 6.2 → 2.7 秒；一条 19K token 的新提示让正在输出的对话每次只停 2.2 秒，而不是 7 秒。 |
| **不存快照的回滚** | 推测验证每验一个 token 就保存一份递归状态（每层 3 MB），以备草稿被拒。现在只记录重算被拒部分所需的数据（每 token 33 KB），下一步再重放：三 / 四路同时时每个验证步缩短 2.8% / 4.0%，输出逐位不变。 |
| **delta net 状态九个 token 才写一次** | 四层里有三层为每个对话保存一份 3 MB 的状态，每个 token 都要读一遍、写一遍：八个对话时每层每个 token 要写 24 MB，比读还费时间。现在每个 token 只写 33 KB 的更新记录，攒满八条才把状态写回显存，用的是同样的记录和同样的运算；每个 warp 也改成一次读两列。八个约 20K 的对话、关闭 MTP：89.4 → 97.1 tok/s（0.4.0），输出逐位不变。 |
| **解码小内核和 BF16 副本** | 在 HIP 图里逐个派发打时间戳的剖析显示，每个 token 有约 3 ms 花在受延迟限制的小内核上：融合的逐元素链每次启动都要取一大段展开的通用寻址代码（8.4 → 2.7 us），top-k 没用 DPP。另有 264 个 F32 权重（路由、门控、注入）的值全是 bfloat16，换成 BF16 副本，乘积不变、读的字节减半。单个对话、关闭 MTP，3K 上下文 25.9 → 27.3 tok/s（0.4.0），输出不变。 |
| **采样时的 MTP** | 温度大于 0 时，草稿 token 只有在模型恰好抽中同一个 token 时才算数。现在草稿头按请求的采样设置从自己的分布里抽候选，再用推测采样逐个接受，接受的概率保证每个 token 仍然严格服从模型单独采样时的分布：温度 0.7 时第一个候选的保留率从 63% 提到 67%。保留率低，每步值得猜的就少，所以采样请求在一两个对话时猜 2 个、三四个对话时猜 1 个。Jan 的默认设置下，单个对话 33.6 → 36.2 tok/s，三个对话 61.3 → 67.1（0.4.1）。贪心解码的输出逐位不变。 |
| **三个正确性修复** | 推测验证批次跑了无 causal mask 的稠密注意力，长答案会跑偏并提前结束。图像输入曾以三种不同方式在 QSA 块机制里让服务端崩溃，第二个已加载的对话收到图片时还有第四种。 |
| **测量仪表** | 逐图、逐派发、逐阶段计时，全部默认关闭，需设环境变量才启用。 |

`patches/MANIFEST.md` 列出每个补丁、它动了哪些文件，以及必须的应用顺序。

## 构建怎么验证

**无法重放的补丁集会悄悄腐烂。** 两个工具保证这件事诚实：

```bash
python bootstrap/bootstrap.py --record    # 记录补丁集产出的哈希
python bootstrap/bootstrap.py --verify    # 当前树是否仍然完全一致
```

`--verify` 回答的是弱问题：这棵树是不是配方产出的？手改完再重新记录一遍哈希，它照样通过。强问题
是配方今天还能不能从零重建出这棵树，这有单独的工具：

```bash
python tools/replay_bootstrap.py          # 干净上游 + 补丁集 == 那 83 个文件，逐字节相同
```

它按 `bootstrap/UPSTREAM.json` 里记的 blob 哈希，从克隆自带的 git 对象里取回那 79 个文件的上游版
本——干净上游是重建出来的，不是信来的——然后重放整套快照和脚本再比对。结果是 **83 / 83**。

一开始并非如此。24 个里有 5 个没有任何脚本负责，其中就包括项目里最大的单项收益（干净重建会把它
静默丢掉）；另有 3 个脚本已经漂移到打不上去。`tools/make_patch_script.py` 从两棵树生成补丁脚本、
锚点自动扩展到唯一，这些缺口就是这样补上的——手写锚点正是最初腐烂的原因。

## 适用范围与限制

- **仅 Windows。** 管理器用 Win32 进程 API；构建目标是 gfx1151。
- **单一模型族。** 注意力部分专属于 `qwen4exp`。内核部分（IQ3_S MMB、MMVQ 每工作组行数）适用面更广。
- **不是 llama.cpp 的分叉。** 本仓库只放补丁、工具和实测数据，不含模型权重、上游源码或二进制。

## 目录结构

```
bootstrap/   拉取、打补丁、构建运行时；UPSTREAM.json 是这份 delta 的哈希记录
patches/     补丁集、清单，以及两处大到无法用锚点表达的整文件替换
tools/       管理器、测量工具、验证工具
integrations/jan/   可选的 Jan 管理页面
docs/        install.md · results.md · measuring.md · decode-budget.md · dead-ends.md
```

在相信本仓库任何数字（包括上面那几个）之前，先读 [docs/measuring.md](docs/measuring.md)——那是一份
这个项目把自己测错的方式清单。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方署名见 [NOTICE.md](NOTICE.md)——特别是本项目修改的是
[pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp)（MIT），Jan 覆盖层用于
[Jan](https://github.com/menloresearch/jan) 的源码树（Apache-2.0）。

如果这些补丁对你的项目有用，欢迎注明出自 Rulith Inference，并附上[本仓库](https://github.com/rulith-dev/rulith-inference)的链接。这是请求，不是许可条款。

---

Rulith Inference 由 [Rulith](https://rulith.ai) 开发。Rulith 做面向 AI 智能体的可验证执行基础设施。
