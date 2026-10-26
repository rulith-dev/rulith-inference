# 安装包上手指南

*[English](getting-started.md)*

这一页讲的是用发布的安装包运行 Rulith Inference。这条路不需要 Python、ROCm、Visual Studio，也不需要源码：安装包自带整套
运行时。自己编译是另一页：[install.md](install.md)。

全部流程是：装应用，在应用里下载一个模型，点"加载模型"，开聊。

## 需要什么

| | |
|---|---|
| 机器 | **Ryzen AI Max+ 395**：Radeon 8060S，`gfx1151`，128 GB 统一内存。自带的内核只为 gfx1151 编译，其他 GPU 上模型加载不起来。 |
| GPU 显存 | **64 GB** 或 **96 GB**，在 BIOS 里设置（选项一般叫 *UMA Frame Buffer Size* 或"专用显存"）。64 GB 时放不下的专家权重由应用放进系统内存，GPU 读取速度相同；96 GB 给对话缓存留的空间更大。 |
| 系统 | Windows 11 |
| 磁盘 | 模型文件 83-112 GB，取决于选哪个文件（推荐的那个 104 GB），应用 0.5 GB |

## 1. 装应用

从 [Releases 页面](https://github.com/rulith-dev/rulith-inference/releases)下载 `Rulith-Inference_<版本>_x64-setup.exe`，
运行。安装到此为止：应用本体、用我们的 fork 编译的 `llama-server`、它需要的 ROCm 库、进程管理器和它自己的 Python 都在里面。

应用会自己更新：启动时检查本项目在 GitHub 上的新版本，有新版本时弹出提示，侧栏底部也会显示**新版本**。下载的安装包只有在
签名与应用内置的公钥校验通过后才会安装；安装前会先卸载模型，因为更新会替换运行时。聊天记录和设置都保留。手动升级时直接用
新安装包覆盖安装。

## 2. 下载模型

打开**模型 › 下载**。*推荐*里列出的是 Rulith Inference 专门调优、所有公布数字都用它们测量的文件：

| 文件 | 大小 | | |
|---|---:|---|---|
| **Qwen3.8-Flash-Next EXL3 4.05 bpw** | 104.3 GB | 推荐：最接近完整模型 | [Rulith](https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF) |
| Qwen3.8-Flash-Next EXL3 3.05 bpw | 82.4 GB | 更小更快，KL 散度是 4.05 的两倍 | [Rulith](https://huggingface.co/Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF) |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 111.3 GB | | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| Qwen3.8-Flash-Next UD-IQ4_XS | 93.7 GB | | [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |

选一个；*下载选项*在同一仓库的不同版本间切换。*配套文件*是 MTP 草稿（推测解码：回答快约 1.5-2 倍）、视觉投影模型（图像
输入），以及 Unsloth 文件用的 Rulith 草稿头（起草更快）；保持打开即可。**下载**会询问来源（Hugging Face 或国内镜像）和
文件夹，并显示那里的剩余空间。每个文件下完都会校验 SHA-256；下载可以暂停，之后从断点继续，重启后也可以。下完后模型已经
在模型库里，点**立即加载**即可加载。

搜索框可以在 Hugging Face 上找其他 GGUF 仓库。Rulith Inference 有专用内核的文件标为*已调优*；其他模型未经 Rulith 测试，
走 llama.cpp 原版内核。

### 已经有的文件

应用在原地读取模型文件，从不复制。装了 LM Studio 的话，它的下载文件夹会自动登记。否则在**模型 › 模型库 › 模型文件夹**里
用**浏览…**或粘贴路径加上上级文件夹（例如 `D:\models`），点**保存并扫描**；文件夹会递归扫描。模型的草稿、投影模型和草稿头
要和它的第一个分片放在同一个文件夹里，并保持原名：应用按名字配对。

也可以用 Hugging Face 命令行手动下载（`pip install huggingface_hub`；用镜像的话先设 `HF_ENDPOINT=https://hf-mirror.com`）：

```bash
hf download Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF --local-dir D:\models\rulith --include "EXL3-4.05bpw/*" "mtp-*.gguf" "mmproj-F16.gguf"
```

命令行会保留仓库里的目录结构，所以下完后把 `mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf` 和 `mmproj-F16.gguf` 移进
`EXL3-4.05bpw\`，放在分片旁边。

## 3. 加载，开聊

在欢迎页、**模型 › 模型库**里模型那一行，或侧栏底部的模型服务面板上点**加载模型**。不用先设任何东西：默认值就是所有公布
数字的测量配置——上下文 262144、8 个对话槽、Flash Attention、稀疏注意力、MTP 草稿和图像输入全开，思考深度"中"。
**模型 › 日志**里能看到加载进度。第一次加载要从磁盘读整个文件，之后会快一些。

加载好后聊天页会自动打开，模型已经选好，直接打字。附一张图片即可针对图片提问。除非**模型 › 配置 › 启动**里设置了，应用启动
时不会自己加载模型。

EXL3 4.05 bpw 的实测预期（0.6.9，64 GB 档）：

| | |
|---|---|
| 单个对话 | 解释说明 58 tok/s，写代码 70，散文 46（开 MTP；速度取决于文本有多好预测） |
| 86K token 上下文之后 | 46 tok/s；长提示词读取约每秒 1,330 token |
| 8 个对话同时 | 合计约 85 tok/s。智能体工具同时运行几个会话就需要几个槽位；*并发对话数*在模型 › 配置 › 并发里 |

打开**模型 › 配置**里的"把对话保存在磁盘上"（默认关闭）后，过一阵子回到某个对话时不会重新处理它：对话状态存在磁盘上（每
token 约 30 KB，最多 16 GB），回来时一秒左右读回。

如果加载时报内存不足，说明这个配置放不下：去 BIOS 加大 GPU 显存、在模型 › 配置里调小上下文长度，或者给 Windows 更大的
页面文件（内存加页面文件的总量通常最先用完，见 [install.md](install.md)）。

模型加载着的时候，`http://127.0.0.1:8080/v1` 也是一个兼容 OpenAI 与 Anthropic 接口的服务，其他客户端（包括编程智能体）
都能接；Codex 和 Claude Code 的设置见 [README](../README.zh.md#给其他程序用)。模型 › 配置 › 网络可以改端口、让局域网里的
其他设备使用它，并为它们设置 API 密钥；模型 › 日志显示正在用的地址。

## 出问题了

- **模型库是空的。** 检查"模型文件夹"：只列出已登记文件夹下的 `.gguf`，分片模型按 `-00001-of-` 那个分片显示。点"重新
  扫描"。
- **MTP 或图像输入是关的，模型 › 配置里说没找到文件。** 草稿或投影模型不在模型文件夹里（或改过名）。补上并重新扫描，开关
  会自己打开。只有当保存过的配置里手动开了开关、而文件后来没了，加载才会因此拒绝启动。
- **开始加载了，然后挂掉。** 看模型 › 日志。常见原因：GPU 不是 gfx1151，或者 GPU 显存、页面文件对这个上下文长度不够。
- **8080 端口被占用。** 本应用早前实例启动的 `llama-server` 还在跑；模型页面会认领它并可以卸载。别的程序占了这个端口就得
  手动停掉，或者在模型 › 配置 › 网络里给服务器换一个端口。
- **东西都在哪。** 应用是 `%LOCALAPPDATA%\Rulith Inference\rulith-inference.exe`。0.6.1 及以前，改名前安装的一直留在
  `%LOCALAPPDATA%\Strix Llama`，程序叫 `strixllama.exe`；0.6.2 的安装程序会把那个文件夹原地改名（有程序占用其中文件时保持
  原位），并把快捷方式和任务栏固定改指过去；如果任务栏图标仍显示空白，重新固定一次即可。管理器的设置、模型目录缓存和提示
  缓存在它下面的 `runtime\config\jan\`，服务日志在 `runtime\logs\`（只保留最近 10 次运行、总共不超过 4 GB；单次运行的
  控制台日志超过 1 GB 时截回到加载完成处，请求日志只保留最近两个 1 GB 文件）；聊天记录在 `%APPDATA%\strixllama\data`。
  卸载会删掉应用；勾选"删除应用数据"时设置和提示缓存一并删除，聊天记录无论如何都保留。
