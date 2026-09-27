# 安装包上手指南

*[English](getting-started.md)*

这一页讲的是用发布的安装包运行 Strix Llama。这条路不需要 Python、ROCm、Visual Studio，也不需要
源码——安装包自带整套运行时。自己编译是另一页：[install.md](install.md)。

全部流程是：装应用，把五个文件下载到同一个文件夹，把应用指向这个文件夹，点"加载模型"，开聊。

## 需要什么

| | |
|---|---|
| 机器 | **Ryzen AI Max+ 395**——Radeon 8060S，`gfx1151`，128 GB 统一内存。自带的内核只为 gfx1151 编译，其他 GPU 上模型加载不起来。 |
| 显存划分 | **96 GB**，在 BIOS 里设置（选项一般叫 *UMA Frame Buffer Size* 或"专用显存"）。64 GB 下 93.7 GB 的模型会溢出到共享内存，解码慢 28%，没有任何软件设置能补回来。 |
| 系统 | Windows 11 |
| 磁盘 | 模型文件约 100 GB，应用 0.5 GB |
| 下载 | 五个文件，约 97 GB，来自 Hugging Face |

## 1. 装应用

从 [Releases 页面](https://github.com/rulith-dev/strixllama/releases)下载
`Strix-Llama_<版本>_x64-setup.exe`，运行。安装到此为止：应用本体、用我们的 fork 编译的
`llama-server`、它需要的 ROCm 库、进程管理器和它自己的 Python 都在里面，不用再装任何东西。

从 0.2.5 起应用会自己更新：启动时检查本项目在 GitHub 上的新版本，有新版本时会弹出提示，侧栏底部也会
显示**新版本**。下载的安装包只有在签名与应用内置的公钥校验通过后才会安装；安装前会先卸载模型，因为更新
会替换运行时。聊天记录和设置都保留。手动升级（包括从 0.2.4 及更早版本升级）时直接用新安装包覆盖安装；
访问不了 GitHub 时，请自行下载安装包。

## 2. 下载模型文件——全部放进同一个文件夹

应用要加载三类文件，第二、三类是**按文件名在第一类旁边找**的，所以五个文件必须在同一个文件夹里，
文件名保持原样。

推荐用 LM Studio 的目录布局：

```
D:\models\unsloth\Qwen3.8-Flash-Next-GGUF\
```

如果 LM Studio 已经下载过这个模型，大部分文件你已经有了，可以直接跳到下一节——应用会原地读
LM Studio 的目录，不复制。

五个文件都来自同一个 Hugging Face 仓库
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)。下面的链接和
命令都固定在它的 `38bb39e` 版本上，每个文件都对这个版本核对过 sha256（2026-09-26）：

| # | 仓库内路径 | 大小 | 作用 |
|---|---|---|---|
| 1 | [`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf) | 11 MB | 模型分片 1/3（元数据） |
| 2 | [`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf) | 49.8 GB | 模型分片 2 |
| 3 | [`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf) | 43.8 GB | 模型分片 3 |
| 4 | [`MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf) | 1.9 GB | MTP 草稿：推测解码，解码速度约 +60% |
| 5 | [`mmproj-F16.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf) | 904 MB | 视觉投影：图像输入 |

**为什么正好是这几个。** 本项目公布的所有数字都是在 `UD-IQ4_XS` 这个量化上测的；同一模型的其他
量化也能加载，但没有针对它们调过、也没验证过。第 1–3 个是模型本体，必须有。第 4、5 个是可选的：
草稿和投影在文件夹里就自动开启，不在就保持关闭，"模型配置"页会说明没找到哪一个。没有草稿，
解码慢约 40%。（第 4 个换成 `mtp-…-shared-Q8_0.gguf` 也行，两者相差不到 1%。）

**再加一个，可选：草稿头**——[`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`](https://github.com/rulith-dev/strixllama/releases/download/v0.1.2/mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf)，349 MB，在本项目的
[Releases 页面](https://github.com/rulith-dev/strixllama/releases)。放到第 4 个文件所在的同一个
文件夹。下次重新扫描时应用会把两者合并成 `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf`
（几秒钟，只做一次）并优先使用：英文解码快 5%，中文快 9%，接受率不变，下面的数字就是这个配置测的。
见[第 5 节](#5-可选更快的草稿头解码-5–9)。

### 怎么下

用 [aria2](https://aria2.github.io/) 在 PowerShell 里跑——五个文件平铺进一个文件夹，断点续传，
这么大的文件我们知道的最快办法：

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
# 可选：草稿头（第 5 节）
aria2c -x 8 -s 8 -c -d $dir "https://github.com/rulith-dev/strixllama/releases/download/v0.1.2/mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf"
```

国内访问 Hugging Face 慢或不通的话用镜像：把 `$base` 里的 `huggingface.co` 换成 `hf-mirror.com`。

用 Hugging Face 命令行也可以（`pip install huggingface_hub`）：

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --local-dir D:\models\unsloth\Qwen3.8-Flash-Next-GGUF \
  --include "UD-IQ4_XS/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" "mmproj-F16.gguf"
```

这种方式会保留仓库的子目录（`UD-IQ4_XS\`、`MTP\`），下完之后**把那四个文件上移**到
`mmproj-F16.gguf` 所在的文件夹。用镜像时先设 `HF_ENDPOINT=https://hf-mirror.com`。
（旧版本的命令叫 `huggingface-cli download`。）

用浏览器也行：在仓库页面逐个打开这五个路径下载，全部存到同一个文件夹。

## 3. 告诉应用文件夹在哪

打开应用，欢迎页会查找模型文件。

- 装了 LM Studio 的话，它的下载目录会自动登记，模型已经能找到。
- 否则点**模型文件夹**，用**浏览…**选中或直接粘贴上级目录——`D:\models`——再点**保存并扫描**。
  文件夹是递归扫描的，填上级目录就够。

侧栏**模型 › 模型库**里是同一份列表：模型显示成一行，*IQ4_XS · 93.7 GB · 3 个分片*。显示"文件不完整"
说明缺分片。列表有缓存：加了或移动了文件之后点**重新扫描**。把文件类型筛选切到 *MTP 草稿* 或
*视觉投影*，可以确认第 4、5 个文件被识别到了。

## 4. 加载，开聊

在欢迎页点**加载模型**（或在**模型 › 模型库**里模型那一行点）。不用先设任何东西：默认值就是所有
公布数字的测量配置——上下文 262144、batch 和 ubatch 8192、Flash Attention、稀疏注意力、MTP 草稿、
图像输入全开，思考深度"中"。**模型 › 日志**里能看到加载进度。第一次加载要从磁盘读 93.7 GB，之后会快
一些。

加载好后聊天页会自动打开，模型已经选好，直接打字。图像输入就是在消息里附一张图。应用启动时不会自己
加载模型：在聊天页上方的提示条或侧栏的"模型服务"面板里点**加载模型**；想让它启动时自动加载上次用的
模型，可以打开**模型 › 配置 › 启动**。加载期间聊天页上方也有提示。侧栏底部的"模型服务"面板在每一页都显示同样的状态和按钮。

在参考机器上的实测预期：

| | |
|---|---|
| 短上下文 | 约 41 tok/s |
| 86K token 上下文 | 约 33 tok/s；预填充约 1180 t/s |
| 三 / 四路同时 | 合计约 54 / 61 tok/s；"模型 › 配置 › 并发"里的并发对话数默认为 8，每个槽位都会保留自己的对话。智能体工具同时运行几个会话，就需要几个槽位 |

打开**模型 › 配置**里的"把对话保存在磁盘上"开关（默认关闭）后，过一阵子回到某个对话时不会重新处理它的提示词：对话状态会存在
磁盘上（每 token 约 30 KB，最多 16 GB），回来时一秒左右读回。

如果加载时报内存不足，说明这个配置放不下：去 BIOS 加大显存划分、在**模型 › 配置**里调小上下文长度，或者给
Windows 更大的页面文件（内存加页面文件的总量通常最先用完，见 [install.md](install.md)）。

模型加载着的时候，`http://127.0.0.1:8080/v1` 也是一个普通的 OpenAI 兼容接口，其他客户端都能接。

## 5. 可选：更快的草稿头（解码 +5–9%）

上面的解码数字是用一个自带 IQ4_XS 输出投影的草稿文件测的，它不再借用模型的输出层，英文快 5%，
中文快 9%，接受率不变。这一页其他所有内容不需要它。

Unsloth 的 `shared-*` 草稿没有自己的输出投影，每一步草稿都要去读模型那份 521 MB 的。这个投影只是
一张张量，所以单独发布：`mtp-Qwen3.8-Flash-Next-head-iq4_xs.gguf`，349 MB，在
[Releases 页面](https://github.com/rulith-dev/strixllama/releases)。下载到模型文件夹里、第 4 个文件
旁边，然后在**模型 › 模型库**里点**重新扫描**。应用会在旁边写出
`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf`（两个文件逐字节拼接，几秒钟，只做一次；这是
应用唯一会写进模型文件夹的文件），弹一条提示，之后所有没手动选过草稿的配置都用它。**模型 › 配置**的
草稿列表里三个都能看到。

想自己生成草稿头，需要带工具链的源码树：[install.md 第 5 节](install.md#5-model-files)。

## 出问题了

- **模型库是空的。** 检查"模型文件夹"：只列出已登记文件夹下的 `.gguf`，分片模型按 `-00001-of-` 那个
  分片显示。点"重新扫描"。
- **MTP 或图像输入是关的，模型 › 配置里说没找到文件。** 第 4 或第 5 个文件不在模型文件夹里（或改过名）。
  补上并重新扫描，开关会自己打开。只有当保存过的配置里手动开了开关、而文件后来没了，加载才会因此
  拒绝启动。
- **开始加载了，然后挂掉。** 看**模型 › 日志**。常见原因：GPU 不是 gfx1151，或者显存划分对这个上下文长度
  或页面文件不够。
- **8080 端口被占用。** 本应用早前实例启动的 `llama-server` 还在跑；模型页面会认领它并可以卸载。
  别的程序占了这个端口就得手动停掉。
- **东西都在哪。** 应用在 `%LOCALAPPDATA%\Strix Llama`；管理器的设置、模型目录缓存和最近一次服务日志在
  它下面的 `runtime\config\jan\`；聊天记录在 `%APPDATA%\strixllama\data`。卸载会删掉应用文件夹连同
  这些设置，聊天记录保留。
