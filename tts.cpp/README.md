# TTS.cpp（中文说明）

TTS.cpp 是一个基于 GGML 的 C/C++ 推理库，用于在本地设备上运行开源文本转语音（TTS）模型，目标是提供类似 `whisper.cpp` / `llama.cpp` 的轻量、可移植推理体验。

本仓库在 Kokoro 路线上做了中文增强，重点解决了「中文提示词为空 / 无法发声」的问题，并支持中英文混合输入。

- 路线图：`https://github.com/users/mmwillet/projects/1`
- GGML 分支/补丁背景：`https://github.com/mmwillet/ggml/tree/support-for-tts`

## 中文支持（Kokoro）

### 1) 不依赖 eSpeak 的中文

Kokoro 的部分 GGUF（例如 “no_espeak” 版本）原本会在遇到 CJK 字符时丢弃文本，导致 `Got empty response for prompt`。

本仓库已在 Kokoro 前端加入内置普通话处理：
- 中文输入不再被吞掉
- 可直接生成中文语音
- 不需要安装/链接 eSpeak-ng

### 2) 中英混合

同一条 prompt 支持混合输入（例如 `你好 hello`、`hello 你好`）。实现思路是按片段拆分：
- ASCII 段落走现有英文字素化（IPA）
- 非 ASCII 段落走内置中文前端

### 3) Windows 友好

Windows 下 `tts-cli` 已确保命令行参数按 UTF-8 处理，中文参数不会因代码页导致解析失败。

## 模型与功能支持概览

### 模型

推荐优先使用 **Kokoro**（体积小、速度快、效果稳定）。

| 模型 | CPU | Metal | 量化/低精度 | 备注 |
| --- | --- | --- | --- | --- |
| Kokoro | 支持 | 暂不支持 | 支持（但对体积影响有限，见下文） | 推荐；支持中文/中英混合 |
| Parler TTS (Mini/Large) | 支持 | 支持 | 支持 | 条件提示词等玩法更丰富但更重 |
| Dia | 支持 | 支持 | 支持 | 需按推荐采样参数使用 |
| Orpheus | 支持 | 暂不支持 | 暂不支持 | 依模型而定 |

### 语音（voice）

Kokoro 的 voice pack 会被 **直接打包进 GGUF 文件**（转换阶段读取 `voices/*.pt`，运行时不需要再带 `voices/` 目录）。

用户可通过 CLI 查询可用 voice：

```powershell
.\tts-cli.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_F16.gguf --list-voices
```

生成时通过 `--voice` 选择：

```powershell
.\tts-cli.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_F16.gguf --voice zf_001 -p "你好，欢迎使用 EVA" -sp out.wav
```

## 快速开始（生成中文/中英混合）

### 中文

```powershell
.\tts-cli.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_F16.gguf --voice zf_001 -p "你好"
```

### 中英混合

```powershell
.\tts-cli.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_F16.gguf --voice zf_001 -p "你好 hello"
```

## Kokoro 中文模型转换（Torch -> GGUF）

转换脚本位于 `py-gguf/`，核心入口：`py-gguf/convert_kokoro_to_gguf`。

### 1) Python 依赖（仅 Kokoro，CPU 版 torch）

建议用虚拟环境，且使用清华源：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  "numpy<2" sentencepiece `
  torch==2.4.1 torchaudio==2.4.1 `
  gguf==0.10.0 kokoro==0.9.4 `
  huggingface-hub transformers safetensors
```

说明：`gguf==0.10.0` 目前与 `numpy>=2` 不兼容，因此需要固定 `numpy<2`。

### 2) 从本地目录转换（推荐）

假设本地模型目录结构类似：
- `config.json`
- `*.pth`（checkpoint）
- `voices/*.pt`

```powershell
python .\py-gguf\convert_kokoro_to_gguf `
  --repo-id .\Kokoro-82M-v1___1-zh `
  --save-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh.gguf `
  --tts-phonemizer `
  --phonemizer-repo D:\EVA_MODELS\text2speech\Kokoro_no_espeak_F16.gguf
```

`--tts-phonemizer` 表示不使用 eSpeak（避免 GPL 依赖），并把英文字素化所需的键值拷贝进新 GGUF（`--phonemizer-repo` 支持填写本地 `.gguf` 文件或 HF repo id）。

## 量化/低精度（Kokoro）

Kokoro 可用 `examples/quantize` 生成 F16/Q8/Q4 等版本：

```powershell
.\quantize.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh.gguf `
  --quantized-model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_F16.gguf `
  --quantized-type F16 --convert-non-quantized-to-f16

.\quantize.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh.gguf `
  --quantized-model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_Q8_0.gguf `
  --quantized-type Q8_0 --convert-non-quantized-to-f16

.\quantize.exe --model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh.gguf `
  --quantized-model-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh_Q4_0.gguf `
  --quantized-type Q4_0 --convert-non-quantized-to-f16
```

注意：Kokoro 当前量化策略只覆盖部分张量，且 `voice_tensors` 不参与量化（会保留 FP32），所以 Q8/Q4 相比 F16 的体积下降幅度有限。

## 构建

如果你只需要使用 CLI，可直接构建项目并得到 `tts-cli`：

```powershell
cmake -B build
cmake --build build --config Release
```

如需 eSpeak-ng（可选，不推荐给需要避免 GPL 的场景）：

```bash
export ESPEAK_INSTALL_DIR=/absolute/path/to/espeak-ng/dir
cmake -B build
cmake --build build --config Release
```

## 使用说明

更多 CLI 参数可参考：`examples/cli/README.md`。

## 许可协议

默认情况下，本仓库为 `MIT` 许可。

按模型原实现要求，部分超参数/后处理逻辑可能保留原模型的 `Apache-2.0` 许可约束（不包含 ggml/C++ 移植代码）。

如果启用 eSpeak-ng 支持，最终二进制可能会受到 `GPL-3.0-or-later` 许可影响；如需规避 GPL，请优先使用 `--tts-phonemizer` 方案与本仓库内置中文前端。
