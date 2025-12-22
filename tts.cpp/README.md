# tts.cpp

tts.cpp 是一个基于 ggml 的 文字转语音推理库，用于在本地设备上运行开源的文本转语音（TTS）模型

本仓库在原项目的 Kokoro 路线上做了中文增强，重点解决了「中文提示词为空 / 无法发声」的问题，并支持中英文混合输入。

## 模型与功能支持概览

### 模型

推荐优先使用 **Kokoro**（体积小、速度快、效果稳定）。

| 模型 | CPU | vulkan | 量化/低精度 | 语言 |
| --- | --- | --- | --- | --- |
| Kokoro | 支持 | 支持 | 支持 | 支持中文/中英混合/英语/日语 |


- aimax395 cpu下生成30s语音耗时 15s
- aimax395 gpu下生成30s语音耗时 3s

### 音色

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

## Kokoro 模型转换（Torch -> GGUF）

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
  --phonemizer-repo D:\EVA_MODELS\text2speech\TTS_ipa_en_us_phonemizer_F16.gguf
```

`--phonemizer-repo` 用于指定 phonemizer 配置来源（本地 `.gguf` 文件/目录或 HF repo id），转换脚本会将 `phonemizer.*` 相关键值拷贝进新 GGUF，运行时无需额外 phonemizer 依赖。可以在这里下载 https://hf-mirror.com/mmwillet2/TTS_ipa_en_us_phonemizer/tree/main


- 例如 python .\py-gguf\convert_kokoro_to_gguf  --repo-id .\参考项目\kokoro-zh  --save-path D:\EVA_MODELS\text2speech\Kokoro-82M-v1_1-zh2.gguf --phonemizer-repo "D:\原始模型\kokoro-zh\tts_en_us_phonemizer.gguf"


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

## 使用说明

更多 CLI 参数可参考：`examples/cli/README.md`。

## 鸣谢

- [kokoro](https://github.com/hexgrad/kokoro)
- [TTS.cpp](https://github.com/mmwillet/TTS.cpp)
- [ggml](https://github.com/ggml-org/ggml)
