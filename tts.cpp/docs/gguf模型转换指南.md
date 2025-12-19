# 原始权重 → GGUF 模型转换指南（tts.cpp）

本文档说明如何将 **原始权重（通常为 Hugging Face 上的 PyTorch 权重 / safetensors / checkpoint）转换为本仓库可直接推理的 GGUF**。

本仓库的转换脚本位于：`py-gguf/`（无 `.py` 后缀的可执行 Python 脚本）。

---

## 1. 环境准备

### 1.1 Python 与依赖

建议使用 Python 3.9~3.11，并在仓库根目录创建虚拟环境（示例以 PowerShell 为例）：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\py-gguf\requirements.txt
```

> 说明：若 `--repo-id` 指向 Hugging Face 仓库，将会联网下载权重；若指向本地目录，则可离线转换。

### 1.2 输出精度与后续量化

- `py-gguf/*_to_gguf` 转换脚本 **默认输出 FP32 GGUF**（便于后续量化/兼容）。
- 若需要生成 `F16/Q4_0/Q5_0/Q8_0` 等版本，请使用编译产物 `quantize.exe`：
  - 单配置（MinGW/Ninja）：`build/bin/quantize.exe`
  - 多配置（MSVC）：`build/bin/Release/quantize.exe`（或 Debug）

Kokoro 量化通常建议带上 `--convert-non-quantized-to-f16`，将不支持量化的张量降为 FP16（避免混用 FP32）：

```powershell
.\build\bin\quantize.exe `
  --model-path <fp32.gguf> `
  --quantized-model-path <out.gguf> `
  --quantized-type Q4_0 `
  --convert-non-quantized-to-f16
```

---

## 2. Kokoro（含中文/英文）

脚本：`py-gguf/convert_kokoro_to_gguf`

### 2.1 直接从 Hugging Face 转换（默认）

```powershell
python .\py-gguf\convert_kokoro_to_gguf `
  --save-path .\build\kokoro-f32.gguf
```

常用参数：
- `--repo-id <repo>`：指定模型来源（Hugging Face repo id 或本地目录）。
- `--voices v1,v2,...`：指定要打包进 GGUF 的 voice 列表；不传则自动检测。
- `--phonemizer-repo <repo-or-path>`：用于拷贝 phonemizer 相关 GGUF 配置（`phonemizer.*` 键值）的来源（HF repo id 或本地 GGUF 文件/目录）。

### 2.2 使用本地目录（离线）转换

当你有一个本地 Kokoro 目录（例如 `kokoro-zh/` 或 `kokoro-en/`），目录内通常包含：
- `config.json`
- checkpoint（例如 `*.pth`）
- `voices/*.pt`

可以直接把本地目录路径作为 `--repo-id`：

```powershell
$voices = (Get-ChildItem .\kokoro-zh\voices -Filter *.pt | ForEach-Object { $_.BaseName }) -join ','
python .\py-gguf\convert_kokoro_to_gguf `
  --repo-id .\kokoro-zh `
  --save-path .\build\kokoro-zh-f32.gguf `
  --voices $voices
```

更多 Kokoro 的实操命令可参考：
- `docs/kokoro_en_conversion.md`
- `docs/kokoro_zh_conversion.md`

---

## 3. Parler TTS（mini/large）与 T5 Encoder（可选）

### 3.1 Parler TTS（生成模型）

脚本：`py-gguf/convert_parler_tts_to_gguf`

```powershell
python .\py-gguf\convert_parler_tts_to_gguf `
  --save-path .\build\parler-mini-f32.gguf `
  --voice-prompt "female voice"
```

切换 large：

```powershell
python .\py-gguf\convert_parler_tts_to_gguf `
  --save-path .\build\parler-large-f32.gguf `
  --large-model `
  --voice-prompt "female voice"
```

### 3.2 T5 Encoder（用于“可变条件 prompt”的条件生成）

脚本：`py-gguf/convert_t5_encoder_to_gguf`

```powershell
python .\py-gguf\convert_t5_encoder_to_gguf `
  --save-path .\build\t5-encoder-f32.gguf
```

---

## 4. Dia

脚本：`py-gguf/convert_dia_to_gguf`

```powershell
python .\py-gguf\convert_dia_to_gguf `
  --save-path .\build\dia-f32.gguf
```

---

## 5. Orpheus（含 SNAC 音频解码器）

脚本：`py-gguf/convert_orpheus_to_gguf`

```powershell
python .\py-gguf\convert_orpheus_to_gguf `
  --save-path .\build\orpheus-f32.gguf
```

可选参数：
- `--repo-id <repo>`：Orpheus 主模型来源（HF repo id 或本地目录）。
- `--snac-repo-id <repo>`：SNAC 音频解码器来源（HF repo id 或本地目录）。

---

## 6. 常见坑（建议先看）

1) `quantize` 输入必须是 **FP32 GGUF**
- 例如 Kokoro 若你手里只有 `*_F16.gguf`，请先用原始权重跑 `py-gguf` 导出 FP32，再量化。

2) 联网下载失败
- 直接把已下载好的模型目录作为 `--repo-id`（脚本支持本地目录），即可离线转换。
