# Kokoro 中文模型（kokoro-zh）GGUF 转换说明

以下步骤在仓库根目录 `C:\Users\32138\Desktop\TTS.cpp` 中执行，并假设已经创建并安装好了虚拟环境 `.venv`。

## 1. 启用虚拟环境

```powershell
.\.venv\Scripts\activate
```

## 2. 统计本地语音包并导出 FP32 GGUF

```powershell
$voices = (Get-ChildItem kokoro-zh\voices -Filter *.pt | ForEach-Object { $_.BaseName }) -join ','
python .\py-gguf\convert_kokoro_to_gguf `
  --repo-id kokoro-zh `
  --save-path build\kokoro-zh-espeak-f32.gguf `
  --voices $voices
```

> 说明：打过补丁的转换脚本会检测 `--repo-id` 指向的是本地文件夹，于是直接读取 `config.json`、`kokoro-v1_1-zh.pth` 以及 `voices/*.pt`，无需访问 Hugging Face。

## 3. 生成常用量化版本

对 Kokoro 来说，量化时需要带上 `--convert-non-quantized-to-f16`，避免不支持量化的张量仍然保持 FP32。

```powershell
.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-zh-espeak-f32.gguf `
  --quantized-model-path build\kokoro-zh-espeak-f16.gguf `
  --quantized-type F16 `
  --convert-non-quantized-to-f16

.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-zh-espeak-f32.gguf `
  --quantized-model-path build\kokoro-zh-espeak-q8_0.gguf `
  --quantized-type Q8_0 `
  --convert-non-quantized-to-f16

.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-zh-espeak-f32.gguf `
  --quantized-model-path build\kokoro-zh-espeak-q5_0.gguf `
  --quantized-type Q5_0 `
  --convert-non-quantized-to-f16

.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-zh-espeak-f32.gguf `
  --quantized-model-path build\kokoro-zh-espeak-q4_0.gguf `
  --quantized-type Q4_0 `
  --convert-non-quantized-to-f16
```

## 4. 当前产物体积

| 文件 | 约尺寸 |
| --- | --- |
| `build\kokoro-zh-espeak-f32.gguf` | ~361 MB |
| `build\kokoro-zh-espeak-f16.gguf` | ~219 MB |
| `build\kokoro-zh-espeak-q8_0.gguf` | ~204 MB |
| `build\kokoro-zh-espeak-q5_0.gguf` | ~198 MB |
| `build\kokoro-zh-espeak-q4_0.gguf` | ~196 MB |

如需更新模型，只需要用新的 `kokoro-zh` 目录替换原始文件，再重复以上步骤即可。
