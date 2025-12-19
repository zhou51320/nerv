# Kokoro English GGUF Conversion Cheatsheet

This note documents the exact commands used to turn the locally prepared `kokoro-en/` checkout into GGUF models and to produce several quantized variants. All commands are meant to be executed from the repository root (`C:\Users\32138\Desktop\TTS.cpp`) with the existing virtual environment already created at `.venv`.

## 1. Activate the Conversion Environment

```powershell
.\.venv\Scripts\activate
```

## 2. Generate the FP32 GGUF from the Local `kokoro-en/` Folder

```powershell
$voices = (Get-ChildItem kokoro-en\voices -Filter *.pt | ForEach-Object { $_.BaseName }) -join ','
python .\py-gguf\convert_kokoro_to_gguf `
  --repo-id kokoro-en `
  --save-path build\kokoro-en-f32.gguf `
  --voices $voices
```

The patched converter auto-detects that `--repo-id` points at a local directory and loads `config.json`, the checkpoint (`kokoro-v1_0.pth`), and the listed voice embeddings directly from disk.

## 3. Produce Quantized Variants

The bundled `quantize.exe` supports Kokoro with the `--convert-non-quantized-to-f16` flag so that tensors which cannot be quantized are down-cast to FP16. Run the following commands (each can be executed independently):

```powershell
.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-en-f32.gguf `
  --quantized-model-path build\kokoro-en-f16.gguf `
  --quantized-type F16 `
  --convert-non-quantized-to-f16

.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-en-f32.gguf `
  --quantized-model-path build\kokoro-en-q8_0.gguf `
  --quantized-type Q8_0 `
  --convert-non-quantized-to-f16

.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-en-f32.gguf `
  --quantized-model-path build\kokoro-en-q5_0.gguf `
  --quantized-type Q5_0 `
  --convert-non-quantized-to-f16

.\build\bin\Release\quantize.exe `
  --model-path build\kokoro-en-f32.gguf `
  --quantized-model-path build\kokoro-en-q4_0.gguf `
  --quantized-type Q4_0 `
  --convert-non-quantized-to-f16
```

## 4. Resulting Artifacts (current sizes)

| File | Approx. Size |
| --- | --- |
| `build\kokoro-en-f32.gguf` | ~336 MB |
| `build\kokoro-en-f16.gguf` | ~194 MB |
| `build\kokoro-en-q8_0.gguf` | ~180 MB |
| `build\kokoro-en-q5_0.gguf` | ~174 MB |
| `build\kokoro-en-q4_0.gguf` | ~172 MB |

Keep this sheet with the repository; future refreshes only need to replace the source files under `kokoro-en\` and rerun the same commands.
