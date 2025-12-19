# perf_battery（benchmark）

## 概述

`perf_battery` 提供一套固定的基准用例（中文/英文/中英混合），用于测量同一模型在不同推理后端或不同版本下的耗时差异。  
默认只需指定模型路径，输出会直接给出每个用例的平均耗时与 RTF（Real Time Factor）。

## 构建

```powershell
cmake -B build
cmake --build build --config Release
```

## 使用

```powershell
.\perf_battery.exe --model-path D:\models\Kokoro-82M-v1_1-zh_F16.gguf

.\perf_battery.exe --model-path D:\models\Kokoro-82M-v1_1-zh_F16.gguf --backend vulkan --vulkan-device 0

.\perf_battery.exe --model-path D:\models\Kokoro-82M-v1_1-zh_F16.gguf --repeat 3 --warmup 1
```

## 参数说明

- `--model-path`：必填，GGUF 模型路径  
- `--backend`：可选，`cpu/metal/vulkan/auto`，默认 `cpu`  
- `--vulkan-device`：可选，Vulkan 设备索引，默认 `0`  
- `--n-threads`：可选，CPU 线程数，默认硬件并发数  
- `--repeat`：可选，每个用例重复次数，默认 `1`  
- `--warmup`：可选，预热次数，默认 `1`  
- `--voice`：可选，语音包模型的 voice id

## 输出说明

输出包含两部分：  
1) 每个用例的平均 `time_ms / audio_s / rtf`  
2) 总体 `summary`（总耗时、总音频时长、均值/分位数、总体 RTF）
