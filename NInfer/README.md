# NInfer：Win7 / CUDA 11.x / RTX 2080 Ti

本目录是 nerv 的独立 NInfer 后端源码，不修改或替换已有 llama.cpp 后端。

**当前是第一阶段兼容移植，不是已验收的 Win7 二进制发行版。** 目标为 RTX 2080 Ti
**22GB 改卡**、Windows 7 x64 SP1、CUDA 11.7、MSVC v142 14.29。Windows 编译、旧驱动加载、
真实模型推理和数值正确性必须分别验证；主机端测试通过不能代替这些检查。

## 固定来源与边界

- 原始项目：[Neroued/ninfer](https://github.com/Neroued/ninfer)。
- 移植底座：[mr-september/ninfer-2080ti-22g，6460bf03](https://github.com/mr-september/ninfer-2080ti-22g/tree/6460bf03a86f1288dc7b240d70c30887199099af)。
- 精确来源及快照校验值见 [UPSTREAM.json](UPSTREAM.json)。保留 Apache-2.0 及第三方各自的许可证。
- 模型文件使用 **`.ninfer v2`**；当前上游生成的 v3 文件不能直接使用，也不能修改版本字节冒充 v2。
- 首个验收模型为 **Qwen3.6-27B / groupwise-int**。它是自有混合分组整数量化布局，不是通用 AWQ/GPTQ/GGUF 加载器。
- 首期仅纯文本 CLI、单 GPU、单请求。默认上下文 2048、prefill chunk 256、INT8 group-64 KV。
- Win7 profile 禁止 CUDA Graph、MTP/DFlash、视觉输入和 HTTP server。不是静默降级：不支持的请求会报错。
- 不启用 CUDA VMM，不依赖 FFmpeg、curl 或 NVTX。沿用已有 SM75 运算及 BF16 软件回退，尚不承诺性能收益。

原分支的 Linux/CUDA 12.9 性能、长上下文及 MTP 数据**不代表此 Win7 profile 的实测结果**。
原装 11GB 卡装不下首个模型的现成配置；减小上下文无法解决权重本身超显存的问题。

## 构建

### GitHub Actions：仅编译 NInfer

工作流为 [Build NInfer Win7 CUDA](../.github/workflows/build-ninfer-win7-cuda.yml)，
**仅手动触发、只有一个构建任务**，不会调用现有多后端工作流：

- `windows-2022` + MSVC v142 14.29 + CUDA 11.7 + Ninja，目标为 `ninfer`。
- 只检出 NInfer、CUDA 安装 action 和 YY-Thunks，不递归检出其他后端。
- 不运行测试或推理，不下载模型，不发布 Release。
- 成功时上传 `NInfer-win7-cuda11.7-sm75-unverified-<run_number>`；失败时仍上传已有配置/编译日志。

先将 `NInfer/` 和工作流文件同步到 GitHub 默认分支，再从 Actions 页面选择该工作流并
点击 **Run workflow**，或执行：

```bash
gh workflow run build-ninfer-win7-cuda.yml --repo zhou51320/nerv --ref main
```

这个 build-only 流程直接调用 CMake，**不需要 `Win7SystemDir`**。它不执行本地严格打包脚本，
也不把 Windows Server 2022 编译成功当成 Win7 验收；产物内的 `BUILD-INFO.txt` 会注明尚未验证。
GitHub Actions 只能编译远端已有源码，不能使用尚未推送的本地文件。

### 本地构建及严格包验证

使用现代 Windows x64 构建机，不要求在 Win7 安装开发工具：

1. Visual Studio 的 **v142 14.29 x64** C++ 工具链与 Windows SDK。
2. **CUDA Toolkit 11.7**。源码兼容入口也允许 11.8，但首期验收以 11.7 为准。
3. CMake 3.28+、Ninja、PowerShell 5.1+。
4. nerv 提供的 `third_party/YY-Thunks/objs/x64/YY_Thunks_for_Win7.obj`。

在 nerv 根目录执行（`D:\win7-system32` 是已安装所需 CRT/UCRT 的目标 Win7 System32 快照）：

```powershell
.\NInfer\scripts\build-win7-cuda.ps1 -Win7SystemDir 'D:\win7-system32'
```

参数及工具链检测规则以脚本 `Get-Help` 为准。脚本只构建 `ninfer` CLI，不运行模型，
不下载模型或安装驱动。已存在的非空输出目录不会被默认覆盖。

核心 CMake 配置如下，通常由脚本负责传入：

```text
-DNINFER_WIN7=ON
-DNINFER_TEXT_ONLY=ON
-DNINFER_BUILD_SERVER=OFF
-DNINFER_DISABLE_NVTX=ON
-DCMAKE_CUDA_ARCHITECTURES=75-real
-DBUILD_TESTING=OFF
-DNINFER_BUILD_BENCHMARKS=OFF
```

CUDA 编译单元使用 C++17，主机代码保持 C++20；`75-real` 生成 sm75 cubin，避免依赖旧驱动的
新 PTX JIT。Win7 构建使用静态 cudart、受控的 `/MD` CRT 和 YY-Thunks。**静态 cudart 不等于
静态 CRT，也不保证所有系统／驱动 API 都兼容 Win7。**

按 nerv 的目录规范，产物放在：

```text
EVA_BACKEND/x86_64/win/cuda/NInfer/
```

这里是专用 Win7 profile 包；不要混入其他 CUDA 版本、其他后端或新版 Visual C++ 运行库的 DLL。
不要将 `nvcuda.dll` 当作应用依赖复制进包，它由目标机 NVIDIA 驱动提供。

## 模型准备

先使用已经确认格式的 v2 文件。首个模型的上游 artifact 约 16.29 GiB；文件大小不是峰值显存，
仍须为运行时、KV 和工作区留余量。不要直接下载“最新版”后假设它仍是 v2。

若需自行转换，在另一台具备 Python/PyTorch 和源模型的机器上使用**本目录附带的转换器**，
从 `NInfer/` 运行：

```bash
python -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B/base-hf-bf16 \
  --out /path/to/qwen3_6_27b.ninfer
```

转换是独立的离线工作；Win7 推理不需要 Python、PyTorch 或 Triton。本次移植不会自动下载或转换模型。
完整格式规则见 [artifact-container.md](docs/maintainer/artifact-container.md)。

## Win7 运行与验收

目标机需 Win7 x64 SP1、与显卡匹配的 NVIDIA R470 系列驱动、Win7 可用的 VC++ 2019 x64
14.29 运行库及 UCRT 补丁。具体前置条件见 [WIN7-RUNTIME.txt](scripts/WIN7-RUNTIME.txt)。
CUDA 11.x 并非 NVIDIA 对 Win7 的官方支持组合；必须固定并记录实际可用的驱动和运行库。
**目标机不需要安装 CUDA Toolkit。**

从独立、干净的包目录开始：

```powershell
.\ninfer.exe --help
.\ninfer.exe D:\models\qwen3_6_27b.ninfer --prompt "Reply with one short sentence." `
  --max-context 2048 --kv-capacity 2048 --prefill-chunk 256 --max-new 64 `
  --kv-dtype int8 --no-cuda-graph --greedy --no-thinking --print-token-ids
```

`--help` 只检查启动和参数路径，不能证明 GPU 推理可用。正文写 stdout，加载、token ID、
prefill/decode 计时及显存摘要写 stderr。

第一阶段的验收条件：

1. Windows v142/CUDA 11.7 完整编译链接，包内依赖检查通过。
2. 原生 Win7（不依赖 VxKex）能够启动、加载模型和运行真实 prefill/decode。
3. 同一个 v2 artifact，在固定输入和采样条件下，与已知可用参考执行比较 token、数值和质量；
   浮点路径不应只凭“输出看起来合理”验收，也不强求不同精度路径逐位相同。
4. 无 NaN/Inf、异常退出、驱动超时；重复运行并记录显存峰值和系统/驱动/toolkit版本。

在此之前，不能把编译成功、`--help` 成功或 CPU 单元测试成功写成“移植验收完成”。

## 不需要 GPU 的主机端回归

这个独立测试入口不经过顶层 CUDA compiler detection：

```bash
cmake -S NInfer/tests/win7 -B NInfer/build-host-tests -DCMAKE_BUILD_TYPE=Release
cmake --build NInfer/build-host-tests -j
ctest --test-dir NInfer/build-host-tests --output-on-failure
```

它用于检查首期 CLI 参数契约及可在主机执行的兼容逻辑，不代替 CUDA 或 Win7 实机测试。
包验证脚本及其 mock-dumpbin 策略回归位于 `scripts/`。

## 后续工作

先完成上述实机验收，再评估 CUDA Graph、MTP、SM75 热点内核及新上游格式的迁移。
暂不合并全部上游提交，也不复制无源码或未核实的 3060 优化声明。
