# nerv
这是为eva提供后端的项目
用于编译llama.cpp whisper.cpp stable-diffusion.cpp用

## 目标
- 量产
- 补完

## 编译
build.bat 
默认只会编译cpu和vulkan版本，-d all 为所有版本
如果编译器是mingw则自动编译win7版本
## 更新后端时注意
- 为了能在win7下运行所有项目的主CMakeLists.txt中顶部添加
```cmake
if (MSVC)
    unset(GGML_WIN_VER CACHE)
else()
    set(GGML_WIN_VER "0x601" CACHE STRING "ggml: Windows version")
endif()
```

## BeeLlama KVarN

`beellama.cpp/` 单独保存 BeeLlama v0.4.6，用于 KVarN KV Cache。它与官方 `llama.cpp/` 并行维护，不替换官方版本。

Win7 Vulkan 构建由 `.github/workflows/build-beellama-win7-vulkan.yml` 完成，产物位于 `EVA_BACKEND/x86_64/win7/vulkan/beellama.cpp/`，支持 `kvarn2`、`kvarn3`、`kvarn4`、`kvarn5`、`kvarn6` 和 `kvarn8`。

## ik_llama.cpp

`ik_llama.cpp/` 保存 Iwan Kawrakow 的 llama.cpp 分支（支持各种专用 IQ 量化与推理优化）。作为普通 vendor 目录由父仓库管理。

修改与适配（兼容 Win7）：
- 主 `CMakeLists.txt` 顶部设置 `GGML_WIN_VER "0x601"`
- `vendor/cpp-httplib/httplib.h` 移除 `_WIN32_WINNT < 0x0A00` 的 `#error` 报错
- `vendor/cpp-httplib/httplib.cpp` 中 `mmap::open` 替换为兼容 Windows 7 的 Win32 API（`CreateFileW`、`CreateFileMappingW`、`MapViewOfFile`）

Win7 Vulkan 构建由 `.github/workflows/build-ik-llama-win7-vulkan.yml` 完成，产物位于 `EVA_BACKEND/x86_64/win7/vulkan/ik_llama.cpp/`。
- llama.cpp 为了能在win7下运行
    - 使用mingw编译器 gcc 12版本以上
    - 去掉llama.cpp/vendor/cpp-httplib/httplib.h 中 
```cpp
#ifdef _WIN32
#if defined(_WIN32_WINNT) && _WIN32_WINNT < 0x0A00
#error                                                                         \
    "cpp-httplib doesn't support Windows 8 or lower. Please use Windows 10 or later."
#endif
#endif
```

    - 搜索 bool mmap::open(const char *path) 替换
```cpp

inline bool mmap::open(const char *path) {
  close();

#if defined(_WIN32)
  auto wpath = u8string_to_wstring(path);
  if (wpath.empty()) { return false; }

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM | WINAPI_PARTITION_GAMES) && (_WIN32_WINNT >= _WIN32_WINNT_WIN8)
  hFile_ = ::CreateFile2(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                         OPEN_EXISTING, NULL);
#else
  hFile_ = ::CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#endif
  if (hFile_ == INVALID_HANDLE_VALUE) { return false; }

  LARGE_INTEGER size{};
  if (!::GetFileSizeEx(hFile_, &size)) { return false; }
  // If the following line doesn't compile due to QuadPart, update Windows SDK.
  // See:
  // https://github.com/yhirose/cpp-httplib/issues/1903#issuecomment-2316520721
  if (static_cast<ULONGLONG>(size.QuadPart) >
      (std::numeric_limits<decltype(size_)>::max)()) {
    // `size_t` might be 32-bits, on 32-bits Windows.
    return false;
  }
  size_ = static_cast<size_t>(size.QuadPart);

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM) && (_WIN32_WINNT >= _WIN32_WINNT_WIN8)
  hMapping_ =
      ::CreateFileMappingFromApp(hFile_, NULL, PAGE_READONLY, size_, NULL);
#else
  hMapping_ =
      ::CreateFileMappingW(hFile_, NULL, PAGE_READONLY, size.HighPart,
                           size.LowPart, NULL);
#endif
  // Special treatment for an empty file...
  if (hMapping_ == NULL && size_ == 0) {
    close();
    is_open_empty_file = true;
    return true;
  }

  if (hMapping_ == NULL) {
    close();
    return false;
  }

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM) && (_WIN32_WINNT >= _WIN32_WINNT_WIN8)
  addr_ = ::MapViewOfFileFromApp(hMapping_, FILE_MAP_READ, 0, 0);
#else
  addr_ = ::MapViewOfFile(hMapping_, FILE_MAP_READ, 0, 0, 0);
#endif
  if (addr_ == nullptr) {
    close();
    return false;
  }
#else
  fd_ = ::open(path, O_RDONLY);
  if (fd_ == -1) { return false; }

  struct stat sb;
  if (fstat(fd_, &sb) == -1) {
    close();
    return false;
  }
  size_ = static_cast<size_t>(sb.st_size);

  addr_ = ::mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);

  // Special treatment for an empty file...
  if (addr_ == MAP_FAILED && size_ == 0) {
    close();
    is_open_empty_file = true;
    return false;
  }
#endif

  return true;
}
```

- stable-diffusion.cpp 搜索 LOG_DEBUG("Using Vulkan backend");替换为如下代码
```cpp
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        int dev_count = ggml_backend_vk_get_device_count();
        int dev = 0;
        if (const char* s = getenv("GGML_VK_DEVICE")) {
            int v = atoi(s);
            if (v >= 0 && v < dev_count) dev = v;
        }
        // prefer a discrete NVIDIA device if available; fallback to first
        int preferred = -1;
        for (int i = 0; i < dev_count; ++i) {
            char desc[256] = {0};
            ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
            // avoid SwiftShader/Software devices and prefer NVIDIA/GeForce/RTX naming
            std::string d(desc);
            if (d.find("NVIDIA") != std::string::npos || d.find("GeForce") != std::string::npos || d.find("RTX") != std::string::npos) {
                preferred = i;
                break;
            }
        }
        if (preferred >= 0) dev = preferred;
        // Log available devices
        for (int i = 0; i < dev_count; ++i) {
            char desc[256] = {0};
            ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
            LOG_INFO("ggml_vulkan: %d = %s", i, desc);
        }
        backend = ggml_backend_vk_init(dev);
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend (device %d)", dev);
        } else {
            char desc[256] = {0};
            ggml_backend_vk_get_device_description(dev, desc, sizeof(desc));
            LOG_INFO("Vulkan selected device: %d - %s", dev, desc);
        }
#endif

```

## fastllm（Win7 CUDA / sm_75）
- 上游：`https://github.com/ztxz16/fastllm`，master @ `a679ccad`，作为普通 vendor 目录放在 `fastllm/`
- 构建：`build-fastllm-win7-cuda.ps1 -Clean -CudaArch 75`（MSVC v142 14.29 + Ninja + CUDA 11.7）
- CI：`.github/workflows/build-fastllm-win7-cuda.yml`（windows-2022 + windows-setup-cuda 11.7）
- 产物：`EVA_BACKEND/x86_64/win7/cuda/fastllm/{main.exe, quant.exe, fastllm-apiserver.exe} + cuda 运行库 + VC 运行库`
- fastllm 无 Vulkan 后端，Windows GPU 路径只有 CUDA；本仓库定位为 sm75（2080Ti）
- Win7 兼容本地补丁（全在 `fastllm/` 内，不改上游）：
  - `CMakeLists.txt`：WIN32 时加 `-D_WIN32_WINNT=0x0601`；剔除可选 Triton 源；链接只保留 `cublasLt cublas`（去掉 nccl/cuda）；把 `third_party/nccl_stub` 加入构建，并新增 `fastllm-apiserver` 可执行目标
  - `src/devices/cuda/fastllm-cuda.cu`：`FastllmCudaValidatePointerRange` 在 Windows 用 runtime API `cudaPointerGetAttributes`（无 driver import lib）
  - `third_party/nccl_stub/`：Windows 无 NCCL，提供最小 nccl.h + stub 实现（多卡运行时不可用，单卡无影响）与 `cuda_profiler_api.h` fallback
- 运行要求：目标机 Win7 + NVIDIA 驱动（支持 Turing/2080Ti）+ 打包目录内附带的 CUDA/VC 运行库 DLL
- 目标机使用：`main.exe` 交互对话、`quant.exe` 量化、`fastllm-apiserver.exe` OpenAI 兼容 HTTP server（自带 winsock 网络栈，无需外部依赖）

## llama.cpp（Win7 CUDA / sm_75 / 2080Ti）
2080Ti 在 Win7 上的主力后端（prefill 约为 Vulkan 的 3 倍，decode 持平或更快），Vulkan 作为备选持续关注。
- 目标环境：Windows 7 x64 SP1 + NVIDIA 官方 472.xx / 474.xx 驱动（R470，最高支持 CUDA 11.x，**无法使用 CUDA 12**）+ RTX 2080 Ti (sm_75)
- 构建脚本：`build-llama-win7-cuda.ps1 -Clean -CudaArch 75 -Generator Ninja`（亦可通过 `-CudaArch 61` 兼容 Pascal 卡）
- 兼容旧脚本：`build-win7-cuda-sm61.ps1` 转发调用该统一脚本
- CI：`.github/workflows/build-llama-win7-cuda.yml`（windows-2022 + windows-setup-cuda + MSVC v142 14.29），push 时同时编译 CUDA 11.4 与 11.7，手动触发可指定单个版本；构建后检查所有 exe/dll 不含 `api-ms-win-core-*` 导入
- 产物：`EVA_BACKEND/x86_64/win7/cuda/llama.cpp/` 下包含 `llama-server.exe`, `llama-quantize.exe`, `llama-cli.exe`, `llama.dll`, `ggml-cuda.dll` 等，及 `cublas64_11.dll`、`cublasLt64_11.dll`（cudart 已静态链接进 `ggml-cuda.dll`）
- 运行要求：目标机安装 NVIDIA 472.12 或 474.xx 驱动、KB2999226（Universal C Runtime）及 Visual C++ 2015-2022 x64 运行库；**不需要安装 CUDA Toolkit**（反而可能因 PATH 中的 cudart/cublas 版本冲突导致加载错误）

### Win7 兼容适配
- 外部传入 `-DGGML_WIN_VER=0x601` 时通过宏注入 `/D_WIN32_WINNT=0x0601 /DWINVER=0x0601`，自动隔离 Win8+ API（如 `PrefetchVirtualMemory`、`SetThreadInformation`）；未指定时不影响默认构建
- 集成 YY-Thunks（`YY_Thunks_for_Win7.obj`）：链接阶段挂钩缺失的 Win8+ 系统调用（如 `GetSystemTimePreciseAsFileTime`、`CreateFile2`、`SetThreadDescription`）
- 不打包 CI 容器内 Windows Server 2022 的 VC 运行时 DLL，由目标机 VC++ 2015-2022 运行库提供
- `-DGGML_STATIC=ON`：MSVC 下只作用于 cudart（改为 `cudart_static`）。CUDA 11.7 的 `cudart64_110.dll` 通过 `api-ms-win-core-libraryloader-l1-2-0.dll` 等 Win8+ api-set 导入 `LoadLibraryExW` 等函数，Win7 启动报 DLL 缺失；静态链接后直接从 KERNEL32 导入，不再需要 VxKex
- `-DGGML_CUDA_NO_VMM=ON`：不依赖老驱动的 CUDA VMM

### 编译期优化
- `-DGGML_CUDA_GRAPHS=ON`：整图一次提交，大幅降低 Win7 WDDM 下的 kernel 发射开销，MoE decode 从约 40-45 t/s 提升到与 Vulkan 持平（70+ t/s），也让 MTP 可以用更大的 draft。CUDA 11.x 走 `cudaGraphExecUpdate` 旧签名分支。已知问题：显存可能随对话轮数上涨（上游 #25835），遇到时运行前设 `GGML_CUDA_DISABLE_GRAPHS=1`
- `-DGGML_CUDA_FA=ON` + `-DGGML_CUDA_FA_QUANTS=all`：Turing 走 mma-f16 Tensor Core 路径，K/V 可用任意量化组合（如 K=q8_0、V=q4_0）
- **优先使用 CUDA 11.7 产物**：11.7 起启用 CUB，词表级（ne0>1024）的 TOP_K/ARGSORT 可在 GPU 执行。11.4 下 MTP 默认的 draft backend sampling 会因 TOP_K 不支持而回落 CPU 并拆图（Qwen3.x 27B Q3 仅 3-4 t/s），11.4 只能加 `--no-spec-draft-backend-sampling` 规避
- 不开 `GGML_CUDA_FORCE_MMQ`（Turing 默认已走 MMQ）与 `GGML_CUDA_FORCE_CUBLAS`（显存更高且有 fp16 溢出风险）
- CUDA 11.8 相比 11.7 只多 `movmatrix`（FA/MMA 略快），收益很小；PDL 在 MSVC 下需 CUDA ≥ 12.3，Win7 不可用

### 运行期优化（2080Ti 22G 实测参考：Qwen3.5-35B MoE 11.7 prefill ~1620 t/s、decode ~77 t/s；27B 稠密 UD-Q4 + MTP decode 32-34 t/s）
- 全量放显存：`-ngl 99`；放不下时用 `-ncmoe N` 或 `-ot "exps=CPU"` 只把部分 MoE 专家放 CPU，不要让 llama.cpp 自动减层
- Flash Attention + KV 量化：`-fa on -ctk q8_0 -ctv q4_0`（或 `-ctv q8_0` 换精度）
- MTP 投机解码：`--spec-type draft-mtp --spec-draft-n-max 2`。开 graphs 后 CUDA 上 2 最快（接受率约 63%）；Vulkan / 不开 graphs 时 1 最快（Win7 每次提交开销大，draft 越多越慢）
- 长 prompt prefill：MoE 可试 `-b 2048 -ub 1024`（或 2048）提高 MMQ 批量，代价是计算缓冲显存增加；爆显存时减小 `-ub`
- 多轮对话复用：`--cache-reuse 256`；Qwen3.5 这类混合（线性注意力）模型保留默认 `--ctx-checkpoints`，避免每轮重算整段 prompt
- 单用户使用时 `-np 1`，把 KV 和计算缓冲留给单个会话
- 可选实验：`GGML_CUDA_GRAPH_OPT=1`（graph 内可并行分支走多 stream，MoE 可能再有少量提升，异常就关）
- 功耗：140W 限功耗主要影响 prefill 和稠密模型 decode；默认 250-260W 下稠密模型 decode 更高
- Win7 驱动没有"系统内存回落"，显存不够直接 OOM，需给 KV 和计算缓冲预留余量
- Turing 无 bf16 Tensor Core：bf16 权重的模型请转 f16 或量化

## llama.cpp（Win7 Vulkan / 2080Ti）
- CI：`.github/workflows/build-llama-win7-vulkan.yml`（MinGW 静态链接 + Vulkan SDK 1.3.290，push `llama.cpp/**` 或手动触发）
- 产物：`EVA_BACKEND/x86_64/win7/vulkan/llama.cpp/`
- 本地补丁 `ggml/src/ggml-vulkan/ggml-vulkan.cpp` `get_device_architecture`：Win7 驱动无 `VK_KHR_cooperative_matrix`，原逻辑把 2080Ti 误判为 `NVIDIA_PRE_TURING`（dmmv 工作组等启发式走老卡参数）；改为先用 `VK_NV_shader_sm_builtins` 的 `shaderWarpsPerSM==32` 识别 Turing。更新 llama.cpp 时需重新打上
- Vulkan SDK 不要低于 1.3.290：v0.5.0 新增的 `flash_attn_decode_phase_*.comp` 无条件使用 `GL_KHR_cooperative_matrix`
- Win7 474.xx 驱动只提供 Vulkan 1.2.175，没有 integer dot / cooperative matrix，prefill 明显慢于 CUDA，2080Ti 建议优先用 CUDA 后端；`GGML_VK_DISABLE_COOPMAT*`、`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` 在该驱动上无效，不要设 `GGML_VK_PREFER_HOST_MEMORY`

## llama-swap（Win7）
- 源码：作为普通 vendor 目录存放在 `llama-swap/`
- 构建：CI 由 `.github/workflows/build-llama-swap-win7.yml` 自动编译
- 工具链：Node.js 22（构建 Web UI 资源）+ `thongtech/go-legacy-win7` 工具链（针对 Windows 7 回退适配 `RtlGenRandom`，标记 PE 子系统 6.1）
- 产物：`EVA_BACKEND/x86_64/win7/llama-swap/llama-swap.exe`（单文件内嵌 Web UI，无外部依赖）
- 运行要求：原生 Windows 7 x64 SP1 无需任何系统补丁即可直接双击运行
