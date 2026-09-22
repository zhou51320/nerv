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
- 目标环境：Windows 7 x64 SP1 + NVIDIA 官方 472.xx / 474.xx 驱动 + RTX 2080 Ti (sm_75) + CUDA 11.4
- 构建脚本：`build-llama-win7-cuda.ps1 -Clean -CudaArch 75 -Generator Ninja`（亦可通过 `-CudaArch 61` 兼容 Pascal 卡）
- 兼容旧脚本：`build-win7-cuda-sm61.ps1` 转发调用该统一脚本
- CI：`.github/workflows/build-llama-win7-cuda.yml`（windows-2022 + windows-setup-cuda 11.4 + MSVC v142 14.29）
- 产物：`EVA_BACKEND/x86_64/win7/cuda/llama.cpp/` 下包含 `llama-server.exe`, `llama-quantize.exe`, `llama-cli.exe`, `llama.dll`, `ggml-cuda.dll` 等，及打包好的 CUDA 运行时（`cudart64_110.dll`、`cublas64_11.dll`、`cublasLt64_11.dll`）和 VC 运行时
- Win7 兼容适配：
  - 保持各后端独立性：外部传入 `-DGGML_WIN_VER=0x601` 时通过宏注入 `/D_WIN32_WINNT=0x0601 /DWINVER=0x0601`，自动隔离 Win8+ API（如 `PrefetchVirtualMemory`、`SetThreadInformation`）；未指定时不影响默认构建
  - 集成 YY-Thunks（`YY_Thunks_for_Win7.obj`）：链接阶段自动挂钩缺失的 Win8+ 系统调用（如 `GetSystemTimePreciseAsFileTime`、`CreateFile2`、`SetThreadDescription` 等），在 Win7 下无缝降级到兼容 API
  - 避免打包 CI 容器内 Windows Server 2022 的不兼容 VC 运行时 DLL，由目标机上的 VC++ 2015-2022 运行库或 YY-Thunks 提供干净运行时环境
  - 开启 `-DGGML_CUDA_NO_VMM=ON`，避免依赖老驱动层的 CUDA VMM 造成不稳定
  - 关闭 `-DGGML_CUDA_FA=OFF` 与 `-DGGML_CUDA_GRAPHS=OFF`，使用成熟稳定的 cuBLAS / MMQ 矩阵乘法路径
- 运行要求：目标机安装 Windows 7 最终官方驱动（NVIDIA 472.12 或 474.xx），安装 KB2999226（Universal C Runtime）及 Visual C++ 2015-2022 x64 运行库。直接解压运行即可使用 CUDA 11.4 进行 2080Ti 硬件加速推理。

## llama-swap（Win7）
- 源码：作为普通 vendor 目录存放在 `llama-swap/`
- 构建：CI 由 `.github/workflows/build-llama-swap-win7.yml` 自动编译
- 工具链：Node.js 22（构建 Web UI 资源）+ `thongtech/go-legacy-win7` 工具链（针对 Windows 7 回退适配 `RtlGenRandom`，标记 PE 子系统 6.1）
- 产物：`EVA_BACKEND/x86_64/win7/llama-swap/llama-swap.exe`（单文件内嵌 Web UI，无外部依赖）
- 运行要求：原生 Windows 7 x64 SP1 无需任何系统补丁即可直接双击运行
