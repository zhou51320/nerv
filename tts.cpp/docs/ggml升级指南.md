# ggml 升级指南（本仓库：tts.cpp）

本文档记录 **在本仓库中升级/替换 ggml 的推荐流程**，以及历史上遇到的 **常见兼容点**，方便下次更新时快速定位改动范围。

> 约定：本仓库使用 `参考项目/ggml-xxxxxxx` 作为“新 ggml 参考目录”，并用“**直接替换根目录 `ggml/`**”的方式升级（侵入性更小：不在 ggml 源码里打补丁，优先在项目侧兼容）。

---

## 1. 升级方式（推荐：直接替换 `ggml/`）

1) **准备新 ggml 源码目录**
- 将新版本 ggml 放到：`参考项目/ggml-<commit-or-tag>/`
- 目录结构应包含 `include/`、`src/`、`CMakeLists.txt` 等（与 ggml 官方仓库一致）

2) **替换根目录旧 ggml**
- 删除旧：`ggml/`
- 复制新：把 `参考项目/ggml-<commit-or-tag>/` 整目录复制为根目录 `ggml/`
- 保留本仓库自有目录：`ggml-patches/`（不要把它覆盖掉）

3) **重新生成并编译**
- 重新配置：`cmake -S . -B build`
- 编译：`cmake --build build --parallel`

4) **回归验证**
- Kokoro：`build/bin/tts-cli.exe --model-path <xxx.gguf> --voice <voice> -p "<prompt>" --save-path <out.wav>`
- Quantize：`build/bin/quantize.exe --model-path <fp32.gguf> --quantized-model-path <out.gguf> --quantized-type Q4_0`

---

## 2. 升级时常见兼容点（本仓库历史记录）

下面列出升级到 ggml 0.9.x 期间，本仓库已经做过/可能再次遇到的适配点；下次升级时可按此清单逐项排查。

### 2.1 GGUF / 头文件拆分
- 现象：`gguf_*` 未声明、编译报错。
- 处理：项目侧显式 `#include "gguf.h"`（不要依赖 `ggml.h` 间接包含）。

### 2.2 Scheduler API 变更
- 现象：`ggml_backend_sched_new(...)` 参数个数不匹配。
- 处理：按新签名补齐参数（例如新增 `op_offload`），在项目侧统一适配。

### 2.3 Windows / MinGW 头文件兼容
- 现象：MinGW 编译 ggml-cpu 时，Windows 线程/节能相关结构体缺失导致报错。
- 处理：在本仓库 CMake 里对 ggml 目标设置更低的 `WINVER/_WIN32_WINNT`（例如 `0x0601`）。

### 2.4 GGUF tensor 名称长度限制（Kokoro v1.1 常见）
- 现象：运行时报错：`gguf_init_from_file_impl: tensor name ... is too long: 64 >= 64`
- 原因：ggml 默认 `GGML_MAX_NAME=64`，最多只能存 63 个字符；部分 GGUF tensor 名称长度会达到 64。
- 处理：在本仓库 CMake 全局定义 `GGML_MAX_NAME=128`（注意：这样生成/读取的 gguf 对“保持默认 64 的其它程序”可能不兼容）。

### 2.5 ggml-cpu 二元算子 broadcast 断言（非连续 src1）
- 现象：运行时报错：`GGML_ASSERT(ggml_are_same_shape(src0, src1)) failed`（常见于 `ggml_add/ggml_mul`）
- 原因：ggml 0.9.4 的 CPU 路径对 “src1 非 contiguous + broadcast” 暂未实现，直接断言。
- 处理：对参与 broadcast 的 `ggml_transpose(...)` 结果追加一次 `ggml_cont(...)`，保证 src1 contiguous。

### 2.6 STFT / ISTFT 不再由 ggml 内置（Kokoro 依赖）
- 现象：编译失败：找不到 `ggml_stft/ggml_istft`。
- 处理：项目侧用 `GGML_OP_CUSTOM` 自定义算子补齐（本仓库已在 `src/util.cpp` 实现 `stft()` / `istft()` 包装）。

### 2.7 conv_transpose_1d 能力收敛（padding/groups/output_padding）
- 现象：
  - 编译失败：旧版 `ggml_conv_transpose_1d` 参数过多；
  - 或运行期断言：ggml 0.9.4 限制 `padding==0 && dilation==1`。
- 处理：
  - `groups==1`：用 `ggml_conv_transpose_1d(padding=0)` 计算更长输出，再用 view 裁剪实现 `padding/output_padding` 等价语义；
  - `groups>1`（Kokoro depthwise）：项目侧用 `GGML_OP_CUSTOM` 兜底实现。
  - 本仓库封装入口：`tts_conv_transpose_1d()`（定义在 `src/util.h`，实现于 `src/util.cpp`）。

### 2.8 quantize 工具对 ggml 新行为的适配
- 说明：ggml 0.9.4 的 GGUF 读取会在 `ggml_context` 中引入内部 tensor：`GGUF tensor data binary blob`（非模型权重）。
- 处理：quantize 遍历 tensor 时需要跳过该内部 tensor，否则会误做 f16/quantize 转换并触发类型断言。
