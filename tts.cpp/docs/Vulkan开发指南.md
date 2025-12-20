# Vulkan 支持与优化指南

本文用于说明 TTS.cpp 在 Vulkan 后端的实现策略、接入规范与性能优化方向，方便后续开发者持续完善与扩展。

## 目标与范围

- 目标：尽可能让 Kokoro 的生成图在 Vulkan 上完整执行，同时保留“安全回退 CPU”的机制，避免崩溃。
- 范围：目前重点覆盖 Kokoro 模型的生成阶段（STFT/ISTFT、噪声注入、上采样等）。

## 当前实现概览

- 后端选择：CLI 通过 `--use-vulkan/-vk` 与 `--vulkan-device/-vd` 选择 Vulkan 设备。
- 回退策略：
  - 生成图若检测到自定义算子（`GGML_OP_CUSTOM` 等）且未强制 Vulkan，会固定自定义算子到 CPU，其余尽量走 Vulkan。
  - 分配/计算失败会回退到 CPU 重新分配/计算。
- STFT/ISTFT：
  - 使用 `stft_graph/istft_graph`（标准 ggml 算子）代替自定义算子，避免 Vulkan 构图崩溃。
  - STFT 基矩阵由 `kokoro_build_stft_basis` 预计算并常驻权重缓冲。
  - 反射 padding 通过索引张量（`stft_pad_indices`）+ `ggml_get_rows` 完成。
- 常量输入：
  - Vulkan 图中不可直接依赖 host 指针常量。
  - 项目侧用 `tts_graph_const_input` 注册常量张量并在 `set_inputs` 中通过 `ggml_backend_tensor_set` 写入。

关键实现位置（便于跟踪）：

- `src/util.cpp`：`stft_graph/istft_graph`、atan2 近似、反射 padding 索引逻辑。
- `src/models/kokoro/model.cpp`：Vulkan 生成图构建、输入准备、回退策略。
- `src/models/loaders.cpp`：Vulkan 权重缓冲选择与 `TTS_VK_HOST_BUFFER`。

## 新增/修改算子时的约定

1. **尽量避免 GGML_OP_CUSTOM**
   - Vulkan 对 `GGML_OP_CUSTOM` 不支持，会触发 CPU 回退或引发构图崩溃。
   - 优先用 ggml 标准算子组合（conv/get_rows/reshape/transpose 等）。
2. **常量不要直接挂 host 指针**
   - 需要常量时，创建张量并 `ggml_set_input`。
   - 使用 `tts_graph_const_input` 统一登记，`set_inputs` 中写入。
3. **保证输入可写**
   - 输入张量一律通过 `ggml_backend_tensor_set` 写入，避免 CPU 直接访问 GPU 内存。
4. **注意连续性**
   - 部分算子在 Vulkan/CPU 下要求 src 连续，必要时调用 `ggml_cont`。
5. **同步与 reset**
   - 异步后端计算结束后必须先 `sync`，再 `ggml_backend_sched_reset`，避免释放仍在使用的 buffer。

## Vulkan 性能优化方向（可按优先级推进）

1. **减少 CPU 回退**
   - 继续替换自定义算子为标准 ggml 算子。
   - 避免在 Vulkan 图中混用 CPU-only 的 view/leaf。
2. **减少不必要的拷贝**
   - 调整计算图，减少 `permute + cont` 的组合。
   - 让中间张量尽量保持连续布局。
3. **提高算子融合与吞吐**
   - 避免把非常小的算子单独放到 GPU；可尝试合并或前后端分配权衡。
   - 评估将更多权重保持为 F16（带宽更友好）。
4. **图构建/分配开销**
   - 评估常见长度下的图复用与缓存（减少每次 build/alloc）。
   - 输出 buffer 尽量复用，避免频繁重新分配。
5. **内存与带宽优化**
   - 使用更小模型或量化模型，降低显存/带宽压力。
   - 调整 ggml-vulkan 的内存参数（见下方环境变量）。

## 调试与环境变量

项目侧：

- `TTS_VK_FORCE_GEN=1`：强制生成阶段走 Vulkan（即便有自定义算子）。
- `TTS_VK_HOST_BUFFER=1`：权重缓冲改为主机可见，避免 CPU 回退读显存导致崩溃（可能降低速度）。
- `TTS_VK_PREALLOC=1`：启用 Vulkan 的 sched 预分配（默认关闭以避免“最坏图”导致启动阶段大额申请/失败耗时）。

ggml-vulkan（上游提供，便于性能/兼容调优）：

- `GGML_VK_VISIBLE_DEVICES`：指定可见设备索引（逗号分隔）。
- `GGML_VK_PREFER_HOST_MEMORY`：优先使用 host 内存。
- `GGML_VK_ALLOW_SYSMEM_FALLBACK`：允许系统内存回退。
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE`：关闭图优化（定位问题时使用）。
- `GGML_VK_DISABLE_FUSION`：关闭算子融合（对比性能/定位错误时使用）。
- `GGML_VK_DISABLE_F16`：禁用 F16（兼容性排查）。
- `GGML_VK_FORCE_MAX_BUFFER_SIZE`：限制单次 buffer 分配上限，避免驱动 OOM。
- `GGML_VK_SUBALLOCATION_BLOCK_SIZE`：调节子分配块大小（权衡碎片与分配次数）。
- `GGML_VK_PERF_LOGGER`：输出 Vulkan 内部性能统计（可配 `GGML_VK_PERF_LOGGER_FREQUENCY`）。

建议：性能调优时先打开 `GGML_VK_PERF_LOGGER`，结合 `kokoro` 内部 timings 一起观察瓶颈。

## 常见问题

- **Vulkan 崩溃多发生在“混合后端 buffer + 自定义算子”**：优先检查图中是否含 `GGML_OP_CUSTOM`。
- **`tensor->data` 指向 host 指针导致 Vulkan 访问非法**：统一改成输入张量 + backend 写入。
- **OOM 或分配失败**：可先允许运行时分配，再尝试降低模型体积或设置 `GGML_VK_FORCE_MAX_BUFFER_SIZE`。
