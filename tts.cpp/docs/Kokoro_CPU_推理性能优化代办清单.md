# Kokoro CPU 推理性能优化：方案与代办清单

> 目标：在**不依赖额外硬件**（仅 CPU）、**不引入第三方库**、**尽量不改 ggml** 的前提下，通过“内部结构与流程”把 Kokoro 的端到端推理速度显著提升，并且让优化过程可度量、可回归、可持续迭代。

## 1. 现状与瓶颈（以现有 timings 为准）

当前端到端日志中：

- `frontend.normalize/phonemize/tokenize` 占比很小（百毫秒量级）。
- 绝大部分时间在 `frontend.run`，其内部主要由**多次 generator 计算**组成（每个 chunk 都会执行 duration + generator）。
- 单次 generator 的细分里，`graph_build/sched_alloc/set_inputs` 很小，主要耗时在 `compute`（也就是图的算子执行）。

结论：CPU 侧想“**大幅**提速”，必须把优化主战场放在：

1) **减少 compute 需要做的工作量**（算子数/算子规模/重复计算），以及  
2) **降低 compute 的每次代价**（内存带宽、布局、避免多余 cont/transpose、减少大张量复制、合理线程策略）。

## 2. 总体原则（只做内部结构/流程优化）

1. **先度量后优化**：任何“快了”的结论都要能被 `tts-bench` / `perf_battery` 复现，并记录配置与 RTF。
2. **先做确定性收益**：例如减少 `ggml_cont`、减少 O(N²) 拼接/复制、缓存不变输入、复用 buffer/graph 元信息。
3. **把大张量的数据流做“零拷贝/少拷贝”**：能就地写就不多走一遍中间 vector；能复用就不重复分配。
4. **线程不是越多越好**：CPU 上常见瓶颈是内存带宽与缓存争用；按阶段设置线程数上限通常收益更稳定。
5. **兼容/回退开关**：所有可能影响质量/稳定性的优化，都要提供环境变量回退路径，便于快速定位问题。

## 3. 可落地的优化方案（按优先级）

下面每一项都遵守“只改项目内部”的约束，并尽量参考 `参考项目/whisper.cpp` 的组织方式（图/内存/调度/缓存/bench 闭环）。

### P0：先建立“可重复”的性能基线与回归机制（必须做）

- 建议新增一套 **CPU 专用基线用例**：
  - 短句（< 2s）、中句（~10s）、长句（> 30s）各 1；
  - 中英混合/纯中文/纯英文各 1；
  - 统一输出：`RTF`、总耗时、每阶段耗时（frontend/duration/generator）。
- 在 docs 里固定记录：模型量化类型、线程数、编译选项、操作系统、CPU 型号（仅记录，不作为依赖）。

预期收益：让后续所有优化“可验证”，避免“感觉变快”。

### P1：减少不必要的数据搬运与重复分配（确定性收益）

1) **消除 chunk 拼接的 O(N²) 复制**

- 当前多段生成会把 `partial` 通过 `append_to_response()` 不断拷贝到 `response`，随着段数增加会出现多次全量拷贝。
- 改造建议：
  - 用 `std::vector<float>` 做聚合缓冲，提前 `reserve(总 samples)` 或分段 push_back；
  - 最终一次性输出为 `tts_response`（或让 `tts_response` 本身持有 vector）。

2) **输入准备尽量“直接写入目标 tensor”**

- 典型点：duration_ids / 噪声 / 窗函数等输入目前常见“先写临时 vector，再 ggml_backend_tensor_set 一次拷贝”。
- 在 CPU backend 下，这些 tensor 的 data 往往就在主存，可考虑：
  - 在确保 tensor 可写（CPU buffer、连续）的前提下，直接写入 `ggml_tensor->data`；
  - 或者为这类输入维持一个长期复用的 host-side 缓冲（避免每次分配/扩容）。

3) **避免无意义的清零**

- `ggml_backend_buffer_clear(buf_output, 0)` 对大输出会造成线性写带宽占用。
- 如果图保证会写满输出（或可以只写有效区），可改为：
  - 仅在 debug/对齐要求时清零；
  - 或者只清零尾部 padding（若存在）。

预期收益：长文本、多段输出时更明显（降低内存带宽与 memcpy 开销）。

### P2：减少 `ggml_cont/transpose/view` 带来的大拷贝（通常是隐藏大头）

CPU 上很多“看起来是小操作”的 `cont()`，一旦发生在大 2D/3D 张量上就会变成实打实的内存搬运。

优化抓手：

- 在 `tts_cont_if_needed()`（或类似封装）里增加统计：
  - 被迫 cont 的次数、总字节数、最大单次字节数；
  - 关联到 tensor 名称/调用点（至少在 debug 模式下可打印 top-k）。
- 逐个消灭大拷贝来源：
  - 调整张量布局，让后续算子天然吃 contiguous；
  - 用 `ggml_permute`/`ggml_transpose` 前先评估是否能换成“时间优先/通道优先”的统一布局（已有类似优化可继续推广到更多块）。

预期收益：如果当前生成图里存在多次大张量 cont，收益可能是“倍数级”。

### P3：图与调度器的“长期复用”（降低固定开销 + 稳定吞吐）

现有实现已经使用 `ggml_backend_sched` 与 meta buffer，但仍可进一步做“长期复用/预热”：

- **固定 worst-case 预分配策略**（参考 whisper.cpp 的 `whisper_sched_graph_init` 思路）：
  - 在不触发超大内存申请的前提下，用“可控 worst-case”做一次 `alloc_graph/reserve`；
  - 让运行时的 `sched_alloc` 不再发生 reallocate/碎片化。
- **将不随输入变化的常量 tensor 永久驻留**：
  - Hann window、STFT pad indices、常用标量常量（Vulkan 已有 const_inputs 机制，CPU 侧也可以统一管理）；
  - 减少每次 build_graph 时重复 new_tensor + set_input 的结构成本。

预期收益：对短句/高频调用场景更明显，且能降低抖动。

### P4：按阶段的线程策略（更少争用、更高吞吐）

参考 whisper.cpp 在 compute helper 里对各 backend 设置线程数的方式，可以在 Kokoro 内做“分阶段线程上限”：

- duration 阶段通常更小、访存更敏感：维持较小线程数（项目已有 `TTS_CPU_THREADS_DURATION`）。
- generator 阶段是绝对大头：需要评估“线程数 vs RTF”的拐点：
  - 增加 `TTS_CPU_THREADS_GEN`（或复用统一 `--threads` 但允许内部按阶段缩放）；
  - 在 Windows/小核 CPU 上，线程过多可能反而更慢（缓存抖动 + 调度开销）。

预期收益：在不改模型的情况下，这是“最容易拿到 10%~30%”的方向之一。

### P5：算子级热点定位后做结构性改写（高风险但可能最大收益）

当确认 CPU 端 compute 主要耗在某几类算子（例如 conv / conv_transpose / matmul / layernorm）后，再做更激进的内部结构改写：

- **把重复的逐层小算子改为更少的大算子**（减少中间张量写回，降低带宽压力）：
  - bias+激活融合；
  - 统一广播常量，减少 repeat/reshape；
  - 能用 `inplace` 的尽量用 inplace。
- **针对固定形状的卷积/FFT 路径做专用实现**：
  - 项目已有 STFT/ISTFT 自定义算子优化经验，可以进一步做到：
    - 预计算更多常量表；
    - 减少 trig/复杂数展开；
    - 在保证 C++17 的前提下，用手写 SIMD（需提供纯标量回退）。
  - 注意：该类优化不需要第三方库，但实现复杂，必须配套回归与开关。

预期收益：如果热点确实集中在少数路径，这类优化可能带来“倍数级”提升；但投入与风险最大。

### P6：长文本的批处理/打包策略（减少重复图执行次数）

`kokoro_runner::generate()` 里已经提示“chunking vs batching”的权衡。可探索：

- **多句合并成尽量少的 chunk**：在不超过 max context 的前提下，把短句合并，减少 `duration+generator` 调用次数。
- **批量 duration 预测**：对多个 clause 组成 batch（需要图支持 batch 维度/attention mask），一次性算多个序列。

预期收益：长文本多段场景端到端更明显；但会引入图结构调整与显存/内存压力评估。

## 4. 参考 whisper.cpp：可直接借鉴的“工程套路”

这些点在 `参考项目/whisper.cpp` 已经验证过工程可行性，建议按需迁移到 TTS：

- `参考项目/whisper.cpp/src/whisper.cpp`：
  - `ggml_graph_compute_helper(...)`：在 compute 前对 backend 统一设置线程数，compute 后 reset。
  - `struct whisper_sched` + `whisper_sched_graph_init(...)`：一次性初始化 scheduler/meta，并通过一次 alloc_graph 获取稳定的 buffer 尺寸与布局。

## 5. 代办清单（按阶段推进）

> 说明：下面用 `- [ ]` 作为可打勾的执行项；每完成一项建议在“优化记录”里写下 RTF 与对比结论。

### A. 基线与剖析（必做）

- [ ] 补充一组 Kokoro CPU 基准用例（短/中/长 + 中/英/混合），统一用 `tts-bench` 或 `perf_battery` 可复现。
- [ ] 增加 `ggml_cont` 统计（次数/字节/Top-K 触发点），用于定位大拷贝。
- [ ] 增加“阶段级”统计：duration 总耗时、generator 总耗时、append/写 wav 耗时。

### B. 内存与拷贝（低风险高收益）

- [ ] 改造多段输出拼接：用 `std::vector<float>` 聚合，消除 `append_to_response()` 的反复全量 memcpy。
- [ ] 输出 buffer 清零策略优化：确认是否必须 `buffer_clear`，能省则省或仅清尾部。
- [ ] duration_ids/noise/window 等输入：减少中间临时分配与二次拷贝（可写则直接写）。

### C. 图构建与复用（降低固定开销与抖动）

- [ ] 引入“可控 worst-case 图”初始化：启动阶段做一次 `reserve/alloc_graph`，避免运行时 reallocate。
- [ ] 将不随输入变化的常量 tensor（窗函数/索引/标量）统一缓存到 context，避免重复构建/写入。

### D. CPU 线程策略（通常有 10%~30%）

- [ ] 增加 generator 阶段线程上限开关（如 `TTS_CPU_THREADS_GEN`），并记录不同线程数的 RTF 曲线，确定默认值。
- [ ] 对 duration/generator 分别设置线程：duration 小线程、generator 大线程，避免互相抢带宽。

### E. 热点路径结构性优化（高风险高收益）

- [ ] 对 compute 做算子级热点定位（至少能打印 top-k op 类型/耗时/张量规模）。
- [ ] 针对 top1~top3 热点算子，评估能否通过“布局统一/融合算子/减少 cont”降低总带宽。
- [ ] 若热点在 STFT/ISTFT/卷积类：继续专用化实现（预计算、SIMD、减少 trig），并提供开关回退。

### F. 长文本批处理（面向吞吐）

- [ ] 改进 chunk 合并策略：尽量把多个短句合并到同一 chunk（不超过 max context），减少图执行次数。
- [ ] 评估 batch 化 duration（多序列一次图）可行性与内存代价，做 PoC 版本对比吞吐。

## 6. 优化记录（建议每次只改一个变量）

| 日期 | 变更点 | 模型 | 线程 | RTF | 备注 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

