# CosyVoice3 CPU 推理待办清单（预制 voice pack / 非流式）

> 目标：在 tts.cpp 中新增 CosyVoice3 的 **CPU 非流式** 推理，并将 **LLM/Flow/HiFT + 预制 voice pack** 全部转换为 **单文件 GGUF**。
> 约束：C++17、尽量不引入第三方库、尽量不改 ggml。
> 备注：Qwen2 结构与权重映射 **直接参考** `参考项目/llama.cpp` 的实现。

## 0. 需求与命名对齐（先做）
- [ ] 梳理 `参考项目/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml` 的关键超参：`sample_rate=24000`、`token_mel_ratio=2`、`n_timesteps=10`、`pre_lookahead_len=3`、`llm_input_size=896`、`llm_output_size=896`、`speech_token_size=6561` 等。
- [ ] 设计 GGUF 架构名与键前缀：建议 `general.architecture=cosyvoice3`，权重前缀使用 `cosyvoice3.llm.* / cosyvoice3.flow.* / cosyvoice3.hift.*`。
- [ ] 设计 voice pack 结构（预制）：至少包含 `prompt_text_token`、`prompt_text_token_len`、`prompt_speech_token`、`prompt_speech_token_len`、`prompt_speech_feat`、`prompt_speech_feat_len`、`embedding`。
- [ ] 统一 voice pack 命名规则：例如 `cosyvoice3.voices` + `cosyvoice3.voice_tensors.<voice>.<field>`。

## 1. 预制 voice pack（离线生成）
- [ ] 用 `参考项目/CosyVoice` 的 Python 前端离线提取：
  - `speech_token`（`speech_tokenizer_v3.onnx`）
  - `speech_feat`（mel 特征）
  - `embedding`（`campplus.onnx`）
  - `prompt_text_token`（Qwen2 tokenizer + CosyVoice3 追加 special tokens）
- [ ] 将上述结果保存为中间文件（如 `.npz`），供 GGUF 转换脚本读取。
- [ ] 约定每个 voice pack 的 prompt 文本与 prompt 音频一一对应（便于复现与后续比对）。

## 2. GGUF 转换脚本（py-gguf）
- [x] 新增 `py-gguf/convert_cosyvoice3_to_gguf` 与 `tts_encoders/cosyvoice3_gguf_encoder.py`。
- [x] 读取 `llm.pt / flow.pt / hift.pt`，并完成 **权重映射 + 权重规整**：
  - [x] HiFT/Flow 中的 `weight_norm` 折叠到卷积权重（避免推理侧再实现 weight_norm）。
  - [x] 将 `CosyVoice-BlankEN` 的 `vocab.json / merges.txt` 写入 GGUF 的 BPE 词表。
  - [x] 将 CosyVoice3 追加的 `special tokens` 按固定顺序拼接到词表尾部，确保 ID 对齐。
- [x] 写入必要元信息（示例）：
  - [x] `cosyvoice3.sample_rate`、`cosyvoice3.token_mel_ratio`、`cosyvoice3.pre_lookahead_len`
  - [x] `cosyvoice3.llm.*`（heads、layers、rope_theta 等）
  - [x] `cosyvoice3.flow.*`（DiT depth/heads/ff_mult 等）
  - [x] `cosyvoice3.hift.*`（upsample rates、n_fft/hop_len、audio_limit 等）
- [ ] 写入常量张量：
  - [x] CFM 固定噪声（`rand_noise`）
  - [x] STFT Hann 窗（`stft_window`）
  - [ ] `window_squared_sum`（推理侧按长度动态计算）
- [x] 写入 voice pack 张量（单文件 GGUF 内部管理，多 voice 可扩展）。

## 3. C++ 端模型与加载器
- [ ] 在 `include/common.h` 与 `src/models/loaders.*` 中注册 `cosyvoice3` 架构与 loader。
- [ ] 新增目录 `src/models/cosyvoice3/`，包含 `loader.cpp / model.h / model.cpp`。
- [ ] 设计 `cosyvoice3_model / cosyvoice3_runner`：
  - 读取 GGUF 超参与 voice pack 张量。
  - 构建 CPU 计算图（非流式）。

## 4. Qwen2 LLM（CPU）
- [ ] 直接参考 `参考项目/llama.cpp` 的 Qwen2 结构与权重布局（RMSNorm、SwiGLU、RoPE、GQA/KV cache）。
- [ ] 复用项目内的 BPE tokenizer（`bpe_tokenizer_from_gguf`），确保与 Qwen2 vocab/merges 对齐。
- [ ] 实现 RAS 采样（top_p/top_k + repetition window/tau_r），并对齐 CosyVoice3 的 stop tokens（`speech_token_size` 到 `speech_token_size+199`）。

## 5. Flow（DiT + CFM）
- [ ] 实现 `PreLookaheadLayer`（1D conv + residual）。
- [ ] 实现 DiT 栈（InputEmbedding / CausalConvPositionEmbedding / AdaLayerNormZero / Rotary / Attention / FFN）。
- [ ] 实现 CFM Euler 解算 + CFG（cond/uncond 双 batch），并使用 GGUF 内置的固定噪声。
- [ ] 输出 mel 特征（`[1, 80, T]`）。

## 6. HiFT（vocoder）
- [ ] 实现 `CausalConvRNNF0Predictor`（右/左因果卷积）。
- [ ] 实现 `SourceModuleHnNSF`（SineGen2 + Snake），可复用 `util.cpp` 的 `snake_1d`。
- [ ] 实现 `conv_pre / upsample / resblocks / conv_post` + STFT/ISTFT。
- [ ] 输出 24kHz PCM 浮点音频并 clamp 到 `audio_limit`。

## 7. 端到端串联（非流式）
- [ ] 通过 voice pack 直接提供 `prompt_*` 与 `embedding`，跳过在线提取。
- [ ] 生成流程：`text -> LLM -> speech_token -> Flow -> mel -> HiFT -> audio`。
- [ ] 添加最小 CLI 路径（或复用现有 `tts-cli` 接口）用于本地验证。

## 8. 对齐与验证
- [ ] 使用 Python 参考实现生成一组对齐样例（短句 + 中英混合）。
- [ ] 逐模块对齐（LLM logits / Flow mel / HiFT audio），记录误差阈值与失败样例。
- [ ] 输出基准日志（RTF、耗时分段），为后续优化做基线。
