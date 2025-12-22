#ifndef util_h
#define util_h

#include "common.h"
#include "numbers_compat.h"
#include <math.h>
#include <functional>
#include <random>
#include <mutex>
#include <stdio.h>
#include <string>
#include <string_view>
#include <cstring>
#include <vector>
#include <stdint.h>
#include <sys/types.h>
#include "ggml-metal.h"
#include "ggml-vulkan.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-impl.h"
#include "ggml-cpp.h"

// C++17 兼容工具：老编译器/标准库没有 std::string(_view)::starts_with / ends_with（C++20 才加入）。
// 这里提供轻量替代，便于在不升标准的情况下继续使用 string_view 风格的前后缀判断。
inline bool tts_starts_with(std::string_view s, std::string_view prefix) noexcept {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

inline bool tts_ends_with(std::string_view s, std::string_view suffix) noexcept {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

#define TTS_ABORT(...) tts_abort(__FILE__, __LINE__, __VA_ARGS__)
#define TTS_ASSERT(x) if (!(x)) TTS_ABORT("TTS_ASSERT(%s) failed", #x)

struct model_tensor_meta {
	uint32_t n_tensors = 0;
	size_t n_bytes = 0;
};

// 说明：用于在“无自定义算子”图中注册常量输入（避免 Vulkan 读取未分配的 host 指针）。
struct tts_graph_const_input {
    struct ggml_tensor * tensor = nullptr;
    float value = 0.0f;
};

/**
 * Both of these random fill the tgt array with count random floating point values.
 * the default parameter values are consistent with pytorch random function defaults.
 */
void random_uniform_gen(int count, float * tgt, float min = 0.0f, float max = 1.0f);
void random_normal_gen(int count, float * tgt, float mean = 0.0f, float std = 1.0f);

std::pair<int, std::string> parse_layer_count(std::string name, int skip = 0);

struct model_tensor_meta compute_tensor_meta(std::string name_prefix, ggml_context * weight_ctx, std::function<void(ggml_tensor*)>* callback = nullptr);
struct ggml_tensor * snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a,
                              std::vector<tts_graph_const_input> * const_inputs = nullptr);
int search_for_gguf_keys(gguf_context * meta, std::vector<std::string> possible_keys);

// a simple window function for stft
void hann_window(size_t n_fft, std::vector<float>& tgt);

// currently this assumes a center view in which the output vector is reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames, float * tgt, float * window);

// these functions wrap the stft and istft ggml ops and compute the necessary view and division ops for their indepentent settings.
struct ggml_tensor * stft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided);
struct ggml_tensor * istft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window_squared_sum, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided);

// 使用标准 ggml 算子（conv/get_rows/数学算子）构建 STFT/ISTFT 计算图：
// - 不依赖 GGML_OP_CUSTOM，便于 Vulkan 后端完整执行。
// - forward_basis / inverse_basis 需由外部预先计算（形状 [n_fft, 1, 2*(n_fft/2+1)]）。
// - pad_indices 为反射 padding 的索引（形状 [padded_len, batch]，类型 I32）。
// - const_inputs 用于注册图中的标量常量（Vulkan 需要显式输入缓冲）。
struct ggml_tensor * stft_graph(
    ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * forward_basis,
    struct ggml_tensor * pad_indices,
    std::vector<tts_graph_const_input> * const_inputs,
    size_t n_fft,
    size_t hop,
    bool abs_and_angle,
    bool one_sided);

struct ggml_tensor * istft_graph(
    ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * window_squared_sum,
    struct ggml_tensor * inverse_basis,
    size_t n_fft,
    size_t hop,
    bool abs_and_angle,
    bool one_sided);

// This is a custom op for sine_generation in the Kokoro model.
void uv_noise_compute(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);

// This is a custom op for logit correction in the Dia model.
void cfg_scale(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);

struct ggml_tensor * reciprocal(ggml_context * ctx, struct ggml_tensor * x,
                                std::vector<tts_graph_const_input> * const_inputs = nullptr);

// ---------------------------
// ggml 后端选择（CPU/Metal/Vulkan）
// ---------------------------
//
// 设计说明：
// - 历史上项目用 `cpu_only` 这个 bool 来区分 “CPU / Metal” 两种模式；但 Vulkan 加入后，bool 无法表达更多后端。
// - 为了尽量少改动模型/loader 的函数签名，这里使用 thread_local 保存“当前线程的后端配置”。
// - 模型加载时（runner_from_file）会在当前线程设置该配置；各模型的 context / tts_model 在初始化加速后端时读取它。
// - 使用 thread_local 的原因：tts-server 等场景可能多线程并行加载/推理，避免全局变量竞争。

// 设置/获取当前线程的后端配置（仅影响当前线程后续的模型加载/推理上下文创建）。
void               tts_set_backend_config(const tts_backend_config & cfg);
tts_backend_config tts_get_backend_config();

// RAII：临时切换后端配置，作用域结束自动恢复。
struct tts_backend_config_guard {
    explicit tts_backend_config_guard(const tts_backend_config & cfg);
    ~tts_backend_config_guard();

    tts_backend_config_guard(const tts_backend_config_guard &)             = delete;
    tts_backend_config_guard & operator=(const tts_backend_config_guard &) = delete;

  private:
    tts_backend_config prev_;
};

// 根据当前线程的后端配置，初始化一个“加速后端”（Metal/Vulkan）。
// 返回 nullptr 表示当前构建或运行环境不支持所请求的后端。
ggml_backend_t tts_backend_init_accel();

/**
 * 转置卷积 1D（兼容旧版 ggml 的扩展签名）。
 *
 * 背景：
 * - 旧版项目使用的 ggml 曾经提供过带 output_padding / groups 的 `ggml_conv_transpose_1d` 扩展签名；
 * - ggml 0.9.4 将其收敛为仅 (stride, padding, dilation)，且当前实现还限制 `padding == 0 && dilation == 1`。
 *
 * 目标：
 * - 在“不修改 ggml 源码”的前提下，尽量复用 ggml 0.9.4 现有实现；
 * - 对于 groups==1 的常见场景：调用 ggml 原生转置卷积（padding 固定为 0）后，通过 view + cont 做裁剪，等价实现 padding/output_padding；
 * - 对于 groups>1（主要是 Kokoro 的 depthwise 转置卷积）：
 *   在项目侧用 GGML_OP_CUSTOM 补齐一个 CPU 版本实现，保证功能正确。
 */
// 一维卷积（项目侧封装，补齐 F32 卷积核支持）。
struct ggml_tensor * tts_conv_1d(
    ggml_context * ctx,
    struct ggml_tensor * a,  // kernel
    struct ggml_tensor * b,  // input
    int s0,                  // stride
    int p0,                  // padding
    int d0);                 // dilation

struct ggml_tensor * tts_conv_transpose_1d(
    ggml_context * ctx,
    struct ggml_tensor * a,  // kernel
    struct ggml_tensor * b,  // input
    int s0,                  // stride
    int p0,                  // padding
    int d0,                  // dilation
    int output_padding = 0,
    int groups = 1);

bool has_suffix(std::string value, std::string suffix);
bool has_prefix(std::string value, std::string prefix);

std::vector<std::string> split(std::string target, std::string split_on, bool include_split_characters = false);
std::vector<std::string> split(std::string target, const char split_on, bool include_split_characters = false);
std::string strip(std::string target, std::string vals = " ");
std::string replace_any(std::string target, std::string to_replace, std::string replacement);

[[noreturn]] void tts_abort(const char * file, int line, const char * fmt, ...);

// 统一初始化 ggml 计时（只做一次），避免多处重复调用 ggml_time_init 导致计时基准被重置。
void tts_time_init_once();

#endif
