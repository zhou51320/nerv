#include "model.h"
#include "numbers_compat.h"
#include "multilingual.h"
#include "zh_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

// 计时工具：为了分析 CPU 推理慢在哪里（中文前端/英文 phonemizer、duration、generator 等），
// 这里提供一个统一的 us 计时入口，并用环境变量开关避免默认污染输出/引入额外开销。
static int64_t tts_time_us() {
    tts_time_init_once();
    return ggml_time_us();
}

// 说明：在 Vulkan 后端下，默认输入张量会被调度器分配到 CPU，
// 这会导致 Vulkan 图里混入非 Vulkan buffer，从而在后端构图时崩溃。
// 这里主动将输入叶子节点绑定到 Vulkan 后端，避免 CPU/Vulkan buffer 混用。
static bool tts_backend_is_vulkan(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    const char * name = ggml_backend_name(backend);
    return name && std::strncmp(name, "Vulkan", 6) == 0;
}

static bool tts_env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

// 说明：带默认值的布尔环境变量读取。
// - 未设置：返回 default_value
// - 设置为 0/off/false：返回 false
// - 其他非空值：返回 true
static bool tts_env_truthy_default(const char * name, bool default_value) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return default_value;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

// 说明：Kokoro 的 tokenizer 是“按 UTF-8 字符（codepoint）”做单字切分的：
// - 对英文/标点：1 字节 == 1 token（与 std::string::size() 一致）
// - 对中文/注音符号等：通常是 3 字节/字符，但仍然应该算作 1 token
// 因此在判断是否需要 chunking（是否超过 max_context_length）时，不能用字节长度，
// 必须用 UTF-8 字符数量，否则会在中文场景下“误判为超长”，导致不必要的分句/多次推理，
// 进而显著拉低 Vulkan 端到端吞吐（尤其是输出回读开销会被放大）。
static size_t tts_utf8_codepoint_count(std::string_view s) {
    size_t n = 0;
    for (unsigned char c : s) {
        // UTF-8 continuation byte 形如 10xxxxxx（0x80..0xBF），不计为新字符起始。
        if ((c & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

static int tts_env_int_default(const char * name, int default_value) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return default_value;
    }
    char * end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v) {
        return default_value;
    }
    if (x <= 0) {
        return default_value;
    }
    // 说明：避免输入超大值导致不必要的调度开销；上限取一个足够大的安全值。
    const long capped = std::min<long>(x, 256);
    return (int) capped;
}

static int kokoro_duration_threads(const runner_context * kctx) {
    const int max_threads = (kctx && kctx->n_threads > 0) ? kctx->n_threads : 1;
    // 说明：时长图以小矩阵/多分支算子为主，线程数过多容易被调度开销淹没；
    // 默认限制到 4（经验值），如需调整可设置环境变量：
    // - TTS_CPU_THREADS_DURATION=1..N
    const int def = std::min(max_threads, 4);
    const int v = tts_env_int_default("TTS_CPU_THREADS_DURATION", def);
    return std::max(1, std::min(v, max_threads));
}

// 说明：仅在 Vulkan 后端时才需要准备额外的权重版本（如 F32 拷贝/展开后的 depthwise kernel）。
static bool kokoro_use_vk_weights(const kokoro_model * model) {
    return model && tts_backend_is_vulkan(model->backend);
}

// 说明：前向声明（下面的 LSTM 融合工具函数需要用到）。
static void kokoro_copy_to_f32(const ggml_tensor * src, float * dst, size_t n);

// 说明：判断某个张量是否位于 Vulkan buffer 中。
// - build_lstm_run() 属于“图构建期”工具函数，拿不到 runner_context/backend，因此只能从权重 tensor 的 buffer 类型推断。
// - ggml-vulkan 当前不支持 GGML_OP_SET（但支持 SET_ROWS），因此 LSTM 的“set 输出预分配”只能在 CPU 下启用。
static bool kokoro_tensor_on_vulkan(const ggml_tensor * t) {
    if (!t || !t->buffer) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    const char * name = ggml_backend_buft_name(buft);
    return name && std::strncmp(name, "Vulkan", 6) == 0;
}

// 说明：为 Vulkan 额外权重分配独立 buffer，避免挤占主权重 buffer 的预留空间。
static ggml_backend_buffer_t kokoro_alloc_extra_buffer(kokoro_model * model, size_t size, const char * tag) {
    if (!model || !model->buffer) {
        TTS_ABORT("Vulkan 权重额外 buffer 分配失败：model/buffer 未初始化。\n");
    }
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(model->buffer, size);
    if (!buf) {
        TTS_ABORT("Vulkan 权重额外 buffer 分配失败：%s (size=%zu bytes)\n", tag ? tag : "unknown", size);
    }
    model->extra_buffers.push_back(buf);
    return buf;
}

// --------------------------- LSTM gate 融合（CPU） ---------------------------
// 说明：
// - Kokoro 的多处 LSTM 在 CPU 推理中占据可观时间；原实现每个 gate 做一次 mul_mat，并逐步 concat 输出，
//   会产生大量 GEMM 重复与 O(T^2) 的拷贝。
// - 这里在“模型加载后”把 I/F/G/O 四个 gate 的权重按输出维拼接成一个大矩阵，从而：
//   - 输入投影：4 次 mul_mat -> 1 次 mul_mat
//   - 隐状态投影：每步 4 次 mul_mat -> 每步 1 次 mul_mat
//   - bias：预先做 b_x + b_h，推理时少一次 add
// - 默认仅在 CPU 后端启用（Vulkan 下避免额外 buffer/拷贝与潜在不支持的算子组合）。
static bool kokoro_env_cpu_lstm_fuse() {
    // 默认开启；如需回退可设置：TTS_CPU_LSTM_FUSE=0
    return tts_env_truthy_default("TTS_CPU_LSTM_FUSE", true);
}

static bool kokoro_env_cpu_lstm_set_output() {
    // 默认开启；如需回退可设置：TTS_CPU_LSTM_SET_OUTPUT=0
    return tts_env_truthy_default("TTS_CPU_LSTM_SET_OUTPUT", true);
}

static void kokoro_tensor_get_bytes(const ggml_tensor * t, void * dst, size_t bytes) {
    if (!t || !dst || bytes == 0) {
        return;
    }
    if (t->buffer) {
        ggml_backend_tensor_get(t, dst, 0, bytes);
    } else {
        std::memcpy(dst, t->data, bytes);
    }
}

static ggml_tensor * kokoro_alloc_and_set_extra_tensor(kokoro_model * model, ggml_tensor * t, const void * data, size_t bytes, const char * name) {
    if (!model || !t) {
        return nullptr;
    }
    if (name && *name) {
        ggml_set_name(t, name);
    }
    ggml_backend_buffer_t buf = kokoro_alloc_extra_buffer(model, ggml_nbytes(t), name ? name : "kokoro.extra");
    t->buffer = buf;
    t->data = ggml_backend_buffer_get_base(buf);
    if (data && bytes > 0) {
        ggml_backend_tensor_set(t, data, 0, bytes);
    }
    return t;
}

static bool kokoro_is_f16_or_f32(ggml_type t) {
    return t == GGML_TYPE_F16 || t == GGML_TYPE_F32;
}

static ggml_tensor * kokoro_fuse_mat4_outdim(kokoro_model * model,
                                             const ggml_tensor * w0,
                                             const ggml_tensor * w1,
                                             const ggml_tensor * w2,
                                             const ggml_tensor * w3,
                                             const char * name) {
    if (!model || !w0 || !w1 || !w2 || !w3) {
        return nullptr;
    }
    if (ggml_n_dims(w0) != 2 || ggml_n_dims(w1) != 2 || ggml_n_dims(w2) != 2 || ggml_n_dims(w3) != 2) {
        return nullptr;
    }
    if (w0->type != w1->type || w0->type != w2->type || w0->type != w3->type) {
        return nullptr;
    }
    if (!kokoro_is_f16_or_f32(w0->type)) {
        return nullptr;
    }
    if (w0->ne[0] != w1->ne[0] || w0->ne[0] != w2->ne[0] || w0->ne[0] != w3->ne[0]) {
        return nullptr;
    }
    if (w0->ne[1] != w1->ne[1] || w0->ne[1] != w2->ne[1] || w0->ne[1] != w3->ne[1]) {
        return nullptr;
    }
    // 权重张量应为连续内存；若不是连续，直接跳过融合（避免错误拼接）。
    if (!ggml_is_contiguous(w0) || !ggml_is_contiguous(w1) || !ggml_is_contiguous(w2) || !ggml_is_contiguous(w3)) {
        return nullptr;
    }

    const size_t part_bytes = ggml_nbytes(w0);
    std::vector<uint8_t> p0(part_bytes);
    std::vector<uint8_t> p1(part_bytes);
    std::vector<uint8_t> p2(part_bytes);
    std::vector<uint8_t> p3(part_bytes);
    kokoro_tensor_get_bytes(w0, p0.data(), part_bytes);
    kokoro_tensor_get_bytes(w1, p1.data(), part_bytes);
    kokoro_tensor_get_bytes(w2, p2.data(), part_bytes);
    kokoro_tensor_get_bytes(w3, p3.data(), part_bytes);

    ggml_tensor * dst = ggml_new_tensor_2d(model->ctx, w0->type, w0->ne[0], w0->ne[1] * 4);
    std::vector<uint8_t> out(part_bytes * 4);
    std::memcpy(out.data() + part_bytes * 0, p0.data(), part_bytes);
    std::memcpy(out.data() + part_bytes * 1, p1.data(), part_bytes);
    std::memcpy(out.data() + part_bytes * 2, p2.data(), part_bytes);
    std::memcpy(out.data() + part_bytes * 3, p3.data(), part_bytes);

    return kokoro_alloc_and_set_extra_tensor(model, dst, out.data(), out.size(), name);
}

static ggml_tensor * kokoro_fuse_lstm_bias4_sum(kokoro_model * model,
                                                const ggml_tensor * bx0, const ggml_tensor * bh0,
                                                const ggml_tensor * bx1, const ggml_tensor * bh1,
                                                const ggml_tensor * bx2, const ggml_tensor * bh2,
                                                const ggml_tensor * bx3, const ggml_tensor * bh3,
                                                const char * name) {
    if (!model || !bx0 || !bh0 || !bx1 || !bh1 || !bx2 || !bh2 || !bx3 || !bh3) {
        return nullptr;
    }
    const size_t n0 = ggml_nelements(bx0);
    if (n0 == 0 ||
        ggml_nelements(bh0) != n0 ||
        ggml_nelements(bx1) != n0 || ggml_nelements(bh1) != n0 ||
        ggml_nelements(bx2) != n0 || ggml_nelements(bh2) != n0 ||
        ggml_nelements(bx3) != n0 || ggml_nelements(bh3) != n0) {
        return nullptr;
    }

    std::vector<float> out(n0 * 4, 0.0f);
    std::vector<float> bx(n0);
    std::vector<float> bh(n0);

    kokoro_copy_to_f32(bx0, bx.data(), n0);
    kokoro_copy_to_f32(bh0, bh.data(), n0);
    for (size_t i = 0; i < n0; ++i) {
        out[i] = bx[i] + bh[i];
    }

    kokoro_copy_to_f32(bx1, bx.data(), n0);
    kokoro_copy_to_f32(bh1, bh.data(), n0);
    for (size_t i = 0; i < n0; ++i) {
        out[n0 + i] = bx[i] + bh[i];
    }

    kokoro_copy_to_f32(bx2, bx.data(), n0);
    kokoro_copy_to_f32(bh2, bh.data(), n0);
    for (size_t i = 0; i < n0; ++i) {
        out[n0 * 2 + i] = bx[i] + bh[i];
    }

    kokoro_copy_to_f32(bx3, bx.data(), n0);
    kokoro_copy_to_f32(bh3, bh.data(), n0);
    for (size_t i = 0; i < n0; ++i) {
        out[n0 * 3 + i] = bx[i] + bh[i];
    }

    ggml_tensor * dst = ggml_new_tensor_1d(model->ctx, GGML_TYPE_F32, (int64_t) (n0 * 4));
    return kokoro_alloc_and_set_extra_tensor(model, dst, out.data(), out.size() * sizeof(float), name);
}

static void kokoro_try_fuse_lstm_cell(kokoro_model * model, int lstm_index, int cell_index, lstm_cell * cell, bool reversed) {
    if (!model || !cell) {
        return;
    }
    // 仅在 CPU 后端启用：避免 Vulkan 额外 buffer/拷贝与不支持的算子（GGML_OP_SET）组合风险。
    if (!kokoro_env_cpu_lstm_fuse() || kokoro_use_vk_weights(model)) {
        return;
    }

    std::vector<ggml_tensor *> & w = reversed ? cell->reverse_weights : cell->weights;
    std::vector<ggml_tensor *> & b = reversed ? cell->reverse_biases : cell->biases;
    if (w.size() < 8 || b.size() < 8) {
        return;
    }
    for (int i = 0; i < 8; ++i) {
        if (!w[i] || !b[i]) {
            return;
        }
    }

    // 如果权重已经位于 Vulkan buffer，说明当前模型不是 CPU 后端，直接跳过。
    if (kokoro_tensor_on_vulkan(w[0])) {
        return;
    }

    char name_wx[128];
    char name_wh[128];
    char name_b[128];
    std::snprintf(name_wx, sizeof(name_wx), "kokoro.lstm%d.cell%d.%s_fused_wx", lstm_index, cell_index, reversed ? "rev" : "fwd");
    std::snprintf(name_wh, sizeof(name_wh), "kokoro.lstm%d.cell%d.%s_fused_wh", lstm_index, cell_index, reversed ? "rev" : "fwd");
    std::snprintf(name_b,  sizeof(name_b),  "kokoro.lstm%d.cell%d.%s_fused_b",  lstm_index, cell_index, reversed ? "rev" : "fwd");

    ggml_tensor * fused_wx = kokoro_fuse_mat4_outdim(model, w[0], w[2], w[4], w[6], name_wx);
    ggml_tensor * fused_wh = kokoro_fuse_mat4_outdim(model, w[1], w[3], w[5], w[7], name_wh);
    ggml_tensor * fused_b  = kokoro_fuse_lstm_bias4_sum(model,
                                                       /*I=*/b[0], b[1],
                                                       /*F=*/b[2], b[3],
                                                       /*G=*/b[4], b[5],
                                                       /*O=*/b[6], b[7],
                                                       name_b);
    if (!fused_wx || !fused_wh || !fused_b) {
        return;
    }

    lstm_cell::fused_gates & dst = reversed ? cell->fused_reverse : cell->fused;
    dst.w_x = fused_wx;
    dst.w_h = fused_wh;
    dst.b   = fused_b;
}

static ggml_tensor * kokoro_new_tensor_like(ggml_context * ctx, ggml_type type, const ggml_tensor * src) {
    return ggml_new_tensor(ctx, type, ggml_n_dims(src), src->ne);
}

// 说明：将权重读到 CPU 并转换为 F32，兼容 Vulkan 对部分算子的类型要求。
static void kokoro_copy_to_f32(const ggml_tensor * src, float * dst, size_t n) {
    if (!src || !dst || n == 0) {
        return;
    }

    if (src->type == GGML_TYPE_F32) {
        if (src->buffer) {
            ggml_backend_tensor_get(src, dst, 0, n * sizeof(float));
        } else {
            std::memcpy(dst, src->data, n * sizeof(float));
        }
        return;
    }

    if (src->type != GGML_TYPE_F16) {
        TTS_ABORT("Vulkan 权重转换仅支持 F16/F32，当前类型=%d。\n", (int) src->type);
    }
    if (src->buffer) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(src, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) {
            dst[i] = ggml_fp16_to_fp32(tmp[i]);
        }
        return;
    }
    const ggml_fp16_t * raw = (const ggml_fp16_t *) src->data;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = ggml_fp16_to_fp32(raw[i]);
    }
}

// 说明：为 Vulkan 生成 F32 权重拷贝（用于 conv_transpose_1d 等仅支持 F32 的算子）。
static ggml_tensor * kokoro_make_f32_weight(kokoro_model * model, const ggml_tensor * src, const char * name) {
    if (!model || !src) {
        return nullptr;
    }
    if (src->type == GGML_TYPE_F32) {
        return nullptr;
    }
    ggml_tensor * dst = kokoro_new_tensor_like(model->ctx, GGML_TYPE_F32, src);
    if (name && *name) {
        ggml_set_name(dst, name);
    }
    const size_t n = (size_t) ggml_nelements(src);
    std::vector<float> data(n);
    kokoro_copy_to_f32(src, data.data(), n);

    ggml_backend_buffer_t buf = kokoro_alloc_extra_buffer(model, ggml_nbytes(dst), name);
    dst->buffer = buf;
    dst->data = ggml_backend_buffer_get_base(buf);
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(float));
    return dst;
}

static float kokoro_read_3d_f32(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
    const char * base = (const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2];
    if (t->type == GGML_TYPE_F32) {
        return *(const float *) base;
    }
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) base);
    }
    TTS_ABORT("Vulkan depthwise 权重展开仅支持 F16/F32，当前类型=%d。\n", (int) t->type);
    return 0.0f;
}

// 说明：将 depthwise 转置卷积权重展开为“全连接通道”形式（K, C, C），
// 从而可直接使用 ggml_conv_transpose_1d（groups==1）。
//
// 性能/兼容收益：
// - Vulkan：ggml-vulkan 当前不支持 groups>1 的 conv_transpose，展开后才能走 Vulkan；
// - CPU：避免 GGML_OP_CUSTOM 的 groups>1 转置卷积实现（该实现为朴素循环，通常明显更慢）。
static ggml_tensor * kokoro_expand_depthwise_pool_weight(kokoro_model * model, const ggml_tensor * src, const char * name) {
    if (!model || !src) {
        return nullptr;
    }
    if (src->ne[1] != 1 || src->ne[3] != 1) {
        return nullptr;
    }
    const int64_t K = src->ne[0];
    const int64_t C = src->ne[2];
    if (K <= 0 || C <= 0) {
        return nullptr;
    }

    const size_t elems = (size_t) K * (size_t) C * (size_t) C;
    const size_t bytes = elems * sizeof(float);
    // 防御：避免极端配置下的过大展开导致内存爆炸。
    if (bytes > (size_t) 128 * 1024 * 1024) {
        fprintf(stderr, "[kokoro] depthwise 权重展开过大（%.2f MiB），继续使用 CPU 自定义算子。\n", bytes / (1024.0 * 1024.0));
        return nullptr;
    }

    ggml_tensor * dst = ggml_new_tensor_3d(model->ctx, GGML_TYPE_F32, K, C, C);
    if (name && *name) {
        ggml_set_name(dst, name);
    }
    std::vector<float> data(elems, 0.0f);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t k = 0; k < K; ++k) {
            const float w = kokoro_read_3d_f32(src, k, 0, c);
            const size_t idx = (size_t) k + (size_t) K * ((size_t) c + (size_t) C * (size_t) c);
            data[idx] = w;
        }
    }

    ggml_backend_buffer_t buf = kokoro_alloc_extra_buffer(model, ggml_nbytes(dst), name);
    dst->buffer = buf;
    dst->data = ggml_backend_buffer_get_base(buf);
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(float));
    return dst;
}

// --------------------------- STFT/ISTFT 基矩阵（conv 形式） ---------------------------
// 说明：
// - 参考 whisper.cpp / HiFi-GAN 的做法：用 conv_1d/conv_transpose_1d 实现 STFT/ISTFT。
// - 这样可以完全使用 ggml 标准算子，避免 GGML_OP_CUSTOM 在 Vulkan 上的崩溃。
// - n_fft 很小（Kokoro 默认 20），用 CPU 做一次矩阵伪逆即可。
static bool kokoro_invert_square(std::vector<double> & mat, size_t n) {
    std::vector<double> inv(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        inv[i * n + i] = 1.0;
    }

    for (size_t i = 0; i < n; ++i) {
        // 选主元，避免数值不稳定
        size_t pivot = i;
        double pivot_val = std::abs(mat[i * n + i]);
        for (size_t r = i + 1; r < n; ++r) {
            const double v = std::abs(mat[r * n + i]);
            if (v > pivot_val) {
                pivot = r;
                pivot_val = v;
            }
        }
        if (pivot_val < 1e-12) {
            return false;
        }
        if (pivot != i) {
            for (size_t c = 0; c < n; ++c) {
                std::swap(mat[i * n + c], mat[pivot * n + c]);
                std::swap(inv[i * n + c], inv[pivot * n + c]);
            }
        }

        const double diag = mat[i * n + i];
        for (size_t c = 0; c < n; ++c) {
            mat[i * n + c] /= diag;
            inv[i * n + c] /= diag;
        }

        for (size_t r = 0; r < n; ++r) {
            if (r == i) {
                continue;
            }
            const double factor = mat[r * n + i];
            if (factor == 0.0) {
                continue;
            }
            for (size_t c = 0; c < n; ++c) {
                mat[r * n + c] -= factor * mat[i * n + c];
                inv[r * n + c] -= factor * inv[i * n + c];
            }
        }
    }

    mat.swap(inv);
    return true;
}

static void kokoro_build_stft_basis(
    const std::vector<float> & window,
    size_t n_fft,
    std::vector<float> & forward_out,
    std::vector<float> & inverse_out) {
    const size_t cutoff = n_fft / 2 + 1;
    const size_t rows = cutoff * 2;
    const double two_pi = 2.0 * std::numbers::pi;

    if (window.size() != n_fft) {
        TTS_ABORT("STFT window size mismatch: got=%zu expected=%zu\n", window.size(), n_fft);
    }

    // A = [cos; -sin]，形状 [rows, n_fft]
    std::vector<double> fourier(rows * n_fft, 0.0);
    for (size_t k = 0; k < cutoff; ++k) {
        for (size_t n = 0; n < n_fft; ++n) {
            const double angle = two_pi * double(k) * double(n) / double(n_fft);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            fourier[k * n_fft + n] = c;
            fourier[(k + cutoff) * n_fft + n] = -s;
        }
    }

    // pinv(A) = (A^T A)^-1 A^T
    std::vector<double> ata(n_fft * n_fft, 0.0);
    for (size_t i = 0; i < n_fft; ++i) {
        for (size_t j = 0; j < n_fft; ++j) {
            double sum = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                sum += fourier[r * n_fft + i] * fourier[r * n_fft + j];
            }
            ata[i * n_fft + j] = sum;
        }
    }
    if (!kokoro_invert_square(ata, n_fft)) {
        TTS_ABORT("STFT basis inversion failed: matrix is singular.\n");
    }

    // pinv = inv(A^T A) * A^T，形状 [n_fft, rows]
    std::vector<double> pinv(n_fft * rows, 0.0);
    for (size_t i = 0; i < n_fft; ++i) {
        for (size_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (size_t j = 0; j < n_fft; ++j) {
                sum += ata[i * n_fft + j] * fourier[r * n_fft + j];
            }
            pinv[i * rows + r] = sum;
        }
    }

    forward_out.resize(rows * n_fft);
    inverse_out.resize(rows * n_fft);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t n = 0; n < n_fft; ++n) {
            const float w = window[n];
            forward_out[r * n_fft + n] = float(fourier[r * n_fft + n] * w);
            inverse_out[r * n_fft + n] = float(pinv[n * rows + r] * w); // pinv 转置后乘 window
        }
    }
}

// 说明：部分 Vulkan 算子（如 scale/step）要求输入连续；这里封装一个“必要时再 cont”的小工具，减少无谓拷贝。
static ggml_tensor * tts_cont_if_needed(ggml_context * ctx, ggml_tensor * t) {
    if (!t) {
        return t;
    }
    return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

// 说明：允许运行时通过环境变量强制 Vulkan 生成阶段（便于回归测试）。
static bool kokoro_env_force_vulkan_gen() {
    const char * v = std::getenv("TTS_VK_FORCE_GEN");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

static void kokoro_force_inputs_backend(runner_context * ctx, ggml_cgraph * gf) {
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched) {
        return;
    }
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor * leaf = gf->leafs[i];
        if (!leaf) {
            continue;
        }
        if (leaf->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, leaf, ctx->backend);
        }
    }
}

// 说明：Vulkan 图中若包含自定义算子（CPU-only），会引入跨后端 buffer 混用，
// 在某些驱动下 ggml-vulkan 构图会崩溃；此处用于提前侦测并回退。
static bool kokoro_graph_has_custom_ops(const ggml_cgraph * gf) {
    if (!gf) {
        return false;
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * node = gf->nodes[i];
        if (!node) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_CUSTOM:
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
                return true;
            default:
                break;
        }
    }
    return false;
}

static void kokoro_force_graph_backend(runner_context * ctx, ggml_cgraph * gf, ggml_backend_t backend) {
    if (!ctx || !gf || !backend || !ctx->sched) {
        return;
    }
    for (int i = 0; i < gf->n_leafs; ++i) {
        ggml_tensor * leaf = gf->leafs[i];
        if (!leaf) {
            continue;
        }
        // 说明：若输入叶子已落在非主机 buffer（如 Vulkan 设备内存），
        // 强行改 CPU 会阻断调度器的拷贝逻辑，导致 CPU 直接访问 GPU 内存崩溃；
        // 因此对非主机 buffer 的叶子保持原后端，由调度器负责拷贝。
        if (leaf->buffer && !ggml_backend_buffer_is_host(leaf->buffer)) {
            continue;
        }
        ggml_backend_sched_set_tensor_backend(ctx->sched, leaf, backend);
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (node) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, backend);
        }
    }
}

static bool kokoro_tensor_depends_on_custom(const ggml_tensor * t, std::unordered_map<const ggml_tensor *, bool> & cache) {
    if (!t) {
        return false;
    }
    auto it = cache.find(t);
    if (it != cache.end()) {
        return it->second;
    }
    bool is_custom = t->op == GGML_OP_CUSTOM ||
                     t->op == GGML_OP_MAP_CUSTOM1 ||
                     t->op == GGML_OP_MAP_CUSTOM2 ||
                     t->op == GGML_OP_MAP_CUSTOM3;
    if (!is_custom) {
        if (t->view_src && kokoro_tensor_depends_on_custom(t->view_src, cache)) {
            is_custom = true;
        } else {
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (kokoro_tensor_depends_on_custom(t->src[i], cache)) {
                    is_custom = true;
                    break;
                }
            }
        }
    }
    cache[t] = is_custom;
    return is_custom;
}

static void kokoro_force_custom_views_cpu(runner_context * ctx, ggml_cgraph * gf) {
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched || !ctx->backend_cpu) {
        return;
    }
    std::unordered_map<const ggml_tensor *, bool> cache;
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node || !node->view_src) {
            continue;
        }
        if (kokoro_tensor_depends_on_custom(node->view_src, cache)) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
        }
    }
}

// 说明：优先将 Vulkan 可执行的非自定义算子固定到 Vulkan，避免整图被 CPU anchor。
// 说明：Vulkan 的 storage buffer offset 需要对齐；若算子输入来自 view 且 view_offs 未对齐，
// ggml-vulkan 会触发断言崩溃。这里在图编译前对“需要严格对齐”的算子做兜底处理：
// 一旦检测到未对齐 view 输入，就将该算子回退到 CPU。
static size_t kokoro_vk_tensor_view_offset_bytes(const ggml_tensor * t) {
    size_t offset = 0;
    const ggml_tensor * cur = t;
    while (cur && cur->view_src) {
        offset += cur->view_offs;
        cur = cur->view_src;
    }
    return offset;
}

static bool kokoro_vk_tensor_view_misaligned(const ggml_tensor * t, size_t alignment) {
    if (!t || alignment <= 1) {
        return false;
    }
    const size_t offset = kokoro_vk_tensor_view_offset_bytes(t);
    if (offset == 0) {
        return false;
    }
    return (offset % alignment) != 0;
}

static ggml_tensor * kokoro_vk_cont_if_misaligned_view(ggml_context * ctx, ggml_tensor * t, const kokoro_model * model) {
    if (!ctx || !t || !model || !kokoro_use_vk_weights(model) || !t->view_src) {
        return t;
    }
    const size_t alignment = model->buffer ? ggml_backend_buft_get_alignment(model->buffer) : 0;
    if (alignment <= 1) {
        return t;
    }
    if (!kokoro_vk_tensor_view_misaligned(t, alignment)) {
        return t;
    }
    // 说明：view 偏移未对齐时，先做一次 cont 复制，避免 Vulkan 断言。
    return ggml_cont(ctx, t);
}

static size_t kokoro_vk_required_alignment(runner_context * ctx) {
    if (!ctx || !ctx->backend) {
        return 0;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend);
    if (!dev) {
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft) {
        return 0;
    }
    return ggml_backend_buft_get_alignment(buft);
}

static bool kokoro_vk_op_requires_aligned_view(enum ggml_op op) {
    switch (op) {
        // 说明：以下算子在 Vulkan 后端支持 misalign offset（unary/binary/sum_rows/pad 等），
        // 允许 view_offs 非对齐；其余算子默认要求对齐以规避断言崩溃。
        // 注意：GGML_OP_UNARY 在 Vulkan 中使用通用 push constant，仍要求对齐，故不放行。
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD1:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_PAD:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_UPSCALE:
        case GGML_OP_ROLL:
        case GGML_OP_CLAMP:
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
        case GGML_OP_SET_ROWS:
            return false;
        default:
            return true;
    }
}

static bool kokoro_is_view_op(const ggml_tensor * node) {
    if (!node) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static void kokoro_force_vk_misaligned_view_ops_cpu(runner_context * ctx, ggml_cgraph * gf) {
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched || !ctx->backend_cpu) {
        return;
    }
    const size_t alignment = kokoro_vk_required_alignment(ctx);
    if (alignment <= 1) {
        return;
    }

    size_t forced = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node || kokoro_is_view_op(node)) {
            continue;
        }
        if (!kokoro_vk_op_requires_aligned_view(node->op)) {
            continue;
        }
        bool has_misaligned_view = false;
        const ggml_tensor * bad_src = nullptr;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (kokoro_vk_tensor_view_misaligned(node->src[j], alignment)) {
                has_misaligned_view = true;
                bad_src = node->src[j];
                break;
            }
        }
        if (!has_misaligned_view) {
            continue;
        }
        if (forced < 4 && bad_src) {
            const char * node_name = ggml_get_name(node);
            const char * src_name = ggml_get_name(bad_src);
            const char * src_op = ggml_op_name(bad_src->op);
            const char * src0_name = bad_src->view_src ? ggml_get_name(bad_src->view_src) : nullptr;
            const char * src0_op = bad_src->view_src ? ggml_op_name(bad_src->view_src->op) : nullptr;
            fprintf(stderr,
                    "[kokoro][vk-align] force_cpu op=%s node=%s src=%s src_op=%s view_offs=%zu align=%zu view_src=%s view_src_op=%s\n",
                    ggml_op_name(node->op),
                    node_name ? node_name : "(unnamed)",
                    src_name ? src_name : "(unnamed)",
                    src_op ? src_op : "(unknown)",
                    kokoro_vk_tensor_view_offset_bytes(bad_src),
                    alignment,
                    src0_name ? src0_name : "(unnamed)",
                    src0_op ? src0_op : "(unknown)");
            if (node->op == GGML_OP_UNARY) {
                fprintf(stderr,
                        "[kokoro][vk-align]   unary_op=%s\n",
                        ggml_unary_op_name(ggml_get_unary_op(node)));
            }
            fprintf(stderr,
                    "[kokoro][vk-align]   src_ne=(%lld,%lld,%lld,%lld) src_nb=(%zu,%zu,%zu,%zu)\n",
                    (long long) bad_src->ne[0], (long long) bad_src->ne[1],
                    (long long) bad_src->ne[2], (long long) bad_src->ne[3],
                    (size_t) bad_src->nb[0], (size_t) bad_src->nb[1],
                    (size_t) bad_src->nb[2], (size_t) bad_src->nb[3]);
            if (bad_src->view_src) {
                fprintf(stderr,
                        "[kokoro][vk-align]   view_src_ne=(%lld,%lld,%lld,%lld) view_src_nb=(%zu,%zu,%zu,%zu)\n",
                        (long long) bad_src->view_src->ne[0], (long long) bad_src->view_src->ne[1],
                        (long long) bad_src->view_src->ne[2], (long long) bad_src->view_src->ne[3],
                        (size_t) bad_src->view_src->nb[0], (size_t) bad_src->view_src->nb[1],
                        (size_t) bad_src->view_src->nb[2], (size_t) bad_src->view_src->nb[3]);
            }
        }
        ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
        ++forced;
    }

    if (forced > 0) {
        fprintf(stderr,
                "[kokoro] Vulkan 检测到 view 偏移未对齐，已将 %zu 个算子回退 CPU（对齐=%zu 字节）。\n",
                forced,
                alignment);
    }
}

// 说明：在图分配完成后检查 Vulkan 实际 buffer offset 对齐情况，必要时回退相关算子到 CPU。
static bool kokoro_vk_tensor_offset_misaligned(const ggml_tensor * t, size_t alignment) {
    if (!t || alignment <= 1) {
        return false;
    }
    const ggml_tensor * base_t = t;
    size_t view_offset = 0;
    while (base_t && base_t->view_src) {
        view_offset += base_t->view_offs;
        base_t = base_t->view_src;
    }
    ggml_backend_buffer_t buf = base_t->buffer;
    if (!buf || !base_t->data) {
        return false;
    }
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(buf);
    const uint8_t * data = (const uint8_t *) base_t->data;
    if (!base || !data) {
        return false;
    }
    const size_t offset = (size_t) (data - base) + view_offset;
    return (offset % alignment) != 0;
}

static size_t kokoro_vk_tensor_offset_bytes(const ggml_tensor * t) {
    if (!t) {
        return 0;
    }
    const ggml_tensor * base_t = t;
    size_t view_offset = 0;
    while (base_t && base_t->view_src) {
        view_offset += base_t->view_offs;
        base_t = base_t->view_src;
    }
    ggml_backend_buffer_t buf = base_t->buffer;
    if (!buf || !base_t->data) {
        return 0;
    }
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(buf);
    const uint8_t * data = (const uint8_t *) base_t->data;
    if (!base || !data) {
        return 0;
    }
    return (size_t) (data - base) + view_offset;
}

static void kokoro_vk_log_misaligned_node(const ggml_tensor * node, size_t alignment) {
    if (!node) {
        return;
    }
    const char * node_name = ggml_get_name(node);
    fprintf(stderr, "[kokoro][vk-align] op=%s node=%s\n", ggml_op_name(node->op), node_name ? node_name : "(unnamed)");
    if (kokoro_vk_tensor_offset_misaligned(node, alignment)) {
        fprintf(stderr,
                "[kokoro][vk-align]   dst offset=%zu align=%zu\n",
                kokoro_vk_tensor_offset_bytes(node),
                alignment);
        return;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * src = node->src[j];
        if (!src) {
            continue;
        }
        if (kokoro_vk_tensor_offset_misaligned(src, alignment)) {
            const char * src_name = ggml_get_name(src);
            fprintf(stderr,
                    "[kokoro][vk-align]   src%d=%s offset=%zu align=%zu\n",
                    j,
                    src_name ? src_name : "(unnamed)",
                    kokoro_vk_tensor_offset_bytes(src),
                    alignment);
            return;
        }
    }
}

static bool kokoro_collect_vk_misaligned_nodes(runner_context * ctx, ggml_cgraph * gf,
                                               std::vector<ggml_tensor *> & out_nodes) {
    out_nodes.clear();
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched) {
        return false;
    }
    const size_t alignment = kokoro_vk_required_alignment(ctx);
    if (alignment <= 1) {
        return false;
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node || kokoro_is_view_op(node)) {
            continue;
        }
        if (!kokoro_vk_op_requires_aligned_view(node->op)) {
            continue;
        }
        if (ggml_backend_sched_get_tensor_backend(ctx->sched, node) != ctx->backend) {
            continue;
        }
        bool misaligned = kokoro_vk_tensor_offset_misaligned(node, alignment);
        if (!misaligned) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (kokoro_vk_tensor_offset_misaligned(node->src[j], alignment)) {
                    misaligned = true;
                    break;
                }
            }
        }
        if (misaligned) {
            out_nodes.push_back(node);
            if (out_nodes.size() <= 3) {
                kokoro_vk_log_misaligned_node(node, alignment);
            }
        }
    }
    return !out_nodes.empty();
}

static void kokoro_force_supported_ops_backend(runner_context * ctx, ggml_cgraph * gf, ggml_backend_t backend) {
    if (!ctx || !gf || !backend || !ctx->sched) {
        return;
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_CUSTOM:
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
                continue;
            default:
                break;
        }
        // 说明：仅强制“收益最大且稳定”的算子走 Vulkan，降低跨后端拷贝与 Vulkan 图构建崩溃风险。
        switch (node->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_CONV_TRANSPOSE_1D:
                break;
            default:
                continue;
        }
        if (ggml_backend_supports_op(backend, node)) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, backend);
        }
    }
}

// 说明：为排查/规避部分 Vulkan 驱动在 STFT/ISTFT（conv 版）上的数值/实现问题，
// 提供一个“仅把 STFT/ISTFT 核心卷积算子固定到 CPU”的开关。
//
// 设计目标：
// - 尽量保持其余大头算子（mul_mat / upsample conv_transpose 等）继续走 Vulkan，以保留加速收益；
// - 只对 `kokoro.stft_forward_basis` / `kokoro.stft_inverse_basis` 这两组基矩阵相关的卷积算子做定点处理。
//
// 开关：
// - TTS_VK_STFT_CPU=1：将 STFT 的 ggml_conv_1d（kernel=stft_forward_basis）固定到 CPU
// - TTS_VK_ISTFT_CPU=1：将 ISTFT 的 ggml_conv_transpose_1d（kernel=stft_inverse_basis）固定到 CPU
// - TTS_VK_AUDIO_QUALITY=1：等价于同时开启上述两项
static void kokoro_force_vk_stft_istft_cpu_if_needed(runner_context * ctx, ggml_cgraph * gf, const kokoro_model * model) {
    if (!ctx || !gf || !model || !ctx->sched || !ctx->backend_cpu) {
        return;
    }
    if (!ctx->backend || !tts_backend_is_vulkan(ctx->backend)) {
        return;
    }

    const bool quality_mode = tts_env_truthy("TTS_VK_AUDIO_QUALITY");
    const bool stft_cpu = quality_mode || tts_env_truthy("TTS_VK_STFT_CPU");
    const bool istft_cpu = quality_mode || tts_env_truthy("TTS_VK_ISTFT_CPU");
    if (!stft_cpu && !istft_cpu) {
        return;
    }

    const ggml_tensor * fwd0 = model->stft_forward_basis;
    const ggml_tensor * fwd1 = model->stft_forward_basis_cpu;
    const ggml_tensor * inv0 = model->stft_inverse_basis;
    const ggml_tensor * inv1 = model->stft_inverse_basis_cpu;

    auto is_fwd_basis = [&](const ggml_tensor * t) {
        return t != nullptr && (t == fwd0 || (fwd1 != nullptr && t == fwd1));
    };
    auto is_inv_basis = [&](const ggml_tensor * t) {
        return t != nullptr && (t == inv0 || (inv1 != nullptr && t == inv1));
    };

    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node) {
            continue;
        }

        // ggml_conv_1d 当前是由 im2col + mul_mat 组合实现的（没有独立的 GGML_OP_CONV_1D）。
        // 因此这里通过“权重指针”精确定位 STFT 相关节点：
        // - im2col(src0==stft_forward_basis)
        // - mul_mat(src1 的 view_src==stft_forward_basis)
        if (stft_cpu) {
            if (node->op == GGML_OP_IM2COL && is_fwd_basis(node->src[0])) {
                ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
                continue;
            }
            if (node->op == GGML_OP_MUL_MAT && node->src[1] && is_fwd_basis(node->src[1]->view_src)) {
                ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
                continue;
            }
        }
        if (istft_cpu && node->op == GGML_OP_CONV_TRANSPOSE_1D && is_inv_basis(node->src[0])) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
            continue;
        }
    }
}

// 说明：自定义算子目前仅支持 CPU；在 Vulkan 图中显式固定它们到 CPU，避免被错误分配到 GPU 后端。
static void kokoro_force_custom_ops_cpu(runner_context * ctx, ggml_cgraph * gf) {
    if (!ctx || !gf || !ctx->backend || !tts_backend_is_vulkan(ctx->backend) || !ctx->sched || !ctx->backend_cpu) {
        return;
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        if (!node) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_CUSTOM:
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
                ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->backend_cpu);
                break;
            default:
                break;
        }
    }
}

static bool tts_timings_enabled() {
    const char * v = std::getenv("TTS_TIMINGS");
    // 计时默认开启：便于直接观测端到端推理性能。
    // 如需关闭，可在运行前设置环境变量：TTS_TIMINGS=0
    if (v == nullptr) {
        return true;
    }
    if (v[0] == '\0') {
        return true;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

static double us_to_ms(const int64_t us) {
    return us / 1000.0;
}

} // namespace

static struct ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens, ggml_tensor * positions, ggml_tensor * token_types) {
	struct ggml_tensor * tinpts = ggml_cont(ctx, ggml_get_rows(ctx, model->token_embd, input_tokens));
	struct ggml_tensor * pinpts = ggml_get_rows(ctx, model->position_embd, positions);

	struct ggml_tensor * inpts = ggml_cont(ctx, ggml_add(ctx, tinpts, pinpts));
	if (!model->static_token_types) {
		// Token type embeddings are actually static for kokoro at the moment, so we should never need to compute this on the fly.
		return ggml_add(ctx, inpts, ggml_get_rows(ctx, model->token_type_embd, token_types));
	}
	struct ggml_tensor * ainpts = ggml_add(ctx, inpts, model->static_token_type_values);

	struct ggml_tensor * out = ggml_cont(ctx, build_albert_norm(ctx, ainpts, model->input_norm_weight, model->input_norm_bias));
	return ggml_add(ctx, ggml_mul_mat(ctx, model->embd_hidden, out), model->embd_hidden_bias);
}

static struct ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias) {
	// this is the standard eps for Albert
	float eps = 0.000000000001;
    cur = ggml_norm(ctx, cur, eps);
    cur = ggml_cont(ctx, ggml_add(ctx, ggml_mul(ctx, cur, weight), bias));
    return cur;
}

static struct ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, const lstm_cell * cell, uint32_t sequence_length, bool reversed = false);

static struct ggml_tensor * build_lstm(ggml_context * ctx, ggml_tensor * input, lstm* rnn, uint32_t sequence_length, ggml_cgraph * gf) {
	struct ggml_tensor * resp = input;
	struct ggml_tensor * reverse_resp = input;

	// iterate over cells first so that at each pass to the next cell we have a fully formed vector (this improves performance as well as allocation for stacked lstms)
	for (int c = 0; c < rnn->cells.size(); c++) {
		ggml_build_forward_expand(gf, resp);
		resp = build_lstm_run(ctx, gf, resp, rnn->hidden[c], rnn->states[c], rnn->cells[c], sequence_length);
		if (rnn->bidirectional) {
			reverse_resp = build_lstm_run(ctx, gf, reverse_resp, rnn->hidden[c], rnn->states[c], rnn->cells[c], sequence_length, true);
		}
	}
	if (rnn->bidirectional) {
		resp = ggml_concat(ctx, resp, reverse_resp, 0);
	}
	return resp;
}

static struct ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, const lstm_cell * cell, uint32_t sequence_length, bool reversed) {
	const std::vector<ggml_tensor*> & weights = reversed ? cell->reverse_weights : cell->weights;
	const std::vector<ggml_tensor*> & biases  = reversed ? cell->reverse_biases  : cell->biases;

	const lstm_cell::fused_gates & fused = reversed ? cell->fused_reverse : cell->fused;
	const bool is_vulkan = !weights.empty() && kokoro_tensor_on_vulkan(weights[0]);
    auto vk_cont_view = [ctx, is_vulkan](ggml_tensor * t) -> ggml_tensor * {
        // 说明：Vulkan 对 view 偏移对齐要求严格，遇到 view 直接做一次 cont，避免断言崩溃。
        if (is_vulkan && t && t->view_src) {
            return ggml_cont(ctx, t);
        }
        return t;
    };

	// 说明：逐步 concat 会产生 O(T^2) 的拷贝；CPU 下用 ggml_set_2d_inplace 直接写入预分配输出，可显著降低开销。
	const bool use_set_output = !is_vulkan && kokoro_env_cpu_lstm_set_output();

	// 说明：若存在 fused gate 权重（仅 CPU 后端预处理生成），优先使用以减少 mul_mat 次数。
	const bool use_fused = !is_vulkan && fused.w_x && fused.w_h && fused.b;

	struct ggml_tensor * outputs = nullptr;
	if (use_set_output) {
		// 输出张量形状：[hidden, T, batch]；batch 一般为 1，但保持与输入一致。
		const int64_t hidden = weights.empty() ? 0 : weights[0]->ne[1];
		const int64_t batch  = input ? input->ne[2] : 1;
		outputs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, sequence_length, batch);
	}

	if (use_fused) {
		// gates_x: [4*hidden, T, batch]，已包含 (b_x + b_h) 的合并 bias。
		struct ggml_tensor * gates_x = ggml_add(ctx, ggml_mul_mat(ctx, fused.w_x, input), fused.b);
		const int64_t hidden = fused.w_h->ne[0];

		for (int index = 0; index < (int) sequence_length; ++index) {
			const int t = reversed ? (int) sequence_length - 1 - index : index;

			struct ggml_tensor * gates_x_t = ggml_view_3d(ctx, gates_x, gates_x->ne[0], 1, gates_x->ne[2],
			                                             gates_x->nb[0], gates_x->nb[1], gates_x->nb[1] * t);
			// gates_h: [4*hidden, 1, batch]（不再单独加 b_h，bias 已合并进 gates_x）
			struct ggml_tensor * gates_h = ggml_mul_mat(ctx, fused.w_h, h_0);
			struct ggml_tensor * gates   = ggml_add(ctx, gates_x_t, gates_h);

			const size_t gate_off = (size_t) hidden * gates->nb[0];
			struct ggml_tensor * I_cur = ggml_sigmoid(ctx, vk_cont_view(ggml_view_3d(ctx, gates, hidden, 1, gates->ne[2], gates->nb[0], gates->nb[1], gate_off * 0)));
			struct ggml_tensor * F_cur = ggml_sigmoid(ctx, vk_cont_view(ggml_view_3d(ctx, gates, hidden, 1, gates->ne[2], gates->nb[0], gates->nb[1], gate_off * 1)));
			struct ggml_tensor * G_cur = ggml_tanh   (ctx, vk_cont_view(ggml_view_3d(ctx, gates, hidden, 1, gates->ne[2], gates->nb[0], gates->nb[1], gate_off * 2)));
			struct ggml_tensor * O_cur = ggml_sigmoid(ctx, vk_cont_view(ggml_view_3d(ctx, gates, hidden, 1, gates->ne[2], gates->nb[0], gates->nb[1], gate_off * 3)));

			c_0 = ggml_add(ctx, ggml_mul(ctx, F_cur, c_0), ggml_mul(ctx, I_cur, G_cur));
			h_0 = ggml_mul(ctx, ggml_tanh(ctx, c_0), O_cur);

			if (use_set_output) {
				// 说明：写入 outputs[:, t]，保证即便 reversed 扫描也能得到“时间正序”的输出。
				struct ggml_tensor * h_cont = tts_cont_if_needed(ctx, h_0);
				const size_t off = (size_t) outputs->nb[1] * (size_t) t;
				outputs = ggml_set_2d_inplace(ctx, outputs, h_cont, outputs->nb[1], off);
			} else {
				// Vulkan / 兼容回退：保持旧的 concat 实现。
				if (index == 0) {
					outputs = h_0;
				} else {
					outputs = reversed ? ggml_concat(ctx, h_0, outputs, 1) : ggml_concat(ctx, outputs, h_0, 1);
				}
			}

			ggml_build_forward_expand(gf, outputs);
		}

		return outputs;
	}

	// --------------------------- 未融合：保留原始实现（并在 CPU 下可选使用 set 输出） ---------------------------
	struct ggml_tensor * I = ggml_add(ctx, ggml_mul_mat(ctx, weights[0], input), biases[0]);
	struct ggml_tensor * F = ggml_add(ctx, ggml_mul_mat(ctx, weights[2], input), biases[2]);
	struct ggml_tensor * G = ggml_add(ctx, ggml_mul_mat(ctx, weights[4], input), biases[4]);
	struct ggml_tensor * O = ggml_add(ctx, ggml_mul_mat(ctx, weights[6], input), biases[6]);

	for (int index = 0; index < (int) sequence_length; index++) {
		const int t = reversed ? (int) sequence_length - 1 - index : index;

		struct ggml_tensor * I_cur = ggml_view_3d(ctx, I, I->ne[0], 1, I->ne[2], I->nb[0], I->nb[1], I->nb[1] * t);
        {
            struct ggml_tensor * I_sum = ggml_add(ctx, I_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[1], h_0), biases[1]));
            I_sum = vk_cont_view(I_sum);
            I_cur = ggml_sigmoid(ctx, I_sum);
        }

		struct ggml_tensor * F_cur = ggml_view_3d(ctx, F, F->ne[0], 1, F->ne[2], F->nb[0], F->nb[1], F->nb[1] * t);
        {
            struct ggml_tensor * F_sum = ggml_add(ctx, F_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[3], h_0), biases[3]));
            F_sum = vk_cont_view(F_sum);
            F_cur = ggml_sigmoid(ctx, F_sum);
        }

		struct ggml_tensor * G_cur = ggml_view_3d(ctx, G, G->ne[0], 1, G->ne[2], G->nb[0], G->nb[1], G->nb[1] * t);
        {
            struct ggml_tensor * G_sum = ggml_add(ctx, G_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[5], h_0), biases[5]));
            G_sum = vk_cont_view(G_sum);
            G_cur = ggml_tanh(ctx, G_sum);
        }

		struct ggml_tensor * O_cur = ggml_view_3d(ctx, O, O->ne[0], 1, O->ne[2], O->nb[0], O->nb[1], O->nb[1] * t);
        {
            struct ggml_tensor * O_sum = ggml_add(ctx, O_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[7], h_0), biases[7]));
            O_sum = vk_cont_view(O_sum);
            O_cur = ggml_sigmoid(ctx, O_sum);
        }

		c_0 = ggml_add(ctx, ggml_mul(ctx, F_cur, c_0), ggml_mul(ctx, I_cur, G_cur));
		h_0 = ggml_mul(ctx, ggml_tanh(ctx, c_0), O_cur);

		if (use_set_output) {
			struct ggml_tensor * h_cont = tts_cont_if_needed(ctx, h_0);
			const size_t off = (size_t) outputs->nb[1] * (size_t) t;
			outputs = ggml_set_2d_inplace(ctx, outputs, h_cont, outputs->nb[1], off);
		} else {
			if (index == 0) {
				outputs = h_0;
			} else {
				outputs = reversed ? ggml_concat(ctx, h_0, outputs, 1) : ggml_concat(ctx, outputs, h_0, 1);
			}
		}

		ggml_build_forward_expand(gf, outputs);
	}
	return outputs;
}

static struct ggml_tensor * build_ada_residual_conv(ggml_context * ctx, struct ggml_tensor * x, ada_residual_conv_block * block, struct ggml_tensor * style, struct ggml_tensor * sqrt_tensor) {
	struct ggml_tensor * cur = x;
	struct ggml_tensor * gamma;
	struct ggml_tensor * beta;

	gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->norm1_gamma, style), block->norm1_gamma_bias);
	beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->norm1_beta, style), block->norm1_beta_bias);
	cur   = ggml_norm(ctx, x, 0.00001);

	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
	// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
	// 说明：ggml 0.9.4 的二元算子在“src1 非 contiguous + broadcast”时会触发断言（CPU 路径暂未实现该组合）。
	// gamma/beta 这里通过 transpose 来对齐维度，因此需要再做一次 ggml_cont 以确保 src1 contiguous。
	cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, gamma))));
	cur = ggml_add(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, beta)));
	cur = ggml_leaky_relu(ctx, cur, 0.2f, false);

	if (block->pool) {
		// 说明：这里的 pool 是 depthwise 转置卷积（groups=channels）并带 output_padding，
		// ggml 0.9.4 的内置 conv_transpose_1d 不再支持这些扩展能力。
		// 兼容策略：项目侧提供 tts_conv_transpose_1d。
		// - 若 pool 已被展开为全通道权重（K, C, C），则直接走 groups==1 的 Vulkan/CPU 兼容路径；
		// - 否则仍走 groups>1 的 CPU 自定义实现。
		const bool pool_expanded = block->pool->ne[1] == block->pool->ne[2];
		const int pool_groups = pool_expanded ? 1 : (int) cur->ne[1];
		ggml_tensor * pool_input = pool_expanded ? tts_cont_if_needed(ctx, cur) : cur;
		cur = tts_conv_transpose_1d(ctx, block->pool, pool_input,
		                            /*stride=*/2, /*padding=*/1, /*dilation=*/1,
		                            /*output_padding=*/1, /*groups=*/pool_groups);
		cur = ggml_add(ctx, cur, block->pool_bias);
	}

	cur = tts_conv_1d(ctx, block->conv1, cur, 1, 1, 1);

	cur   = ggml_add(ctx, cur, block->conv1_bias);
	gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->norm2_gamma, style), block->norm2_gamma_bias);
	beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->norm2_beta, style), block->norm2_beta_bias);
	cur   = ggml_norm(ctx, cur, 0.00001);

	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
	// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
	// 同上：确保 src1 contiguous，避免 ggml-cpu 对“非连续 broadcast”断言失败。
	cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, gamma))));
	cur = ggml_add(ctx, cur, ggml_cont(ctx, ggml_transpose(ctx, beta)));
	cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
	cur = ggml_add(ctx, tts_conv_1d(ctx, block->conv2, cur, 1, 1, 1), block->conv2_bias);

	struct ggml_tensor * res = cur;
	cur = x;
	if (block->upsample) {
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
		if (block->pool) {
			// 说明：旧版 ggml 的 ggml_upscale_ext 已弃用/变更签名，这里用 ggml_interpolate 取代。
			// 该处只是为了对齐维度（最近邻上采样即可）。
			cur = ggml_interpolate(ctx, cur,
			                       /*ne0=*/cur->ne[0], /*ne1=*/cur->ne[1] * 2, /*ne2=*/cur->ne[2], /*ne3=*/cur->ne[3],
			                       (uint32_t) GGML_SCALE_MODE_NEAREST);
		}
		cur = ggml_mul_mat(ctx, block->upsample, cur);
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
	}
	cur = ggml_div(ctx, ggml_add(ctx, res, cur), sqrt_tensor);
	return cur;
}

static struct ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * style,
                                                             kokoro_generator_residual_block * block,
                                                             std::vector<tts_graph_const_input> * const_inputs) {
    // 说明：Kokoro 生成器的卷积输入以“时间优先（T, C）”布局为主，
    // 这里把 AdaIN 的 gamma/beta 变换到 [1, C, batch]，用于沿时间维广播，
    // 以减少频繁的 transpose + cont。
    auto broadcast_channel_param = [](ggml_context * ctx, ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * src = ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
        // t: [C, batch] => [1, C, batch]
        return ggml_reshape_3d(ctx, src, 1, src->ne[0], src->ne[1]);
    };

	struct ggml_tensor * cur;
	struct ggml_tensor * gamma;
	struct ggml_tensor * beta;
	struct ggml_tensor * inpl = x;
	for (int i = 0; i < block->convs1_weights.size(); i++) {
		gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_gamma_weights[i], style), block->adain1d_1_gamma_biases[i]);
		beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_beta_weights[i], style), block->adain1d_1_beta_biases[i]);
		cur   = ggml_norm(ctx, inpl, 0.00001);

		// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
		cur   = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, broadcast_channel_param(ctx, gamma))), broadcast_channel_param(ctx, beta));
		cur   = snake_1d(ctx, block->input_alphas[i], cur, const_inputs);

		cur   = ggml_add(ctx, tts_conv_1d(ctx, block->convs1_weights[i], cur, 1, block->conv1_paddings[i], block->conv1_dilations[i]), block->convs1_biases[i]);
		gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_gamma_weights[i], style), block->adain1d_2_gamma_biases[i]);
		beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_beta_weights[i], style), block->adain1d_2_beta_biases[i]);
		cur   = ggml_norm(ctx, cur, 0.00001);

		// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
		cur   = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, broadcast_channel_param(ctx, gamma))), broadcast_channel_param(ctx, beta));

		cur   = snake_1d(ctx, block->output_alphas[i], cur, const_inputs);
		cur   = ggml_add(ctx, tts_conv_1d(ctx, block->convs2_weights[i], cur, 1, block->conv1_paddings[0], 1), block->convs2_biases[i]);
		inpl   = ggml_add(ctx, inpl, cur);
	}
	return inpl;
}

static struct ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, struct ggml_tensor * x,
                                              struct ggml_tensor * style, std::vector<tts_graph_const_input> * const_inputs) {
	// This conv_1d seems replaceable with squeezed and transposed ggml_mul_mut, but s0 and p0 are dynamic
	ggml_tensor * cur = ggml_add(ctx, tts_conv_1d(ctx, block->input_conv, x, block->input_conv_stride, block->input_conv_padding, 1), block->input_conv_bias);
	return build_kokoro_generator_res_block(ctx, cur, style, block->res_block, const_inputs);
}

static struct ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, int harmonic_num, int sequence_length, float voice_threshold, float sin_amp, float noise_std) {
    (void) voice_threshold;
	struct ggml_tensor * cur = ggml_mul(ctx, ggml_repeat(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], harmonic_num)), model->harmonic_sampling_norm);
	// 说明：旧版 ggml 有 ggml_mod(x, 1.0)；ggml 0.9.4 移除了该算子。
	// 这里使用 x - floor(x) 来实现“对 1 取模”（即取小数部分），满足 Kokoro 的相位累加需求。
	cur = ggml_sub(ctx, cur, ggml_floor(ctx, cur));
	cur = ggml_mul(ctx, ggml_cumsum(ctx, cur), model->sampling_factor_scalar);

	// 说明：旧版 ggml 的 ggml_upscale_linear / ggml_upscale_ext 已弃用/变更签名，
	// 统一替换为 ggml_interpolate（线性插值时用 BILINEAR 模式即可）。
	cur = ggml_interpolate(ctx, cur,
	                       /*ne0=*/cur->ne[0] * 300, /*ne1=*/cur->ne[1], /*ne2=*/cur->ne[2], /*ne3=*/cur->ne[3],
	                       (uint32_t) GGML_SCALE_MODE_BILINEAR);
	struct ggml_tensor * upscaled = ggml_interpolate(ctx, x,
	                                                 /*ne0=*/x->ne[0] * 300, /*ne1=*/x->ne[1], /*ne2=*/x->ne[2], /*ne3=*/x->ne[3],
	                                                 (uint32_t) GGML_SCALE_MODE_BILINEAR);

	kctx->uv_noise_data = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sequence_length*harmonic_num+4);
	ggml_set_input(kctx->uv_noise_data);

    // 说明：原实现通过自定义 map 生成 uv/noise；为让 Vulkan 参与计算，这里改为标准算子组合。
    // uv_noise_data 前 4 个元素仍保留参数位（voice_threshold / noise_std / sin_amp / sin_amp_div），
    // 后续元素为随机噪声；目前仅使用 voice_threshold（其余用函数参数常量，减少广播开销）。
    const size_t noise_offset = 4 * kctx->uv_noise_data->nb[0];
    struct ggml_tensor * rand_flat = ggml_view_1d(ctx, kctx->uv_noise_data, (int64_t) sequence_length * harmonic_num, noise_offset);
    struct ggml_tensor * rand = ggml_reshape_2d(ctx, rand_flat, sequence_length, harmonic_num);
    rand = tts_cont_if_needed(ctx, rand);

    struct ggml_tensor * upscaled_view = upscaled;
    if (upscaled->ne[1] != 1 || upscaled->ne[2] != 1 || upscaled->ne[3] != 1) {
        // 说明：历史自定义算子忽略 batch 维；此处保持一致，仅使用第一批次。
        upscaled_view = ggml_view_2d(ctx, upscaled, upscaled->ne[0], 1, upscaled->nb[1], 0);
    }
    upscaled_view = tts_cont_if_needed(ctx, upscaled_view);

    struct ggml_tensor * voice_threshold_t = ggml_view_1d(ctx, kctx->uv_noise_data, 1, 0);
    struct ggml_tensor * voice_threshold_b = ggml_repeat(ctx, voice_threshold_t, upscaled_view);
    struct ggml_tensor * mask_1d = ggml_step(ctx, ggml_sub(ctx, upscaled_view, voice_threshold_b));
    struct ggml_tensor * mask = ggml_repeat(ctx, mask_1d, rand);
    mask = tts_cont_if_needed(ctx, mask);

    const float sin_amp_div = sin_amp / 3.0f;
    struct ggml_tensor * rand_mask = ggml_mul(ctx, rand, mask);
    rand_mask = tts_cont_if_needed(ctx, rand_mask);

    struct ggml_tensor * noise = ggml_add(ctx,
                                          ggml_scale(ctx, rand_mask, noise_std - sin_amp_div),
                                          ggml_scale(ctx, rand, sin_amp_div));
    struct ggml_tensor * uv = ggml_scale(ctx, mask, sin_amp);

	return ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_mul(ctx, ggml_sin(ctx, cur), uv), noise)));
}

static struct ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, struct ggml_tensor * style, struct ggml_tensor * f0_curve, kokoro_generator* generator, int sequence_length, struct ggml_tensor * window_sq_sum, ggml_cgraph * gf) {
	struct ggml_tensor * sing = build_sin_gen(ctx, model, kctx, f0_curve, model->harmonic_num + 1, f0_curve->ne[0] * 300, model->voice_threshold, model->sin_amp, model->noise_std);
	struct ggml_tensor * har = ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, generator->m_source_weight, sing), generator->m_source_bias));

    // 说明：
    // - Vulkan 路径默认启用 conv 版 STFT/ISTFT（完整图执行更快）。
    //   如遇明显音质劣化（如“金属音”），可回退到自定义 STFT/ISTFT：
    //   - TTS_VK_GRAPH_STFT=0：har 的 STFT 回退到自定义算子（CPU）
    //   - TTS_VK_GRAPH_ISTFT=0：末端 ISTFT 回退到自定义算子（CPU）
    // - CPU 路径也支持按需启用 conv 版 STFT/ISTFT（默认关闭，便于保留自定义算子性能）：
    //   - TTS_CPU_GRAPH_STFT=1：CPU 启用 conv 版 STFT
    //   - TTS_CPU_GRAPH_ISTFT=1：CPU 启用 conv 版 ISTFT
    const bool vk_backend = tts_backend_is_vulkan(kctx->backend);
    const bool has_stft_basis = model->stft_forward_basis && model->stft_inverse_basis;
    const bool cpu_graph_stft = !vk_backend && has_stft_basis && tts_env_truthy_default("TTS_CPU_GRAPH_STFT", /*default_value=*/false);
    const bool cpu_graph_istft = !vk_backend && has_stft_basis && tts_env_truthy_default("TTS_CPU_GRAPH_ISTFT", /*default_value=*/false);
    const bool use_graph_stft = vk_backend
                                    ? (has_stft_basis && tts_env_truthy_default("TTS_VK_GRAPH_STFT", /*default_value=*/true))
                                    : cpu_graph_stft;
    const bool use_graph_istft = vk_backend
                                     ? (has_stft_basis && tts_env_truthy_default("TTS_VK_GRAPH_ISTFT", /*default_value=*/true))
                                     : cpu_graph_istft;
    const bool use_graph_consts = use_graph_stft && vk_backend; // 常量输入仅用于 Vulkan 的 stft_graph
    std::vector<tts_graph_const_input> * const_inputs = use_graph_consts ? &kctx->graph_const_inputs : nullptr;

    // 说明：质量兜底/排查开关：可选择仅让 STFT/ISTFT 在 CPU 上执行（避免某些 Vulkan 实现导致的音质问题）。
    // 注意：这里并不改变“图结构”（仍使用 conv 版 stft_graph/istft_graph），只是在权重与调度上让其走 CPU 后端。
    const bool vk_quality_mode = vk_backend && tts_env_truthy("TTS_VK_AUDIO_QUALITY");
    const bool stft_cpu = vk_backend && (vk_quality_mode || tts_env_truthy("TTS_VK_STFT_CPU"));
    const bool istft_cpu = vk_backend && (vk_quality_mode || tts_env_truthy("TTS_VK_ISTFT_CPU"));

    // 说明：自定义 STFT/ISTFT（GGML_OP_CUSTOM）在实现中会直接读取 window->data，
    // 因此在 Vulkan 权重位于设备内存时必须改用 CPU 可读副本。
    ggml_tensor * window_custom = generator->window_cpu ? generator->window_cpu : generator->window;

    ggml_tensor * stft_forward_basis = model->stft_forward_basis;
    ggml_tensor * stft_inverse_basis = model->stft_inverse_basis;
    if (stft_cpu && model->stft_forward_basis_cpu) {
        stft_forward_basis = model->stft_forward_basis_cpu;
    }
    if (istft_cpu && model->stft_inverse_basis_cpu) {
        stft_inverse_basis = model->stft_inverse_basis_cpu;
    }
    // 说明：清理上一次图构建的输入指针，避免 set_inputs 误写旧地址。
    kctx->stft_pad_indices = nullptr;
    kctx->graph_const_inputs.clear();

    ggml_tensor * har_in = ggml_cont(ctx, ggml_transpose(ctx, har));
    if (use_graph_stft) {
        // 说明：Vulkan 路径使用 conv 版 STFT，需提前准备反射 padding 索引。
        const int64_t half = (int64_t) (model->true_n_fft / 2);
        const int64_t pad_len = har_in->ne[0] + 2 * half;
        kctx->stft_pad_indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, pad_len, har_in->ne[1]);
        ggml_set_input(kctx->stft_pad_indices);
        ggml_set_name(kctx->stft_pad_indices, "kokoro.stft_pad_indices");
        har = stft_graph(ctx, har_in, stft_forward_basis, kctx->stft_pad_indices, const_inputs,
                         model->true_n_fft, model->stft_hop, true, true);
    } else {
        har = stft(ctx, har_in, window_custom, model->true_n_fft, model->stft_hop, true, true);
    }

	// stft returns a vector of shape [nfft, frames, batch, 2] where the final shape (2) separates the magnitude and the phase
	// kokoro concatenates the n_fft from the magnitude and the phase together so we have to split them up and concatenate
	// along the n_fft axis
	// 说明：view 本身不会产生拷贝；concat 支持非连续输入，因此这里不需要额外 ggml_cont。
	struct ggml_tensor * mhar  = ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], 0);
	struct ggml_tensor * phhar = ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], har->nb[3]);
	struct ggml_tensor * combined_har = ggml_cont(ctx, ggml_transpose(ctx, ggml_concat(ctx, mhar, phhar, 0)));

	struct ggml_tensor * cur = x;
	for (int i = 0; i < generator->ups.size(); i++) {
		cur = ggml_leaky_relu(ctx, cur, 0.1f, false);
		// 说明：ggml 0.9.4 的 conv_transpose_1d 目前限制 padding==0，因此这里走项目侧兼容封装：
		// 先计算 padding=0 的更长输出，再裁剪得到等价结果（groups==1 场景）。
		cur = ggml_add(ctx,
		               tts_conv_transpose_1d(ctx,
		                                     generator->ups[i]->upsample_weight,
		                                     tts_cont_if_needed(ctx, cur),
		                                     (int) generator->ups[i]->stride,
		                                     (int) generator->ups[i]->padding,
		                                     /*dilation=*/1),
		               generator->ups[i]->upsample_bias);
		if (i == generator->ups.size() - 1) {
			// This is a hacky way of implementing the simple reflection padding used here.
			// In general, ggml should eventually be built to support expressive reflective padding but for such simple front padding this makes more sense.
			struct ggml_tensor * temp = ggml_cont(ctx, ggml_view_3d(ctx, cur, 1, cur->ne[1], cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[0]));
			cur = ggml_concat(ctx, temp, cur, 0);
		}
		// 说明：combined_har 已是 cont 输出，避免在循环里重复 ggml_cont 触发大块拷贝。
		struct ggml_tensor * x_source = build_noise_block(ctx, generator->noise_blocks[i], combined_har, style, const_inputs);
		cur = ggml_add(ctx, cur, x_source);
		struct ggml_tensor * x = cur;
		for (int ii = 0; ii < model->n_kernels; ii++) {
			if (ii == 0) {
				cur = build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i*model->n_kernels+ii], const_inputs);
			} else {
				cur = ggml_add(ctx, cur, build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i*model->n_kernels+ii], const_inputs));
			}
		}
		// 说明：保持时间优先布局，减少 transpose/cont 的额外开销。
		cur = ggml_div(ctx, cur, model->n_kernels_tensor);
		ggml_build_forward_expand(gf, cur);
	}

	cur = ggml_leaky_relu(ctx, cur, 0.01f, false);
	cur = ggml_add(ctx, tts_conv_1d(ctx, generator->out_conv_weight, tts_cont_if_needed(ctx, cur), 1, model->out_conv_padding, 1), generator->out_conv_bias);

	struct ggml_tensor * spec = ggml_view_3d(ctx, cur, cur->ne[0], model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], 0);
	struct ggml_tensor * phase = ggml_view_3d(ctx, cur, cur->ne[0], cur->ne[1] - model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[1] * model->post_n_fft);
	phase = ggml_sin(ctx, phase);
	spec = ggml_exp(ctx, spec);

	cur = ggml_concat(ctx, spec, phase, 3); // istft expects the magnitude and phase concatenated after the batch;
    ggml_tensor * istft_in = ggml_cont(ctx, ggml_transpose(ctx, cur));
    if (use_graph_istft) {
        cur = istft_graph(ctx, istft_in, window_sq_sum, stft_inverse_basis,
                          model->true_n_fft, model->stft_hop, true, true);
    } else {
        cur = istft(ctx, istft_in, window_sq_sum, window_custom,
                    model->true_n_fft, model->stft_hop, true, true);
    }
	ggml_set_name(cur, "after_res_gen");
	return cur;
}

static struct kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, std::string base_config_key) {
	struct kokoro_generator_residual_block * grb = new struct kokoro_generator_residual_block;
	// these residual blocks always have 3 convolutional layers
	for (int i = 0; i < 3; i++) {
		grb->adain1d_1_gamma_weights.push_back(nullptr);
		grb->adain1d_2_gamma_weights.push_back(nullptr);
		grb->adain1d_1_gamma_biases.push_back(nullptr);
		grb->adain1d_2_gamma_biases.push_back(nullptr);
		grb->adain1d_1_beta_weights.push_back(nullptr);
		grb->adain1d_2_beta_weights.push_back(nullptr);
		grb->adain1d_1_beta_biases.push_back(nullptr);
		grb->adain1d_2_beta_biases.push_back(nullptr);
		grb->input_alphas.push_back(nullptr);
		grb->output_alphas.push_back(nullptr);
		grb->convs1_weights.push_back(nullptr);
		grb->convs1_biases.push_back(nullptr);
		grb->convs2_weights.push_back(nullptr);
		grb->convs2_biases.push_back(nullptr);
		int padding_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".padding").c_str());
		int dilation_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".dilation").c_str());
		if (padding_key == -1 || dilation_key == -1) {
			TTS_ABORT("Could not find dilation and padding for generator residual block at key, '%s.%d'.", base_config_key.c_str(), i);
		}
		grb->conv1_dilations.push_back(gguf_get_val_u32(meta, dilation_key));
		grb->conv1_paddings.push_back(gguf_get_val_u32(meta, padding_key));
	}
	return grb;
}

static struct kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index) {
	struct kokoro_noise_residual_block * nb = new struct kokoro_noise_residual_block;
	std::string base = "kokoro.decoder.generator.noise_blocks." + std::to_string(index);
	nb->res_block = build_res_block_from_file(meta, base + ".res_block");
	int stride_key = gguf_find_key(meta, (base + ".stride").c_str());
	int padding_key = gguf_find_key(meta, (base + ".padding").c_str());
	if (padding_key == -1 || stride_key == -1) {
		TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro noise block.");
	}
	nb->input_conv_stride = gguf_get_val_u32(meta, stride_key);
	nb->input_conv_padding = gguf_get_val_u32(meta, padding_key);
	return nb;
}

static struct kokoro_generator_upsample_block * kokoro_generator_upsample_block(gguf_context * meta, int index) {
	struct kokoro_generator_upsample_block * usb = new struct kokoro_generator_upsample_block;
	std::string base = "kokoro.decoder.generator.up_convs." + std::to_string(index);
	int stride_key = gguf_find_key(meta, (base + ".stride").c_str());
	int padding_key = gguf_find_key(meta, (base + ".padding").c_str());
	if (padding_key == -1 || stride_key == -1) {
		TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro upsample block.");
	}
	usb->stride = gguf_get_val_u32(meta, stride_key);
	usb->padding = gguf_get_val_u32(meta, padding_key);
	return usb;
}

size_t kokoro_model::max_gen_nodes() {
	return std::max<size_t>(8192, generation_node_counter*2);
}

size_t kokoro_model::max_duration_nodes() {
	return std::max<size_t>(8192, duration_node_counter*2);
}

void kokoro_model::post_load_assign() {
	size_t original_offset = offset;
	n_kernels_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    // 说明：这些“后处理常量张量”会被放进与权重相同的 buffer 中；在 Vulkan 后端下必须满足 storage buffer 的对齐要求。
    alloc_tensor(n_kernels_tensor, "n_kernels_tensor");
	size_t size = ggml_nbytes(n_kernels_tensor);
	float nker = (float) n_kernels;
	ggml_backend_tensor_set(n_kernels_tensor, &nker, 0, size);

	sqrt_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    alloc_tensor(sqrt_tensor, "sqrt_tensor");
    size = ggml_nbytes(sqrt_tensor);
    float sqrt2 = sqrtf(2.0f);
	ggml_backend_tensor_set(sqrt_tensor, &sqrt2, 0, size);

	std::vector<float> data{};
 	for (int l = 0; l < lstms.size(); l++) {
 		lstm * rnn = lstms[l];
 		const int32_t hidden_size = rnn->cells[0]->biases[0]->ne[0];
 		data.resize(hidden_size);
 
 		for (int i = 0; i < rnn->cells.size(); i++) {
 			struct ggml_tensor * h = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
 			struct ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            alloc_tensor(h);
     		size_t size = ggml_nbytes(h);
 			ggml_backend_tensor_set(h, data.data(), 0, size);
 			ggml_format_name(h, "lstm%d_hidden", l);
            alloc_tensor(s);
 			ggml_backend_tensor_set(s, data.data(), 0, size);
 			ggml_format_name(s, "lstm%d_state", l);
     		rnn->hidden.push_back(h);
     		rnn->states.push_back(s);
 		}
  		data.clear();
 	}

    // 说明：CPU LSTM gate 融合权重的预处理（一次性）。
    // - 通过把 I/F/G/O 四个 gate 的权重拼接为一个大矩阵，减少 mul_mat 次数；
    // - 同时把 b_x + b_h 提前合并，减少每步 add。
    // - 仅在 CPU 后端启用（Vulkan 默认关闭）。
    if (!kokoro_use_vk_weights(this) && kokoro_env_cpu_lstm_fuse()) {
        for (int l = 0; l < (int) lstms.size(); ++l) {
            lstm * rnn = lstms[l];
            if (!rnn) {
                continue;
            }
            for (int c = 0; c < (int) rnn->cells.size(); ++c) {
                kokoro_try_fuse_lstm_cell(this, l, c, rnn->cells[c], /*reversed=*/false);
                if (rnn->bidirectional) {
                    kokoro_try_fuse_lstm_cell(this, l, c, rnn->cells[c], /*reversed=*/true);
                }
            }
        }
    }

  	if (window == "hann") {
 		// 说明：保留一份 CPU 侧窗函数缓存，用于输入准备时计算 window_sq_sum。
 		decoder->generator->window_host.clear();
		decoder->generator->window_host.reserve(true_n_fft);
 		hann_window(true_n_fft, decoder->generator->window_host);
 		decoder->generator->window = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, true_n_fft);
        alloc_tensor(decoder->generator->window, "stft_window");
 		size_t size = ggml_nbytes(decoder->generator->window);
 		ggml_backend_tensor_set(decoder->generator->window, decoder->generator->window_host.data(), 0, size);

        // 说明：为自定义 STFT/ISTFT（GGML_OP_CUSTOM）回退路径准备一份 CPU 可读的 window 张量。
        // - 自定义算子在实现中直接读取 win->data，因此当 window 位于 Vulkan 设备内存时会读到不可用地址，
        //   进而表现为“电流声/人声消失/强金属音”等严重失真。
        // - 这里按 buffer 类型判断是否为 host；若不是，则额外分配 CPU buffer 并拷贝一份数据。
        const bool window_host = (buffer != nullptr) ? ggml_backend_buft_is_host(buffer) : true;
        if (window_host) {
            decoder->generator->window_cpu = decoder->generator->window;
        } else {
            decoder->generator->window_cpu = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, true_n_fft);
            ggml_set_name(decoder->generator->window_cpu, "stft_window.cpu");
            ggml_backend_buffer_t win_cpu_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(decoder->generator->window_cpu));
            if (!win_cpu_buf) {
                TTS_ABORT("Failed to allocate CPU buffer for stft_window.cpu (bytes=%zu)\n",
                          ggml_nbytes(decoder->generator->window_cpu));
            }
            extra_buffers.push_back(win_cpu_buf);
            decoder->generator->window_cpu->buffer = win_cpu_buf;
            decoder->generator->window_cpu->data = ggml_backend_buffer_get_base(win_cpu_buf);
            ggml_backend_tensor_set(decoder->generator->window_cpu, decoder->generator->window_host.data(), 0, size);
        }
 	} else {
 		TTS_ABORT("Window of type %s is not supported.", window.c_str());
 	}

    // 说明：生成 STFT/ISTFT 基矩阵（conv 形式），用于 Vulkan 完整图执行。
    // 这些基矩阵是常量权重，放在独立 buffer 中以避免占用主权重空间。
    {
        std::vector<float> forward_basis;
        std::vector<float> inverse_basis;
        kokoro_build_stft_basis(decoder->generator->window_host, true_n_fft, forward_basis, inverse_basis);

        const int64_t cutoff = (int64_t) (true_n_fft / 2 + 1);
        const int64_t rows = cutoff * 2;

        stft_forward_basis = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, true_n_fft, 1, rows);
        ggml_set_name(stft_forward_basis, "kokoro.stft_forward_basis");
        ggml_backend_buffer_t fwd_buf = kokoro_alloc_extra_buffer(this, ggml_nbytes(stft_forward_basis), "kokoro.stft_forward_basis");
        stft_forward_basis->buffer = fwd_buf;
        stft_forward_basis->data = ggml_backend_buffer_get_base(fwd_buf);
        ggml_backend_tensor_set(stft_forward_basis, forward_basis.data(), 0, forward_basis.size() * sizeof(float));

        stft_inverse_basis = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, true_n_fft, 1, rows);
        ggml_set_name(stft_inverse_basis, "kokoro.stft_inverse_basis");
        ggml_backend_buffer_t inv_buf = kokoro_alloc_extra_buffer(this, ggml_nbytes(stft_inverse_basis), "kokoro.stft_inverse_basis");
        stft_inverse_basis->buffer = inv_buf;
        stft_inverse_basis->data = ggml_backend_buffer_get_base(inv_buf);
        ggml_backend_tensor_set(stft_inverse_basis, inverse_basis.data(), 0, inverse_basis.size() * sizeof(float));

        // 说明：为支持 Vulkan 下 STFT/ISTFT 回退 CPU（或混合计算），需要一份 CPU 可读的基矩阵。
        // - 若当前权重 buffer 本身是 host buffer（CPU 或 Vulkan-host-visible），则直接复用即可；
        // - 否则（Vulkan 设备内存）：额外分配一份 CPU buffer 副本，避免 CPU 回退时读到设备内存导致崩溃/异常。
        const bool basis_host = (buffer != nullptr) ? ggml_backend_buft_is_host(buffer) : true;
        if (basis_host) {
            stft_forward_basis_cpu = stft_forward_basis;
            stft_inverse_basis_cpu = stft_inverse_basis;
        } else {
            stft_forward_basis_cpu = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, true_n_fft, 1, rows);
            ggml_set_name(stft_forward_basis_cpu, "kokoro.stft_forward_basis.cpu");
            ggml_backend_buffer_t fwd_cpu_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(stft_forward_basis_cpu));
            if (!fwd_cpu_buf) {
                TTS_ABORT("Failed to allocate CPU buffer for kokoro.stft_forward_basis.cpu (bytes=%zu)\n",
                          ggml_nbytes(stft_forward_basis_cpu));
            }
            extra_buffers.push_back(fwd_cpu_buf);
            stft_forward_basis_cpu->buffer = fwd_cpu_buf;
            stft_forward_basis_cpu->data = ggml_backend_buffer_get_base(fwd_cpu_buf);
            ggml_backend_tensor_set(stft_forward_basis_cpu, forward_basis.data(), 0, forward_basis.size() * sizeof(float));

            stft_inverse_basis_cpu = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, true_n_fft, 1, rows);
            ggml_set_name(stft_inverse_basis_cpu, "kokoro.stft_inverse_basis.cpu");
            ggml_backend_buffer_t inv_cpu_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(stft_inverse_basis_cpu));
            if (!inv_cpu_buf) {
                TTS_ABORT("Failed to allocate CPU buffer for kokoro.stft_inverse_basis.cpu (bytes=%zu)\n",
                          ggml_nbytes(stft_inverse_basis_cpu));
            }
            extra_buffers.push_back(inv_cpu_buf);
            stft_inverse_basis_cpu->buffer = inv_cpu_buf;
            stft_inverse_basis_cpu->data = ggml_backend_buffer_get_base(inv_cpu_buf);
            ggml_backend_tensor_set(stft_inverse_basis_cpu, inverse_basis.data(), 0, inverse_basis.size() * sizeof(float));
        }
    }

 	harmonic_sampling_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, harmonic_num + 1);
    alloc_tensor(harmonic_sampling_norm, "harmonic_sampling_norm");
 	std::vector<float> hdata;
 	hdata.reserve(harmonic_num + 1);
 	for (int i = 0; i < harmonic_num + 1; i++) {
 		hdata.push_back(((float)i + 1.0f) / sample_rate);
 	}
 	size_t hsize = ggml_nbytes(harmonic_sampling_norm);
 	ggml_backend_tensor_set(harmonic_sampling_norm, hdata.data(), 0, hsize);
 	hdata.clear();

 	sampling_factor_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    alloc_tensor(sampling_factor_scalar, "sampling_factor_scalar");
     size_t scsize = ggml_nbytes(sampling_factor_scalar);
     // while it might appear that the upsampling_rate could be used here, the interpolation rate (i.e. the upsampling scale) is actually independent in the kokoro model implementation.
     float sample_scalar = upsample_scale*2.0f*std::numbers::pi;
 	ggml_backend_tensor_set(sampling_factor_scalar, &sample_scalar, 0, scsize);
 	post_load_tensor_bytes = 300 + offset - original_offset;
}

void kokoro_model::assign_lstm(lstm * rnn, std::string name, ggml_tensor * tensor) {
	std::vector<std::string> parts = split(name, ".");
	int i = std::stoi(parts[0]);
	int ii = std::stoi(parts[2]);
	if (parts[1] == "weights") {
		rnn->cells[i]->weights[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->weights[ii], tensor);
	} else if (parts[1] == "biases") {
		rnn->cells[i]->biases[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->biases[ii], tensor);
	} else if (parts[1] == "reverse_weights") {
		rnn->cells[i]->reverse_weights[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->reverse_weights[ii], tensor);
	} else if (parts[1] == "reverse_biases") {
		rnn->cells[i]->reverse_biases[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->reverse_biases[ii], tensor);
	}
}

void kokoro_model::assign_weight(const char * name, ggml_tensor & tensor) {
    if (const string_view name_sv{ name }; tts_starts_with(name_sv, "albert.")) {
        assign_albert_weight(string{ name_sv.substr(sizeof("albert.") - 1) }, &tensor);
    } else if (tts_starts_with(name_sv, "duration_predictor.")) {
        assign_duration_weight(string{ name_sv.substr(sizeof("duration_predictor.") - 1) }, &tensor);
    } else if (tts_starts_with(name_sv, "text_encoder.")) {
        assign_text_encoder_weight(string{ name_sv.substr(sizeof("text_encoder.") - 1) }, &tensor);
    } else if (tts_starts_with(name_sv, "decoder.")) {
        assign_decoder_weight(string{ name_sv.substr(sizeof("decoder.") - 1) }, &tensor);
    } else if (tts_starts_with(name_sv, "voice_tensors.")) {
        const string voice{ name_sv.substr(sizeof("voice_tensors.") - 1) };
        voices[voice] = ggml_dup_tensor(ctx, &tensor);
        set_tensor(voices[voice], &tensor);
    }
}

void kokoro_model::assign_generator_weight(kokoro_generator * generator, std::string name, ggml_tensor * tensor) {
	if (name == "m_source_weight") {
		generator->m_source_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->m_source_weight, tensor);
	} else if (name == "m_source_bias") {
		generator->m_source_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->m_source_bias, tensor);
	} else if (name == "conv_post_weight") {
		generator->out_conv_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->out_conv_weight, tensor);
	} else if (name == "conv_post_bias") {
		generator->out_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(generator->out_conv_bias, tensor);
	} else {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		if (parts[0] == "noise_blocks") {
			if (parts[2] == "conv_weight") {
				generator->noise_blocks[i]->input_conv = ggml_dup_tensor(ctx, tensor);
				set_tensor(generator->noise_blocks[i]->input_conv, tensor);
			} else if (parts[2] == "conv_bias") {
				generator->noise_blocks[i]->input_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
				set_tensor(generator->noise_blocks[i]->input_conv_bias, tensor);
			} else if (parts[2] == "resblock") {
				assign_gen_resblock(generator->noise_blocks[i]->res_block, name.substr(parts[0].size()+parts[1].size()+parts[2].size()+3), tensor);
			}
		} else if (parts[0] == "resblocks") {
			assign_gen_resblock(generator->res_blocks[i], name.substr(parts[0].size()+parts[1].size()+2), tensor);
		} else if (parts[0] == "ups") {
			if (parts[2] == "weight") {
                if (kokoro_use_vk_weights(this)) {
                    ggml_tensor * weight_f32 = kokoro_make_f32_weight(this, tensor, "kokoro.upsample_weight_vk");
                    if (weight_f32) {
                        generator->ups[i]->upsample_weight = weight_f32;
                        return;
                    }
                }
				generator->ups[i]->upsample_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(generator->ups[i]->upsample_weight, tensor);
			} else if (parts[2] == "bias") {
				generator->ups[i]->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
				set_tensor(generator->ups[i]->upsample_bias, tensor);
			}
		}
	}
}

void kokoro_model::assign_gen_resblock(kokoro_generator_residual_block * block, std::string name, ggml_tensor * tensor) {
	std::vector<std::string> parts = split(name, ".");
	int i = std::stoi(parts[0]);
	if (parts[1] == "gamma1_weight") {
		block->adain1d_1_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_gamma_weights[i], tensor);
	} else if (parts[1] == "gamma2_weight") {
		block->adain1d_2_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_gamma_weights[i], tensor);
	} else if (parts[1] == "gamma1_bias") {
		block->adain1d_1_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_gamma_biases[i], tensor);
	} else if (parts[1] == "gamma2_bias") {
		block->adain1d_2_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_gamma_biases[i], tensor);
	} else if (parts[1] == "beta1_weight") {
		block->adain1d_1_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_beta_weights[i], tensor);
	} else if (parts[1] == "beta2_weight") {
		block->adain1d_2_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_beta_weights[i], tensor);
	} else if (parts[1] == "beta1_bias") {
		block->adain1d_1_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_beta_biases[i], tensor);
	} else if (parts[1] == "beta2_bias") {
		block->adain1d_2_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_beta_biases[i], tensor);
	} else if (parts[1] == "convs1_weight") {
		block->convs1_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->convs1_weights[i], tensor);
	} else if (parts[1] == "convs2_weight") {
		block->convs2_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->convs2_weights[i], tensor);
	} else if (parts[1] == "convs1_bias") {
		block->convs1_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->convs1_biases[i], tensor);
	} else if (parts[1] == "convs2_bias") {
		block->convs2_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->convs2_biases[i], tensor);
	} else if (parts[1] == "alpha1") {
		block->input_alphas[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->input_alphas[i], tensor);
	} else if (parts[1] == "alpha2") {
		block->output_alphas[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->output_alphas[i], tensor);
	}
}

/**
 * Removes the last axis, for cases where it's redundantly of length 1.
 * assert x.ndim == 3; numpy.squeeze(x, axis=-1)
 */
static ggml_tensor * squeeze_3d_2d_e0(ggml_context * ctx, ggml_tensor * x) {
	TTS_ASSERT(x->ne[0] == 1);
	TTS_ASSERT(ggml_is_contiguous(x));
	return ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);
}

void kokoro_model::assign_ada_res_block(ada_residual_conv_block * block, std::string name, ggml_tensor * tensor) {
	if (name == "norm1_gamma_weight") {
		block->norm1_gamma = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_gamma, tensor);
	} else if (name == "norm2_gamma_weight") {
		block->norm2_gamma = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_gamma, tensor);
	} else if (name == "norm1_gamma_bias") {
		block->norm1_gamma_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_gamma_bias, tensor);
	} else if (name == "norm2_gamma_bias") {
		block->norm2_gamma_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_gamma_bias, tensor);
	} else if (name == "norm1_beta_weight") {
		block->norm1_beta = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_beta, tensor);
	} else if (name == "norm2_beta_weight") {
		block->norm2_beta = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_beta, tensor);
	} else if (name == "norm1_beta_bias") {
		block->norm1_beta_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_beta_bias, tensor);
	} else if (name == "norm2_beta_bias") {
		block->norm2_beta_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_beta_bias, tensor);
	} else if (name == "conv1_weight") {
		block->conv1 = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->conv1, tensor);
	} else if (name == "conv2_weight") {
		block->conv2 = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->conv2, tensor);
	} else if (name == "conv1_bias") {
		block->conv1_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->conv1_bias, tensor);
	} else if (name == "conv2_bias") {
		block->conv2_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->conv2_bias, tensor);
	} else if (name == "pool_weight") {
        // 说明：pool 是 depthwise 转置卷积（groups=channels）。
        // - Vulkan：必须展开，否则无法在 Vulkan 上执行；
        // - CPU：展开后可走 ggml 的 groups==1 路径，通常比自定义 groups>1 CPU kernel 更快。
        const char * expanded_name = kokoro_use_vk_weights(this) ? "kokoro.pool_weight_vk" : "kokoro.pool_weight_cpu";
        ggml_tensor * expanded = kokoro_expand_depthwise_pool_weight(this, tensor, expanded_name);
        if (expanded) {
            block->pool = expanded;
            return;
        }
		block->pool = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->pool, tensor);
	} else if (name == "pool_bias") {
		block->pool_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->pool_bias, tensor);
	} else if (name == "conv1x1_weight") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		block->upsample = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->upsample, tensor);
	} else if (name == "conv1x1_bias") {
		block->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->upsample_bias, tensor);
	}
}

void kokoro_model::assign_decoder_weight(std::string name, ggml_tensor * tensor) {
	if (name == "f0_conv_weight") {
		decoder->f0_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->f0_conv, tensor);
	} else if (name == "f0_conv_bias") {
		decoder->f0_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->f0_conv_bias, tensor);
	} else if (name == "n_conv_weight") {
		decoder->n_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->n_conv, tensor);
	} else if (name == "n_conv_bias") {
		decoder->n_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->n_conv_bias, tensor);
	} else if (name == "asr_conv_weight") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		decoder->asr_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->asr_conv, tensor);
	} else if (name == "asr_conv_bias") {
		decoder->asr_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->asr_conv_bias, tensor);
	} else if (has_prefix(name, "decoder_blocks")) {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		assign_ada_res_block(decoder->decoder_blocks[i], parts[2], tensor);
	} else if (has_prefix(name, "encoder_block")) {
		std::vector<std::string> parts = split(name, ".");
		assign_ada_res_block(decoder->encoder_block, parts[1], tensor);
	} else if (has_prefix(name, "generator")) {
		assign_generator_weight(decoder->generator, name.substr(10), tensor);
	}
}

void kokoro_model::assign_duration_weight(std::string name, ggml_tensor * tensor) {
	if (name == "encode") {
		prosody_pred->albert_encode = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->albert_encode , tensor);
	} else if (name == "encode_bias") {
		prosody_pred->albert_encode_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->albert_encode_bias, tensor);
	} else if (name == "duration_proj") {
		prosody_pred->duration_proj = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->duration_proj, tensor);
	} else if (name == "duration_proj_bias") {
		prosody_pred->duration_proj_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->duration_proj_bias, tensor);
	} else if (name == "n_proj_kernel") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		prosody_pred->n_proj_kernel = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->n_proj_kernel, tensor);
	} else if (name == "n_proj_bias") {
		prosody_pred->n_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(prosody_pred->n_proj_bias, tensor);
	} else if (name == "f0_proj_kernel") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		prosody_pred->f0_proj_kernel = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->f0_proj_kernel, tensor);
	} else if (name == "f0_proj_bias") {
		prosody_pred->f0_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(prosody_pred->f0_proj_bias, tensor);
	} else {
		std::vector<std::string> parts = split(name, ".");
		if (parts[0] == "shared_lstm") {
			assign_lstm(prosody_pred->shared_lstm, name.substr(parts[0].size()+1), tensor);
		} else if (parts[0] == "duration_lstm") {
			assign_lstm(prosody_pred->duration_proj_lstm, name.substr(parts[0].size()+1), tensor);
		} else if (parts[0] == "f0_blocks") {
			int i = std::stoi(parts[1]);
			assign_ada_res_block(prosody_pred->f0_blocks[i], parts[2], tensor);
		} else if (parts[0] == "n_blocks") {
			int i = std::stoi(parts[1]);
			assign_ada_res_block(prosody_pred->n_blocks[i], parts[2], tensor);
		} else if (parts[0] == "layers") {
			int i = std::stoi(parts[1]);
			i = i / 2;
			if (parts[2] == "gamma_weight") {
				prosody_pred->layers[i]->ada_norm_gamma_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_gamma_weight , tensor);
			} else if (parts[2] == "gamma_bias") {
				prosody_pred->layers[i]->ada_norm_gamma_bias = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_gamma_bias , tensor);
			} else if (parts[2] == "beta_weight") {
				prosody_pred->layers[i]->ada_norm_beta_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_beta_weight , tensor);
			} else if (parts[2] == "beta_bias") {
				prosody_pred->layers[i]->ada_norm_beta_bias = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_beta_bias , tensor);
			} else if (parts[2] == "lstm") {
				assign_lstm(prosody_pred->layers[i]->rnn, name.substr(parts[0].size()+parts[1].size()+parts[2].size()+3), tensor);
			}
		}
	}
}

void kokoro_model::assign_text_encoder_weight(std::string name, ggml_tensor * tensor) {
	if (name == "embedding_weight") {
		text_encoder->embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(text_encoder->embd, tensor);
	} else if (has_prefix(name, "lstm")) {
		assign_lstm(text_encoder->out_lstm, name.substr(5), tensor);
	} else if (has_prefix(name, "layers")) {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		if (parts[2] == "gamma") {
			text_encoder->conv_layers[i]->norm_gamma = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->norm_gamma, tensor);
		} else if (parts[2] == "beta") {
			text_encoder->conv_layers[i]->norm_beta = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->norm_beta, tensor);
		} else if (parts[2] == "weight") {
			text_encoder->conv_layers[i]->conv_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->conv_weight, tensor);
		} else if (parts[2] == "bias") {
			text_encoder->conv_layers[i]->conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
			set_tensor(text_encoder->conv_layers[i]->conv_bias, tensor);
		}
	}
}

void kokoro_model::assign_albert_weight(std::string name, ggml_tensor * tensor) {
	if (name == "embd") {
		embd_hidden = ggml_dup_tensor(ctx, tensor);
		set_tensor(embd_hidden, tensor);
	} else if (name == "embd_bias") {
		embd_hidden_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(embd_hidden_bias, tensor);
	} else if (name == "token_embd") {
		token_embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(token_embd, tensor);
	} else if (name == "position_embd") {
		position_embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(position_embd, tensor);
	} else if (name == "norm") {
		input_norm_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(input_norm_weight, tensor);
	} else if (name == "norm_bias") {
		input_norm_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(input_norm_bias, tensor);
	} else if (name == "token_type_embd") {
		static_token_type_values = ggml_dup_tensor(ctx, tensor);
		set_tensor(static_token_type_values, tensor);
	} else if (has_prefix(name, "layer")) {
		std::vector<std::string> parts = split(name, '.');
		int i = std::stoi(parts[1]);
		if (parts[2] == "ffn") {
			layers[i]->ffn = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn, tensor);
		} else if (parts[2] == "ffn_bias") {
			layers[i]->ffn_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_bias, tensor);
		} else if (parts[2] == "ffn_out") {
			layers[i]->ffn_out = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_out, tensor);
		} else if (parts[2] == "ffn_out_bias") {
			layers[i]->ffn_out_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_out_bias, tensor);
		} else if (parts[2] == "attn_norm") {
			layers[i]->layer_output_norm_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->layer_output_norm_weight, tensor);
		} else if (parts[2] == "attn_norm_bias") {
			layers[i]->layer_output_norm_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->layer_output_norm_bias, tensor);
		} else if (parts[2] == "q") {
			layers[i]->q = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->q, tensor);
		} else if (parts[2] == "k") {
			layers[i]->k = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->k, tensor);
		} else if (parts[2] == "v") {
			layers[i]->v = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->v, tensor);
		} else if (parts[2] == "o") {
			layers[i]->o = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->o, tensor);
		}  else if (parts[2] == "q_bias") {
			layers[i]->q_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->q_bias, tensor);
		} else if (parts[2] == "k_bias") {
			layers[i]->k_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->k_bias, tensor);
		} else if (parts[2] == "v_bias") {
			layers[i]->v_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->v_bias, tensor);
		} else if (parts[2] == "o_bias") {
			layers[i]->o_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->o_bias, tensor);
		} else if (parts[2] == "ffn_norm") {
			layers[i]->attn_norm_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->attn_norm_weight, tensor);
		} else if (parts[2] == "ffn_norm_bias") {
			layers[i]->attn_norm_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->attn_norm_bias, tensor);
		}
	}
}

lstm * kokoro_model::prep_lstm() {
	lstm * rnn = new lstm;
	lstm_cell * cell = new lstm_cell;
	for (int i = 0; i < 8; i++) {
		cell->weights.push_back(nullptr);
		cell->biases.push_back(nullptr);
		cell->reverse_weights.push_back(nullptr);
		cell->reverse_biases.push_back(nullptr);
	}
	rnn->cells.push_back(cell);
	rnn->bidirectional = true;
	lstms.push_back(rnn);
	return rnn;
}

void kokoro_model::prep_layers(gguf_context * meta) {
	prosody_pred = new duration_predictor;
	prosody_pred->shared_lstm = prep_lstm();
	prosody_pred->duration_proj_lstm = prep_lstm();
	text_encoder = new kokoro_text_encoder;
	decoder = new kokoro_decoder;
	decoder->generator = new kokoro_generator;
	decoder->encoder_block = new ada_residual_conv_block;
	text_encoder->out_lstm = prep_lstm();

	for (int i = 0; i < n_layers; i++) {
		layers.push_back(new albert_layer);
	}

	for (int i = 0; i < f0_n_blocks; i++) {
		ada_residual_conv_block * f0 = new ada_residual_conv_block;
		ada_residual_conv_block * n = new ada_residual_conv_block;
		prosody_pred->f0_blocks.push_back(f0);
		prosody_pred->n_blocks.push_back(n);
	}

	for (int i = 0; i < n_duration_prediction_layers; i++) {
		duration_predictor_layer* dpl = new duration_predictor_layer;
		dpl->rnn = prep_lstm();
		prosody_pred->layers.push_back(dpl);
	}

	for (int i = 0; i < n_decoder_blocks; i++) {
		decoder->decoder_blocks.push_back(new ada_residual_conv_block);
	}

	for (int i = 0; i < n_noise_blocks; i++) {
		struct kokoro_noise_residual_block * nb = build_noise_block_from_file(meta, i);
		decoder->generator->noise_blocks.push_back(nb);
	}

	for (int i = 0; i < n_upsamples; i++) {
		struct kokoro_generator_upsample_block * ub = kokoro_generator_upsample_block(meta, i);
		decoder->generator->ups.push_back(ub);
	}

	for (int i = 0; i < n_res_blocks; i++) {
		struct kokoro_generator_residual_block* rb = build_res_block_from_file(meta, "kokoro.decoder.generator.res_blocks." + std::to_string(i));
		decoder->generator->res_blocks.push_back(rb);
	}

	for (int i = 0; i < n_conv_layers; i++) {
		text_encoder->conv_layers.push_back(new kokoro_text_encoder_conv_layer);
	}
}

void kokoro_model::prep_constants(gguf_context * meta) {
	// get constants for the Albert duration prediction model
	int context_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);;
    }

    int vocab_size_key = gguf_find_key(meta, "kokoro.tokenizer.vocab_size");
    if (vocab_size_key != -1) {
        vocab_size = gguf_get_val_u32(meta, vocab_size_key);
    }

    int hidden_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.hidden_size");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int attn_heads_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.attn_heads");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
        head_size = (uint32_t) hidden_size / n_attn_heads;
    }

    int albert_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.layers");
    if (albert_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, albert_layers_key);
    }

    int recurrence_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.recurrence");
    if (recurrence_key != -1) {
        n_recurrence = gguf_get_val_u32(meta, recurrence_key);
    }

    int duration_hidden_key = gguf_find_key(meta, "kokoro.duration_predictor.hidden_size");
    if (duration_hidden_key != -1) {
        duration_hidden_size = gguf_get_val_u32(meta, duration_hidden_key);
    }

    int up_sampling_factor_key = gguf_find_key(meta, "kokoro.decoder.generator.up_sampling_factor");
    if (up_sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, up_sampling_factor_key);
    }

	int f0_n_blocks_key = gguf_find_key(meta, "kokoro.duration_predictor.f0_n_blocks");
    if (f0_n_blocks_key != -1) {
        f0_n_blocks = gguf_get_val_u32(meta, f0_n_blocks_key);
    }

   	int duration_pred_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.layers");
    if (duration_pred_layers_key != -1) {
        n_duration_prediction_layers = gguf_get_val_u32(meta, duration_pred_layers_key);
    }

	// get text and decoding configuration for generation
	int n_conv_layers_key = gguf_find_key(meta, "kokoro.text_encoder.layers");
    if (n_conv_layers_key != -1) {
        n_conv_layers = gguf_get_val_u32(meta, n_conv_layers_key);
    }

   	int n_kernels_key = gguf_find_key(meta, "kokoro.decoder.generator.kernels");
    if (n_kernels_key != -1) {
        n_kernels = gguf_get_val_u32(meta, n_kernels_key);
    }

    int n_upsamples_key = gguf_find_key(meta, "kokoro.decoder.generator.upsamples");
    if (n_upsamples_key != -1) {
        n_upsamples = gguf_get_val_u32(meta, n_upsamples_key);
    }

    int n_decoder_blocks_key = gguf_find_key(meta, "kokoro.decoder.generator.layers");
    if (n_decoder_blocks_key != -1) {
        n_decoder_blocks = gguf_get_val_u32(meta, n_decoder_blocks_key);
    }

    int out_conv_padding_key = gguf_find_key(meta, "kokoro.decoder.generator.padding");
    if (out_conv_padding_key != -1) {
        out_conv_padding = gguf_get_val_u32(meta, out_conv_padding_key);
    }

    int n_fft_key = gguf_find_key(meta, "kokoro.decoder.generator.n_fft");
    if (n_fft_key != -1) {
        true_n_fft = gguf_get_val_u32(meta, n_fft_key);
        post_n_fft = (uint32_t) true_n_fft / 2 + 1;
    }

    int stft_hop_key = gguf_find_key(meta, "kokoro.decoder.generator.hop");
    if (stft_hop_key != -1) {
        stft_hop = gguf_get_val_u32(meta, stft_hop_key);
    }
}

kokoro_ubatch kokoro_duration_runner::build_worst_case_batch() {
	kokoro_ubatch batch;
	batch.n_tokens = model->max_context_length;
	return batch;
}

struct ggml_cgraph * kokoro_duration_runner::build_kokoro_duration_graph(kokoro_ubatch & batch) {
    init_build();
    // This '110000' number is coming from the number of nodes necessary for the longest possible sequence computed by of the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 110000, false);

    struct ggml_tensor * voice = model->voices[kctx->voice];
    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    if (!model->static_token_types) {
    	kctx->token_types = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    	ggml_set_input(kctx->token_types);
    }

    kctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->positions);

    inpL = build_albert_inputs(ctx, model, kctx->inp_tokens, kctx->positions, kctx->token_types);
    ggml_set_name(inpL, "albert_embeddings");
    cur = inpL;

    for (int r = 0; r < model->n_recurrence; r++) {
    	for (int l = 0; l < model->n_layers; l++) {
	        struct ggml_tensor * residual = cur ;
	        struct ggml_tensor * attn_out;

	        // self-attention
	        {
	            struct ggml_tensor * Qcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->q, cur), model->layers[l]->q_bias);
	            struct ggml_tensor * Kcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->k, cur), model->layers[l]->k_bias);
	            struct ggml_tensor * Vcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->v, cur), model->layers[l]->v_bias);

				Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
	            Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

	            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
	            struct ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
	            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

	            // Kokoro 的 duration predictor 不是自回归模型，不需要 causal mask；
	            // 且当前实现里 mask 一直是全 0。这里直接传 nullptr，避免每次推理都构造/填充 mask。
	            kq = ggml_soft_max_ext(ctx, kq, nullptr, model->scale, 0.0f);

	            struct ggml_tensor * v = ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
	            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
	            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
	            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
	            attn_out = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->o, attn_out), model->layers[l]->o_bias);
	        }
	        cur = ggml_add(ctx, attn_out, residual);
	        cur = build_albert_norm(ctx, cur, model->layers[l]->attn_norm_weight, model->layers[l]->attn_norm_bias);

	        struct ggml_tensor * residualffn = cur;

	        // ffn
	        {
                {
                    struct ggml_tensor * ffn_in = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn, cur), model->layers[l]->ffn_bias);
                    ffn_in = kokoro_vk_cont_if_misaligned_view(ctx, ffn_in, model);
                    cur = ggml_gelu(ctx, ffn_in);
                }
	        	cur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn_out, cur), model->layers[l]->ffn_out_bias);
	        }

			cur = ggml_add(ctx, cur, residualffn);
			cur = build_albert_norm(ctx, cur, model->layers[l]->layer_output_norm_weight, model->layers[l]->layer_output_norm_bias);
	    }
        ggml_build_forward_expand(gf, cur);
    }

    // duration / prosody prediction
    cur = ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->albert_encode, cur), model->prosody_pred->albert_encode_bias);

	struct ggml_tensor * style_half = ggml_cont(ctx, ggml_view_1d(ctx, voice, voice->ne[0]/2, voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]));

	cur = ggml_concat(ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);

    for (auto l : model->prosody_pred->layers) {
    	cur = build_lstm(ctx, cur, l->rnn, batch.n_tokens, gf);

    	struct ggml_tensor * gamma = ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_gamma_weight, style_half), l->ada_norm_gamma_bias);
    	struct ggml_tensor * beta = ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_beta_weight, style_half), l->ada_norm_beta_bias);

    	cur = ggml_norm(ctx, cur, 0.00001);

    	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
    	cur = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
    	cur = ggml_concat(ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);
    }

    struct ggml_tensor * d = ggml_cont(ctx, cur);
    ggml_set_name(d, "duration_hidden_states");
    ggml_build_forward_expand(gf, d);

    struct ggml_tensor * len;
    cur = build_lstm(ctx, cur, model->prosody_pred->duration_proj_lstm, batch.n_tokens, gf);
    cur = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->duration_proj, cur), model->prosody_pred->duration_proj_bias));
    // If we were to support speed we would add a constant tensor for the speed and divide here.
    len = ggml_round(ctx, ggml_sum_rows(ctx, cur));
    len = ggml_clamp(ctx, ggml_round(ctx, ggml_sum_rows(ctx, cur)), 1.0f, 50.0f);

    ggml_build_forward_expand(gf, len);

    free_build();

    return gf;
}

void kokoro_duration_runner::prepare_post_load() {
    // 说明：worst-case 图仅用于 sched 预分配；Vulkan 下默认跳过预分配以避免启动阶段大额申请/失败耗时。
    // 如需启用可设置环境变量：TTS_VK_PREALLOC=1
    if (tts_backend_is_vulkan(kctx->backend) && !tts_env_truthy("TTS_VK_PREALLOC")) {
        return;
    }
    auto batch = build_worst_case_batch();
    auto gf = build_kokoro_duration_graph(batch);
    kctx->prep_schedule(gf);
}

void kokoro_duration_runner::set_inputs(kokoro_ubatch & batch) {
	ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0, batch.n_tokens*ggml_element_size(kctx->inp_tokens));
	// 说明：Vulkan 后端下 tensor->data 可能位于 GPU 内存，CPU 侧不可直接写入。
	// 这里在 CPU 侧准备 positions 并通过 backend API 写入。
	static thread_local std::vector<uint32_t> positions_buf;
	positions_buf.resize(batch.n_tokens);
	for (uint32_t i = 0; i < batch.n_tokens; i++) {
		positions_buf[i] = i;
	}
	ggml_backend_tensor_set(kctx->positions, positions_buf.data(), 0, positions_buf.size() * sizeof(uint32_t));
}

void kokoro_model::free() {
    for (ggml_backend_buffer_t buf : extra_buffers) {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
    }
    extra_buffers.clear();
    tts_model::free();
}

void kokoro_duration_runner::run(kokoro_ubatch & batch) {
    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

    kctx->reset_graph();
    const int64_t t_after_sched_reset_us = timings ? tts_time_us() : 0;

    size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    size_t new_size = model->max_context_length * (model->duration_hidden_size + model->style_half_size) * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
	    if (kctx->buf_output) {
	        ggml_backend_buffer_free(kctx->buf_output);
	        kctx->buf_output = nullptr;
	        kctx->logits = nullptr;
	    }
	    kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
	}

    prev_size = kctx->buf_len_output ? ggml_backend_buffer_get_size(kctx->buf_len_output) : 0;
    new_size = model->max_context_length * sizeof(float);

    if (!kctx->buf_len_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_len_output);
            kctx->buf_len_output = nullptr;
            kctx->lens = nullptr;
        }

        kctx->buf_len_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }


    batch.resp->hidden_states = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);
    batch.resp->lengths = (float *) ggml_backend_buffer_get_base(kctx->buf_len_output);
    ggml_backend_buffer_clear(kctx->buf_len_output, 0);

    const int64_t t_after_buffers_us = timings ? tts_time_us() : 0;

    struct ggml_cgraph * gf = NULL;
    gf = build_kokoro_duration_graph(batch);
    const int64_t t_after_graph_build_us = timings ? tts_time_us() : 0;

    // the output is always the last tensor in the graph
    struct ggml_tensor * lens = gf->nodes[gf->n_nodes - 1];

    // 说明：duration 阶段需要把 token 级隐藏状态导出给 generator 使用。
    // 旧实现通过“固定偏移”从 gf->nodes 里取 hidden_states 指针；但图结构一旦微调（例如 LSTM 优化/算子替换），
    // 节点数量与顺序就会变化，导致取到错误节点并引发输出尺寸不匹配。
    // 这里改为按名称查找（build_kokoro_duration_graph 内已将其命名为 "duration_hidden_states"），更稳健。
    struct ggml_tensor * hidden_states = nullptr;
    for (int i = gf->n_nodes - 1; i >= 0; --i) {
        ggml_tensor * node = gf->nodes[i];
        const char * n = node ? ggml_get_name(node) : nullptr;
        if (n && std::strcmp(n, "duration_hidden_states") == 0) {
            hidden_states = node;
            break;
        }
    }
    if (!hidden_states) {
        TTS_ABORT("Kokoro 时长图未找到节点 'duration_hidden_states'，无法导出隐藏状态。\n");
    }
    kokoro_force_inputs_backend(kctx, gf);
    kokoro_force_custom_views_cpu(kctx, gf);
    kokoro_force_vk_misaligned_view_ops_cpu(kctx, gf);
    bool alloc_ok = kctx->alloc_graph(gf);
    if (!alloc_ok && tts_backend_is_vulkan(kctx->backend) && kctx->backend_cpu) {
        // 说明：Vulkan 分配失败时回退到 CPU，避免继续使用未分配成功的张量导致崩溃。
        fprintf(stderr, "[kokoro] Vulkan 时长图分配失败，回退 CPU 重新分配。\n");
        kctx->reset_graph();
        kokoro_force_graph_backend(kctx, gf, kctx->backend_cpu);
        alloc_ok = kctx->alloc_graph(gf);
    }
    if (!alloc_ok) {
        TTS_ABORT("Kokoro 时长图分配失败。\n");
    }

    if (tts_backend_is_vulkan(kctx->backend) && kctx->backend_cpu) {
        std::vector<ggml_tensor *> misaligned_nodes;
        if (kokoro_collect_vk_misaligned_nodes(kctx, gf, misaligned_nodes)) {
            fprintf(stderr,
                    "[kokoro] Vulkan 检测到 %zu 个未对齐节点，回退相关算子并重新分配。\n",
                    misaligned_nodes.size());
            kctx->reset_graph();
            kokoro_force_inputs_backend(kctx, gf);
            kokoro_force_custom_views_cpu(kctx, gf);
            kokoro_force_vk_misaligned_view_ops_cpu(kctx, gf);
            for (ggml_tensor * node : misaligned_nodes) {
                ggml_backend_sched_set_tensor_backend(kctx->sched, node, kctx->backend_cpu);
            }
            alloc_ok = kctx->alloc_graph(gf);
            if (!alloc_ok) {
                fprintf(stderr, "[kokoro] Vulkan 对齐回退后分配失败，改用 CPU。\n");
                kctx->reset_graph();
                kokoro_force_graph_backend(kctx, gf, kctx->backend_cpu);
                alloc_ok = kctx->alloc_graph(gf);
            }
            if (!alloc_ok) {
                TTS_ABORT("Kokoro 时长图分配失败。\n");
            }
        }
    }

    const int64_t t_after_sched_alloc_us = timings ? tts_time_us() : 0;

    set_inputs(batch);

    const int64_t t_after_set_inputs_us = timings ? tts_time_us() : 0;

    // 说明：CPU-only 模式下，时长图以小算子为主，过多线程常常会被调度/同步开销抵消。
    // 这里对时长阶段单独设置一个较小的线程数上限（默认 4，可通过 TTS_CPU_THREADS_DURATION 覆盖）。
    if (kctx->backend == nullptr && kctx->backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(kctx->backend_cpu, kokoro_duration_threads(kctx));
    }

    const enum ggml_status duration_status = kctx->compute_graph_async(gf);
    if (duration_status != GGML_STATUS_SUCCESS) {
        TTS_ABORT("Kokoro 时长计算失败：status=%d。\n", (int) duration_status);
    }
    const int64_t t_after_compute_call_us = timings ? tts_time_us() : 0;

    kctx->get_ggml_node_data(lens, batch.resp->lengths, batch.n_tokens*sizeof(float), kctx->buf_len_output);
    const int64_t t_after_get_lens_us = timings ? tts_time_us() : 0;
    kctx->get_ggml_node_data(hidden_states, batch.resp->hidden_states, batch.n_tokens*(model->duration_hidden_size+model->style_half_size)*sizeof(float));
    const int64_t t_after_get_states_us = timings ? tts_time_us() : 0;

    kctx->sync();
    // 说明：异步后端需要先同步，避免 reset 释放仍在使用的 buffer。
    kctx->reset_graph();
    const int64_t t_after_sync_us = timings ? tts_time_us() : 0;
    batch.resp->n_outputs = batch.n_tokens;

    if (timings) {
        uint32_t total_frames = 0;
        for (size_t i = 0; i < batch.n_tokens; ++i) {
            total_frames += (uint32_t) batch.resp->lengths[i];
        }

        fprintf(stderr,
                "[kokoro][timings] duration: sched_reset=%.2fms outbuf=%.2fms graph_build=%.2fms sched_alloc=%.2fms set_inputs=%.2fms compute=%.2fms get_len=%.2fms get_states=%.2fms sync=%.2fms total=%.2fms (tokens=%zu frames=%u)\n",
                us_to_ms(t_after_sched_reset_us - t_start_us),
                us_to_ms(t_after_buffers_us - t_after_sched_reset_us),
                us_to_ms(t_after_graph_build_us - t_after_buffers_us),
                us_to_ms(t_after_sched_alloc_us - t_after_graph_build_us),
                us_to_ms(t_after_set_inputs_us - t_after_sched_alloc_us),
                us_to_ms(t_after_compute_call_us - t_after_set_inputs_us),
                us_to_ms(t_after_get_lens_us - t_after_compute_call_us),
                us_to_ms(t_after_get_states_us - t_after_get_lens_us),
                us_to_ms(t_after_sync_us - t_after_get_states_us),
                us_to_ms(t_after_sync_us - t_start_us),
                batch.n_tokens,
                total_frames);
    }
}

kokoro_ubatch kokoro_runner::build_worst_case_batch() {
	kokoro_ubatch batch;
	batch.n_tokens = model->max_context_length;
	batch.resp = new kokoro_duration_response;
	batch.resp->n_outputs = model->max_context_length;
	kctx->total_duration = model->max_context_length * model->max_duration_per_token;
	kctx->sequence_length = model->max_context_length;
	std::vector<float> lengths;
	lengths.reserve(model->max_context_length);
	for (int i = 0; i < model->max_context_length; i++) {
		lengths.push_back(50.0f);
	}
	batch.resp->lengths = lengths.data();
	return batch;
}

struct ggml_cgraph * kokoro_runner::build_kokoro_graph(kokoro_ubatch & batch) {
    init_build();
    // This '570000' number is coming from the number of nodes necessary for the longest possible sequence computed by the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 570000, false);

    struct ggml_tensor * voice = model->voices[kctx->voice];
    // 说明：Vulkan 要求 storage buffer offset 对齐，style_half 可能产生未对齐 view；
    // 这里强制做一次 cont，确保后续 Vulkan 算子读取到对齐地址，避免断言崩溃。
    struct ggml_tensor * style_half = ggml_cont(ctx, ggml_view_1d(ctx, voice, voice->ne[0]/2,
                                                                 voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]));
    struct ggml_tensor * cur;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    // 说明：duration_ids[t] 表示第 t 帧由哪个 token 负责（0..sequence_length-1）。
    // 用 get_rows 直接把 token 级别特征 gather 到 frame 级别，
    // 替代旧版的 duration_mask(tokens×frames) + mul_mat 展开：
    // - 避免构造大规模稀疏 mask（内存与写入开销大）；
    // - 避免两次 matmul（算子数与计算量显著下降，Vulkan/CPU 都更快）。
    kctx->duration_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kctx->total_duration);
    ggml_set_input(kctx->duration_ids);

    kctx->duration_pred = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->duration_hidden_size + model->style_half_size, kctx->sequence_length);
    ggml_set_input(kctx->duration_pred);

    // 说明：直接把 duration_pred（hidden×tokens）展开为（hidden×frames）。
    cur = ggml_get_rows(ctx, kctx->duration_pred, kctx->duration_ids);
    cur = tts_cont_if_needed(ctx, cur);

    cur = build_lstm(ctx, cur, model->prosody_pred->shared_lstm, cur->ne[1], gf);


    struct ggml_tensor * f0_curve = cur;
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    for (auto block : model->prosody_pred->f0_blocks) {
    	f0_curve = build_ada_residual_conv(ctx, f0_curve, block, style_half, model->sqrt_tensor);
    }
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    f0_curve = ggml_mul_mat(ctx, model->prosody_pred->f0_proj_kernel, f0_curve);
    f0_curve = squeeze_3d_2d_e0(ctx, f0_curve);
    f0_curve = ggml_add(ctx, f0_curve, model->prosody_pred->f0_proj_bias);
    ggml_set_name(f0_curve, "f0_out");

    struct ggml_tensor * n = cur;
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
    for (auto block : model->prosody_pred->n_blocks) {
		n = build_ada_residual_conv(ctx, n, block, style_half, model->sqrt_tensor);
    }
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
	n = ggml_mul_mat(ctx, model->prosody_pred->n_proj_kernel, n);
	n = squeeze_3d_2d_e0(ctx, n);
	n = ggml_add(ctx, n, model->prosody_pred->n_proj_bias);
	ggml_set_name(n, "n_out");
	ggml_build_forward_expand(gf, n);

	// kokoro text encoding;
	struct ggml_tensor * asr;
	//struct ggml_tensor * embd;
	{
		cur = ggml_get_rows(ctx, model->text_encoder->embd, kctx->inp_tokens);

		for (auto l : model->text_encoder->conv_layers) {
			cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, tts_conv_1d(ctx, l->conv_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, 2, 1), l->conv_bias)));
			cur = ggml_norm(ctx, cur, 0.00001);
			cur = ggml_add(ctx, ggml_mul(ctx, cur, l->norm_gamma), l->norm_beta);
			cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
		}

 		cur = build_lstm(ctx, cur, model->text_encoder->out_lstm, kctx->sequence_length, gf);
        // 说明：把 token 级别的 text encoder 输出直接 gather 到 frame 级别。
        cur = tts_cont_if_needed(ctx, cur);
        asr = ggml_get_rows(ctx, cur, kctx->duration_ids);
        asr = tts_cont_if_needed(ctx, asr);
 	}

	// decoding and generation prep
	struct ggml_tensor * asr_res;
	struct ggml_tensor * f0;
	struct ggml_tensor * n_base;
    // 说明：同上，style_half2 也可能在 Vulkan 下产生未对齐 view，提前 cont 规避。
    struct ggml_tensor * style_half2 = ggml_cont(ctx, ggml_view_1d(ctx, voice, voice->ne[0]/2,
                                                                  (batch.n_tokens - 3) * voice->nb[1]));

	{
		f0 = ggml_add(ctx, tts_conv_1d(ctx, model->decoder->f0_conv, f0_curve, 2, 1, 1), model->decoder->f0_conv_bias);
		n_base = ggml_add(ctx, tts_conv_1d(ctx, model->decoder->n_conv, n, 2, 1, 1), model->decoder->n_conv_bias);
		cur = ggml_concat(ctx, ggml_concat(ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)), f0, 1), n_base, 1);
		cur = build_ada_residual_conv(ctx, cur, model->decoder->encoder_block, style_half2, model->sqrt_tensor);
		ggml_build_forward_expand(gf, cur);

		asr_res = ggml_mul_mat(ctx, model->decoder->asr_conv, asr);
		// 说明：同样避免 src1 非连续视图参与 broadcast（ggml 0.9.4 CPU 路径会断言失败）。
		asr_res = ggml_add(ctx, asr_res, ggml_cont(ctx, ggml_transpose(ctx, model->decoder->asr_conv_bias)));

		asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));
		for (auto l : model->decoder->decoder_blocks) {
			cur = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, cur, asr_res, 1), f0, 1), n_base, 1 );
			cur = build_ada_residual_conv(ctx, cur, l, style_half2, model->sqrt_tensor);
			ggml_build_forward_expand(gf, cur);
		}
		// 说明：生成器内部改为时间优先布局，避免在生成入口做额外 transpose。
		cur = tts_cont_if_needed(ctx, cur);
	}

	kctx->window_sq_sum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kctx->total_duration*model->up_sampling_factor);
	ggml_set_input(kctx->window_sq_sum);

	// run generation
	cur = build_generator(ctx, &*model, kctx, cur, style_half2, f0_curve, model->decoder->generator, (int)kctx->sequence_length, kctx->window_sq_sum, gf);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void kokoro_runner::prepare_post_load() {
    propagate_voice_setting();
	model->post_load_assign();
	drunner->prepare_post_load();

    // 说明：worst-case 图仅用于 sched 预分配；Vulkan 下默认跳过预分配以避免启动阶段大额申请/失败耗时。
    // 如需启用可设置环境变量：TTS_VK_PREALLOC=1
    if (tts_backend_is_vulkan(kctx->backend) && !tts_env_truthy("TTS_VK_PREALLOC")) {
        return;
    }
    auto batch = build_worst_case_batch();
    auto gf = build_kokoro_graph(batch);
    kctx->prep_schedule(gf);
    // batch.resp 由 new 分配，必须用 delete 释放（避免潜在堆损坏）。
    delete batch.resp;
    batch.resp = nullptr;
}

kokoro_gen_input_timings kokoro_runner::set_inputs(kokoro_ubatch & batch, uint32_t total_size) {
    kokoro_gen_input_timings out{};

    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

    // 说明：Vulkan 后端下 tensor->data 可能位于 GPU 内存，CPU 侧不可直接写入。
    // 这里在 CPU 侧准备 uv/noise 输入，并通过 backend API 写入。
    const size_t uv_bytes = ggml_nbytes(kctx->uv_noise_data);
    const size_t uv_elems = uv_bytes / sizeof(float);
    static thread_local std::vector<float> uv_noise_buf;
    uv_noise_buf.resize(uv_elems);
    if (uv_elems >= 4) {
        uv_noise_buf[0] = model->voice_threshold;
        uv_noise_buf[1] = model->noise_std;
        uv_noise_buf[2] = model->sin_amp;
        uv_noise_buf[3] = model->sin_amp / 3.0f;
    }
    const size_t uv_rand_elems = uv_elems > 4 ? (uv_elems - 4) : 0;
    if (uv_rand_elems > 0) {
        random_uniform_gen((int) uv_rand_elems, uv_noise_buf.data() + 4);
    }
    ggml_backend_tensor_set(kctx->uv_noise_data, uv_noise_buf.data(), 0, uv_bytes);

    const int64_t t_after_noise_us = timings ? tts_time_us() : 0;

    const size_t window_bytes = ggml_nbytes(kctx->window_sq_sum);
    const size_t window_elems = window_bytes / sizeof(float);
    static thread_local std::vector<float> window_sq_sum_buf;
    window_sq_sum_buf.resize(window_elems);
    if (model->decoder->generator->window_host.size() != model->true_n_fft) {
        model->decoder->generator->window_host.clear();
        model->decoder->generator->window_host.reserve(model->true_n_fft);
        hann_window(model->true_n_fft, model->decoder->generator->window_host);
    }
    compute_window_squared_sum(model->true_n_fft, model->stft_hop,
                               total_size * model->up_sampling_factor / model->stft_hop,
                               window_sq_sum_buf.data(),
                               model->decoder->generator->window_host.data());
    ggml_backend_tensor_set(kctx->window_sq_sum, window_sq_sum_buf.data(), 0, window_bytes);

    const int64_t t_after_window_us = timings ? tts_time_us() : 0;

    // 说明：Vulkan 图中的“标量常量输入”，在这里统一写入。
    if (!kctx->graph_const_inputs.empty()) {
        for (const auto & item : kctx->graph_const_inputs) {
            if (item.tensor == nullptr) {
                continue;
            }
            ggml_backend_tensor_set(item.tensor, &item.value, 0, sizeof(float));
        }
    }

    // 说明：STFT 反射 padding 索引（用于 Vulkan 图）。
    if (kctx->stft_pad_indices != nullptr) {
        const int64_t pad = (int64_t) (model->true_n_fft / 2);
        const int64_t padded_len = kctx->stft_pad_indices->ne[0];
        const int64_t batch = kctx->stft_pad_indices->ne[1];
        const int64_t in_len = padded_len - 2 * pad;
        if (in_len <= 0) {
            TTS_ABORT("stft_pad_indices: invalid length (padded=%d, pad=%d)\n", (int) padded_len, (int) pad);
        }

        const size_t idx_bytes = ggml_nbytes(kctx->stft_pad_indices);
        const size_t idx_elems = idx_bytes / sizeof(int32_t);
        static thread_local std::vector<int32_t> idx_buf;
        idx_buf.resize(idx_elems);

        for (int64_t b = 0; b < batch; ++b) {
            int32_t * dst = idx_buf.data() + (size_t) b * (size_t) padded_len;
            for (int64_t p = 0; p < padded_len; ++p) {
                const int64_t ai = p - pad;
                int64_t src = 0;
                if (ai < 0) {
                    src = -ai;
                } else if (ai >= in_len) {
                    src = in_len - (ai - in_len + 1);
                } else {
                    src = ai;
                }
                if (src < 0) {
                    src = 0;
                } else if (src >= in_len) {
                    src = in_len - 1;
                }
                dst[p] = (int32_t) src;
            }
        }
        ggml_backend_tensor_set(kctx->stft_pad_indices, idx_buf.data(), 0, idx_bytes);
    }

    kctx->sequence_length = batch.n_tokens;
    kctx->total_duration  = total_size;
    ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0,
                            batch.n_tokens * ggml_element_size(kctx->inp_tokens));
    ggml_backend_tensor_set(kctx->duration_pred, batch.resp->hidden_states, 0,
                            batch.n_tokens * (model->duration_hidden_size + model->style_half_size) *
                                ggml_element_size(kctx->duration_pred));

    const int64_t t_after_backend_sets_us = timings ? tts_time_us() : 0;

    // duration_ids[t] 表示第 t 帧对应的 token 索引（I32）。
    // 旧版 duration_mask 需要构造 tokens×frames 的稀疏矩阵，再做两次 mul_mat；
    // 这里改为“一维索引 + get_rows gather”，输入更小、构图更简单、速度更快。
    const size_t duration_ids_bytes = ggml_nbytes(kctx->duration_ids);
    const size_t duration_ids_elems = duration_ids_bytes / sizeof(int32_t);
    static thread_local std::vector<int32_t> duration_ids_buf;
    duration_ids_buf.resize(duration_ids_elems);

    const size_t n_frames = (size_t) total_size;
    if (n_frames == 0 || batch.n_tokens == 0) {
        if (duration_ids_bytes > 0) {
            std::fill(duration_ids_buf.begin(), duration_ids_buf.end(), 0);
            ggml_backend_tensor_set(kctx->duration_ids, duration_ids_buf.data(), 0, duration_ids_bytes);
        }
        const int64_t t_end_us = timings ? tts_time_us() : 0;
        if (timings) {
            out.noise_ms         = us_to_ms(t_after_noise_us - t_start_us);
            out.window_sq_sum_ms = us_to_ms(t_after_window_us - t_after_noise_us);
            out.tensor_set_ms    = us_to_ms(t_after_backend_sets_us - t_after_window_us);
            out.duration_mask_ms = us_to_ms(t_end_us - t_after_backend_sets_us);
            out.total_ms         = us_to_ms(t_end_us - t_start_us);
        }
        return out;
    }
    if (duration_ids_elems != n_frames) {
        TTS_ABORT("duration_ids size mismatch: elems=%zu expected=%zu (tokens=%u, frames=%zu)\n",
                  duration_ids_elems, n_frames, batch.n_tokens, n_frames);
    }

    uint32_t running = 0;
    for (uint32_t i = 0; i < batch.n_tokens; i++) {
        const uint32_t len   = (uint32_t) batch.resp->lengths[i];
        const uint32_t start = running;
        const uint32_t end   = std::min(running + len, total_size);
        if (end > start) {
            for (uint32_t p = start; p < end; ++p) {
                duration_ids_buf[p] = (int32_t) i;
            }
        }
        running = end;
    }
    // 说明：防御性补齐（理论上 running 应等于 total_size）。
    if (running < total_size) {
        const int32_t fallback_id = batch.n_tokens > 0 ? (int32_t) (batch.n_tokens - 1) : 0;
        for (uint32_t p = running; p < total_size; ++p) {
            duration_ids_buf[p] = fallback_id;
        }
    }
    ggml_backend_tensor_set(kctx->duration_ids, duration_ids_buf.data(), 0, duration_ids_bytes);

    if (timings) {
        const int64_t t_end_us = tts_time_us();
        out.noise_ms         = us_to_ms(t_after_noise_us - t_start_us);
        out.window_sq_sum_ms = us_to_ms(t_after_window_us - t_after_noise_us);
        out.tensor_set_ms    = us_to_ms(t_after_backend_sets_us - t_after_window_us);
        out.duration_mask_ms = us_to_ms(t_end_us - t_after_backend_sets_us);
        out.total_ms         = us_to_ms(t_end_us - t_start_us);
    }

    return out;
}

void kokoro_runner::run(kokoro_ubatch & batch, tts_response & outputs) {
    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

	batch.resp = new kokoro_duration_response;
	drunner->run(batch);

    const int64_t t_after_duration_us = timings ? tts_time_us() : 0;

    kctx->reset_graph();
    const int64_t t_after_sched_reset_us = timings ? tts_time_us() : 0;

    const size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    uint32_t total_length = 0;
    for (int i = 0; i < batch.resp->n_outputs; i++) {
    	total_length += (uint32_t) batch.resp->lengths[i];
    }
    const size_t new_size = total_length * model->up_sampling_factor * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_output);
            kctx->buf_output = nullptr;
            kctx->logits = nullptr;
        }
        kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }

    outputs.data = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);

    kctx->sequence_length = batch.n_tokens;
	kctx->total_duration = total_length;

    const int64_t t_after_outbuf_us = timings ? tts_time_us() : 0;

    struct ggml_cgraph * gf = NULL;
    gf = build_kokoro_graph(batch);
    const int64_t t_after_graph_build_us = timings ? tts_time_us() : 0;

    // the output is always the last tensor in the graph
    struct ggml_tensor * output = gf->nodes[gf->n_nodes - 1];

    const bool force_vulkan_gen = kokoro_env_force_vulkan_gen();
    const bool has_custom_ops = kokoro_graph_has_custom_ops(gf);
    if (tts_backend_is_vulkan(kctx->backend)) {
        if (has_custom_ops && !force_vulkan_gen) {
            // 说明：自定义算子固定在 CPU，其余节点仍让 Vulkan 参与计算。
            kokoro_force_custom_ops_cpu(kctx, gf);
            kokoro_force_custom_views_cpu(kctx, gf);
            kokoro_force_supported_ops_backend(kctx, gf, kctx->backend);
            kokoro_force_inputs_backend(kctx, gf);
            kokoro_force_vk_stft_istft_cpu_if_needed(kctx, gf, model.get());
            kokoro_force_vk_misaligned_view_ops_cpu(kctx, gf);
            fprintf(stderr,
                    "[kokoro] Vulkan 后端检测到自定义算子，已固定为 CPU；其余节点仍走 Vulkan。如需强制全部 Vulkan 可设置 TTS_VK_FORCE_GEN=1。\n");
        } else {
            kokoro_force_inputs_backend(kctx, gf);
            kokoro_force_vk_stft_istft_cpu_if_needed(kctx, gf, model.get());
            kokoro_force_vk_misaligned_view_ops_cpu(kctx, gf);
        }
    }
    bool alloc_ok = kctx->alloc_graph(gf);
    if (!alloc_ok && tts_backend_is_vulkan(kctx->backend) && kctx->backend_cpu) {
        // 说明：Vulkan 分配失败时回退到 CPU，避免继续使用未分配成功的张量导致崩溃。
        fprintf(stderr, "[kokoro] Vulkan 生成图分配失败，回退 CPU 重新分配。\n");
        kctx->reset_graph();
        kokoro_force_graph_backend(kctx, gf, kctx->backend_cpu);
        alloc_ok = kctx->alloc_graph(gf);
    }
    if (!alloc_ok) {
        TTS_ABORT("Kokoro 生成图分配失败。\n");
    }

    if (tts_backend_is_vulkan(kctx->backend) && kctx->backend_cpu) {
        std::vector<ggml_tensor *> misaligned_nodes;
        if (kokoro_collect_vk_misaligned_nodes(kctx, gf, misaligned_nodes)) {
            fprintf(stderr,
                    "[kokoro] Vulkan 检测到 %zu 个未对齐节点，回退相关算子并重新分配。\n",
                    misaligned_nodes.size());
            kctx->reset_graph();
            if (has_custom_ops && !force_vulkan_gen) {
                kokoro_force_custom_ops_cpu(kctx, gf);
                kokoro_force_custom_views_cpu(kctx, gf);
                kokoro_force_supported_ops_backend(kctx, gf, kctx->backend);
                kokoro_force_inputs_backend(kctx, gf);
                kokoro_force_vk_stft_istft_cpu_if_needed(kctx, gf, model.get());
                kokoro_force_vk_misaligned_view_ops_cpu(kctx, gf);
            } else {
                kokoro_force_inputs_backend(kctx, gf);
                kokoro_force_vk_stft_istft_cpu_if_needed(kctx, gf, model.get());
                kokoro_force_vk_misaligned_view_ops_cpu(kctx, gf);
            }
            for (ggml_tensor * node : misaligned_nodes) {
                ggml_backend_sched_set_tensor_backend(kctx->sched, node, kctx->backend_cpu);
            }
            alloc_ok = kctx->alloc_graph(gf);
            if (!alloc_ok) {
                fprintf(stderr, "[kokoro] Vulkan 对齐回退后分配失败，改用 CPU。\n");
                kctx->reset_graph();
                kokoro_force_graph_backend(kctx, gf, kctx->backend_cpu);
                alloc_ok = kctx->alloc_graph(gf);
            }
            if (!alloc_ok) {
                TTS_ABORT("Kokoro 生成图分配失败。\n");
            }
        }
    }

    const int64_t t_after_sched_alloc_us = timings ? tts_time_us() : 0;

    int64_t t_before_set_inputs_us = timings ? tts_time_us() : 0;
    kokoro_gen_input_timings input_timings = set_inputs(batch, total_length);
    int64_t t_after_set_inputs_us = timings ? tts_time_us() : 0;

    // 说明：duration 阶段可能把 CPU 线程数调小；进入 generator 前恢复为用户指定线程数，
    // 以充分利用多核加速卷积/矩阵运算。
    if (kctx->backend == nullptr && kctx->backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(kctx->backend_cpu, kctx->n_threads);
    }

    enum ggml_status gen_status = kctx->compute_graph_async(gf);
    if (gen_status != GGML_STATUS_SUCCESS && tts_backend_is_vulkan(kctx->backend) && kctx->backend_cpu) {
        fprintf(stderr, "[kokoro] Vulkan 生成计算失败，回退 CPU 重新计算。\n");
        kctx->reset_graph();
        kokoro_force_graph_backend(kctx, gf, kctx->backend_cpu);
        bool retry_alloc_ok = kctx->alloc_graph(gf);
        if (!retry_alloc_ok) {
            TTS_ABORT("Kokoro 生成图分配失败。\n");
        }
        t_before_set_inputs_us = timings ? tts_time_us() : 0;
        input_timings = set_inputs(batch, total_length);
        t_after_set_inputs_us = timings ? tts_time_us() : 0;
        gen_status = kctx->compute_graph_async(gf);
    }
    if (gen_status != GGML_STATUS_SUCCESS) {
        TTS_ABORT("Kokoro 生成计算失败：status=%d。\n", (int) gen_status);
    }
    const int64_t t_after_compute_call_us = timings ? tts_time_us() : 0;

    kctx->get_ggml_node_data(output, outputs.data, new_size);
    const int64_t t_after_get_call_us = timings ? tts_time_us() : 0;

    kctx->sync();
    // 说明：异步后端需要先同步，避免 reset 释放仍在使用的 buffer。
    kctx->reset_graph();
    const int64_t t_after_sync_us = timings ? tts_time_us() : 0;
    outputs.n_outputs = total_length*model->up_sampling_factor;
    // batch.resp 由 new 分配，必须用 delete 释放（避免潜在堆损坏）。
    delete batch.resp;
    batch.resp = nullptr;

    if (timings) {
        const double audio_sec = outputs.n_outputs > 0 ? (double) outputs.n_outputs / (double) model->sample_rate : 0.0;
        const double total_sec = (double) (t_after_sync_us - t_start_us) / 1000000.0;
        const double rtf = audio_sec > 0.0 ? total_sec / audio_sec : 0.0;

        // 输入准备的细分耗时（该行不包含打印本身的时间）。
        fprintf(stderr,
                "[kokoro][timings] gen_inputs: call=%.2fms noise=%.2fms window_sq_sum=%.2fms tensor_set=%.2fms duration_mask=%.2fms total=%.2fms (tokens=%zu frames=%u)\n",
                us_to_ms(t_after_set_inputs_us - t_before_set_inputs_us),
                input_timings.noise_ms,
                input_timings.window_sq_sum_ms,
                input_timings.tensor_set_ms,
                input_timings.duration_mask_ms,
                input_timings.total_ms,
                batch.n_tokens,
                total_length);

        fprintf(stderr,
                "[kokoro][timings] generator: duration=%.2fms sched_reset=%.2fms outbuf=%.2fms graph_build=%.2fms sched_alloc=%.2fms set_inputs=%.2fms compute=%.2fms get=%.2fms sync=%.2fms total=%.2fms audio=%.3fs rtf=%.3f (tokens=%zu frames=%u samples=%zu)\n",
                us_to_ms(t_after_duration_us - t_start_us),
                us_to_ms(t_after_sched_reset_us - t_after_duration_us),
                us_to_ms(t_after_outbuf_us - t_after_sched_reset_us),
                us_to_ms(t_after_graph_build_us - t_after_outbuf_us),
                us_to_ms(t_after_sched_alloc_us - t_after_graph_build_us),
                us_to_ms(t_after_set_inputs_us - t_after_sched_alloc_us),
                us_to_ms(t_after_compute_call_us - t_after_set_inputs_us),
                us_to_ms(t_after_get_call_us - t_after_compute_call_us),
                us_to_ms(t_after_sync_us - t_after_get_call_us),
                us_to_ms(t_after_sync_us - t_start_us),
                audio_sec,
                rtf,
                batch.n_tokens,
                total_length,
                outputs.n_outputs);
    }
    return;
}

void kokoro_runner::assign_weight(const char * name, ggml_tensor & tensor) {
    const string_view name_sv{ name };
    GGML_ASSERT(tts_starts_with(name_sv, "kokoro."));
    const string trimmed{ name_sv.substr(sizeof("kokoro.") - 1) };
    model->assign_weight(trimmed.c_str(), tensor);
}

/*
 * #tokenize_chunks is used to split up a larger than max context size (512) token prompt into discrete
 * blocks for generation. This solution, in accordance with Kokoro's pyTorch implementation, splits
 * the prompt by sentence when possible (this can result in slower inference but generally produces cleaner
 * speech). If a disinct sentence is too long, then it splits at the nearest space.
 */
std::vector<std::vector<uint32_t>> kokoro_runner::tokenize_chunks(std::vector<std::string> clauses) {
	std::vector<std::vector<uint32_t>> chunks;
	for (auto clause : clauses) {
		clause = strip(clause);
		if (clause.empty()) {
			continue;
		}
		std::vector<uint32_t> tokens;
		tokens.push_back(model->bos_token_id);
		tokenizer->tokenize(clause, tokens);
		// if there are more clause tokens than the max context length then try to split by space tokens.
		// To be protective, split mid-word when there are no spaces (this should never happen).
		if (tokens.size() > model->max_context_length - 2) {
			// we skip the first token here becuase it is the bos token.
			int last_space_token = 1;
			int last_split = 1;
			for (int i = 1; i < tokens.size(); i++) {
				if (tokens[i] == model->space_token_id) {
					last_space_token = i;
				}
				if ((i - last_split) + chunks.back().size() >= model->max_context_length - 1) {
					if (last_space_token > last_split) {
						std::vector<uint32_t> portion = { model->bos_token_id };
						portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + last_space_token);
						portion.push_back(model->eos_token_id);
						chunks.push_back(portion);
						last_split = last_space_token;
					} else {
						std::vector<uint32_t> portion = { model->bos_token_id };
						portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + i + 1);
						portion.push_back(model->eos_token_id);
						chunks.push_back(portion);
						last_split = i + 1;
					}
				}
			}
			if (last_split + 1 < tokens.size()) {
				std::vector<uint32_t> portion = { model->bos_token_id };
				portion.insert(portion.end(), tokens.begin() + last_split, tokens.end());
				portion.push_back(model->eos_token_id);
				chunks.push_back(portion);
			}
		} else {
			tokens.push_back(model->eos_token_id);
			chunks.push_back(tokens);
		}
	}
	return chunks;
}

void kokoro_runner::propagate_voice_setting() {
    if (voice.empty()) {
        if (model->voices.find("af_heart") != model->voices.end()) {
            voice = "af_heart";
        } else if (!model->voices.empty()) {
            voice = model->voices.begin()->first;
        }
    }
    if (voice.empty() || model->voices.find(voice) == model->voices.end()) {
        TTS_ABORT("Failed to find Kokoro voice '%s' aborting.\n", voice.c_str());
    }
    kctx->voice          = voice;
    drunner->kctx->voice = voice;
}

bool kokoro_runner::try_phonemize(const char * prompt, std::string & out_phonemes, const generation_configuration & config) {
    // 说明：该接口仅用于 CLI 调试输出（原文/音素串），不执行推理。
    // 这里尽量复用 generate() 的“音素化前半段逻辑”，确保输出与实际推理一致。
    voice = config.voice;
    propagate_voice_setting();

    std::string normalized = prompt ? std::string(prompt) : std::string();
    normalized = replace_any(std::move(normalized), "\n", " ");

    // Enable built-in zh phonemization when either:
    // - voice is Mandarin (z*)  OR
    // - prompt contains CJK characters
    const bool contains_cjk = kokoro_contains_cjk(normalized);
    const bool use_multilingual = (config.language == tts_language::ZH) ||
                                  (config.language == tts_language::JA) ||
                                  (!voice.empty() && (voice[0] == 'z' || voice[0] == 'j')) ||
                                  contains_cjk;

    std::string phonemized_prompt;
    if (use_multilingual) {
        phonemized_prompt = kokoro_phonemize_multilingual(normalized, phmzr, config.language, config.zh_dict_dir);
    } else {
        phonemized_prompt = phmzr ? phmzr->text_to_phonemes(normalized) : "";
    }

    // 说明：保持与 generate() 一致的“句末标点处理”逻辑：
    // - 短文本：把 ".!?" 替换为 ","（弱停顿），避免触发 EOS 提前结束；
    // - 长文本：进入 chunking 逻辑（这里不做拼接，直接返回原始音素串）。
    const size_t max_no_special = model->max_context_length > 2 ? (size_t) model->max_context_length - 2 : 0;
    if (!phonemized_prompt.empty() && tts_utf8_codepoint_count(phonemized_prompt) <= max_no_special) {
        phonemized_prompt = strip(replace_any(std::move(phonemized_prompt), ".!?", ","));
    }

    out_phonemes = std::move(phonemized_prompt);
    return !out_phonemes.empty();
}

bool kokoro_runner::try_phonemize_segments(const char * prompt,
                                          std::string & out_phonemes,
                                          std::vector<tts_generation_runner::phoneme_segment> & out_segments,
                                          const generation_configuration & config) {
    // 说明：用于 CLI 调试打印（分词 + 音素），不执行推理。
    // 设计目标：
    // - 输出的 out_phonemes 尽量与 generate() 最终喂给 tokenizer 的串一致（包含短文本标点替换）；
    // - out_segments 更偏“可读性”：中文走词典 DP 分词，输出“词(音素)”方便肉眼排查多音字/数字/卷舌等问题。
    out_segments.clear();
    out_phonemes.clear();

    if (!try_phonemize(prompt, out_phonemes, config)) {
        return false;
    }

    // 目前仅对中文偏好输出“分词级别”的 segments；其它语言退化为“整句一个 segment”。
    if (config.language != tts_language::ZH) {
        tts_generation_runner::phoneme_segment seg;
        seg.text = prompt ? std::string(prompt) : std::string();
        seg.phonemes = out_phonemes;
        seg.is_boundary = false;
        out_segments.push_back(std::move(seg));
        return true;
    }

    // 说明：分词逻辑以中文前端为准；为保证与实际推理一致，这里先应用“单位归一化”。
    std::string text = prompt ? std::string(prompt) : std::string();
    text = replace_any(std::move(text), "\n", " ");
    text = kokoro_normalize_zh_units(text);

    const kokoro_zh::zh_debug_result dbg = kokoro_zh::text_to_zh_phonemes_debug(text, config.zh_dict_dir);
    out_segments.reserve(dbg.items.size());
    for (const auto & it : dbg.items) {
        tts_generation_runner::phoneme_segment seg;
        seg.text = it.text;
        seg.phonemes = it.phonemes;
        seg.is_boundary = it.is_boundary;
        out_segments.push_back(std::move(seg));
    }

    return true;
}

void kokoro_runner::generate(const char * prompt, tts_response & response, const generation_configuration & config) {
    const bool timings = tts_timings_enabled();
    const int64_t t_start_us = timings ? tts_time_us() : 0;

    voice = config.voice;
    propagate_voice_setting();
    std::string normalized{prompt};
    normalized = replace_any(normalized, "\n", " ");

    const int64_t t_after_normalize_us = timings ? tts_time_us() : 0;

    std::string phonemized_prompt;
    // Enable built-in zh phonemization when either:
    // - voice is Mandarin (z*)  OR
    // - prompt contains CJK characters
    const bool contains_cjk = kokoro_contains_cjk(normalized);
    // 说明：
    // - language=ZH/JA 时强制走多语言前端：
    //   - ZH：数字按中文读法处理
    //   - JA：日文片段按假名/标注读法处理
    // - voice=z*/j* 或 prompt 含 CJK（含日文假名）时也走多语言前端，避免被英文 phonemizer 误处理。
    const bool use_multilingual = (config.language == tts_language::ZH) ||
                                  (config.language == tts_language::JA) ||
                                  (!voice.empty() && (voice[0] == 'z' || voice[0] == 'j')) ||
                                  contains_cjk;
    if (use_multilingual) {
        phonemized_prompt = kokoro_phonemize_multilingual(normalized, phmzr, config.language, config.zh_dict_dir);
    } else {
        phonemized_prompt = phmzr->text_to_phonemes(normalized);
    }

  	const int64_t t_after_phonemize_us = timings ? tts_time_us() : 0;

    double tokenize_ms = 0.0;
    double run_ms = 0.0;
    size_t n_tokens_total = 0;

   	// Kokoro users a utf-8 single character tokenizer so if the size of the prompt is smaller than the max context length without the
	// beginning of sentence and end of sentence tokens then we can compute it all at once.
    const size_t max_no_special = model->max_context_length > 2 ? (size_t) model->max_context_length - 2 : 0;
    if (tts_utf8_codepoint_count(phonemized_prompt) <= max_no_special) {
   		// 说明：
   		// - Kokoro 会把 ".!?" 视为 EOS（句末）信号；若直接喂给模型，可能导致提前结束。
   		// - 但“直接删除”会丢失句间停顿，中文听感容易“一口气读完”。
   		// 因此这里把 ".!?" 替换为 ","（更弱的停顿），既避免 EOS，又尽量保留断句节奏。
   		phonemized_prompt = strip(replace_any(phonemized_prompt, ".!?", ","));
   		if (phonemized_prompt.empty()) {
   			return;
   		}
        const int64_t t_before_tokenize_us = timings ? tts_time_us() : 0;
		std::vector<uint32_t> tokens;
		tokens.push_back(model->bos_token_id);
		tokenizer->tokenize(phonemized_prompt, tokens);
		tokens.push_back(model->eos_token_id);
        const int64_t t_after_tokenize_us = timings ? tts_time_us() : 0;
        tokenize_ms = us_to_ms(t_after_tokenize_us - t_before_tokenize_us);
        n_tokens_total = tokens.size();

		kokoro_ubatch batch;
		batch.n_tokens = tokens.size();
		batch.input_tokens = tokens.data();
        const int64_t t_before_run_us = timings ? tts_time_us() : 0;
		run(batch, response);
        const int64_t t_after_run_us = timings ? tts_time_us() : 0;
        run_ms = us_to_ms(t_after_run_us - t_before_run_us);
   	} else {
   		// TODO: determine the performance to memory trade off in using a batched compute approach verse this chunking approach.
   		// This approach is likely to be slower than a batched approach, but given the already huge memory overhead of Kokoro's graph it
   		// might be preferable to use this chunking approach.
   		std::vector<std::string> clauses = split(phonemized_prompt, ".!?");
        // 说明：chunking 模式下 ".!?" 已被用于切句，因此不再喂给模型；这里用 ',' 作为弱停顿，
        // 让多句长文本在拼接时更像自然断句（避免整段听起来“粘在一起”）。
        std::vector<std::string> normalized_clauses;
        normalized_clauses.reserve(clauses.size());
        for (auto & clause : clauses) {
            clause = strip(clause);
            if (!clause.empty()) {
                normalized_clauses.push_back(std::move(clause));
            }
        }
        for (size_t ci = 0; ci + 1 < normalized_clauses.size(); ++ci) {
            normalized_clauses[ci].push_back(',');
        }
        const int64_t t_before_tokenize_us = timings ? tts_time_us() : 0;
        const auto chunks = tokenize_chunks(std::move(normalized_clauses));
        const int64_t t_after_tokenize_us = timings ? tts_time_us() : 0;
        tokenize_ms = us_to_ms(t_after_tokenize_us - t_before_tokenize_us);
        for (const auto & t : chunks) {
            n_tokens_total += t.size();
        }

        const int64_t t_before_run_us = timings ? tts_time_us() : 0;
  		for (const auto & tokens : chunks) {
			kokoro_ubatch batch;
			batch.n_tokens = tokens.size();
			batch.input_tokens = (uint32_t *) tokens.data();
			tts_response partial{};
			run(batch, partial);
			append_to_response(response, partial);
		}
        const int64_t t_after_run_us = timings ? tts_time_us() : 0;
        run_ms = us_to_ms(t_after_run_us - t_before_run_us);
  	}

    if (timings) {
        const int64_t t_end_us = tts_time_us();
        const char * const mode = use_multilingual ? "multilingual" : "tts_phonemizer";
        const char * lang = "zh";
        switch (config.language) {
            case tts_language::ZH: lang = "zh"; break;
            case tts_language::EN: lang = "en"; break;
            case tts_language::JA: lang = "ja"; break;
        }
        fprintf(stderr,
                "[kokoro][timings] frontend: normalize=%.2fms phonemize=%.2fms tokenize=%.2fms run=%.2fms total=%.2fms (mode=%s lang=%s cjk=%d prompt_bytes=%zu phoneme_bytes=%zu tokens=%zu)\n",
                us_to_ms(t_after_normalize_us - t_start_us),
                us_to_ms(t_after_phonemize_us - t_after_normalize_us),
                tokenize_ms,
                run_ms,
                us_to_ms(t_end_us - t_start_us),
                mode,
                lang,
                contains_cjk ? 1 : 0,
                std::strlen(prompt),
                phonemized_prompt.size(),
                n_tokens_total);
    }
}

std::vector<std::string_view> kokoro_runner::list_voices() {
    std::vector<std::string_view> voices;
    voices.reserve(model->voices.size());
    for (const auto & kv : model->voices) {
        voices.emplace_back(kv.first);
    }
    return voices;
}

struct kokoro_duration_context * build_new_duration_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_duration_context * kctx = new kokoro_duration_context(model, n_threads);
    if (!use_cpu) {
        kctx->backend = tts_backend_init_accel();
    }
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_duration_nodes()*5 + ggml_graph_overhead_custom(model->max_duration_nodes()*5, false));
    return kctx;
}


struct kokoro_context * build_new_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_context * kctx = new kokoro_context(model, n_threads);
    if (!use_cpu) {
        kctx->backend = tts_backend_init_accel();
    }
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_gen_nodes()*30 + ggml_graph_overhead_custom(model->max_gen_nodes()*30, false));
    return kctx;
}
