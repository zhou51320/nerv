#include "util.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdarg.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#elif __linux__
#include <unistd.h>
#else
// windows stuff
#endif

namespace {
    // 当前线程的后端配置。
    //
    // 说明：
    // - 选择做成 thread_local，是为了在 tts-server 等多线程场景下，避免并发加载模型时互相覆盖配置。
    // - 默认使用 AUTO：如果上层只表达“我要用 GPU”，则尽量自动选择可用的 GPU 后端（Metal 优先，其次 Vulkan）。
    thread_local tts_backend_config g_backend_cfg{tts_compute_backend::AUTO, 0};

    ggml_backend_t tts_try_init_metal() {
#ifdef GGML_USE_METAL
        return ggml_backend_metal_init();
#else
        return nullptr;
#endif
    }

    ggml_backend_t tts_try_init_vulkan(const int device) {
#ifdef GGML_USE_VULKAN
        const size_t dev = device >= 0 ? (size_t) device : 0;
        return ggml_backend_vk_init(dev);
#else
        (void) device;
        return nullptr;
#endif
    }
}  // namespace

void tts_set_backend_config(const tts_backend_config & cfg) {
    g_backend_cfg = cfg;
}

tts_backend_config tts_get_backend_config() {
    return g_backend_cfg;
}

tts_backend_config_guard::tts_backend_config_guard(const tts_backend_config & cfg) : prev_{tts_get_backend_config()} {
    tts_set_backend_config(cfg);
}

tts_backend_config_guard::~tts_backend_config_guard() {
    tts_set_backend_config(prev_);
}

ggml_backend_t tts_backend_init_accel() {
    const tts_backend_config cfg = tts_get_backend_config();

    switch (cfg.backend) {
        case tts_compute_backend::CPU:
            return nullptr;
        case tts_compute_backend::METAL:
            return tts_try_init_metal();
        case tts_compute_backend::VULKAN:
            return tts_try_init_vulkan(cfg.device);
        case tts_compute_backend::AUTO: {
            if (ggml_backend_t backend = tts_try_init_metal()) {
                return backend;
            }
            if (ggml_backend_t backend = tts_try_init_vulkan(cfg.device)) {
                return backend;
            }
            return nullptr;
        }
    }

    return nullptr;
}

void tts_time_init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_time_init();
    });
}

void tts_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}

// Simple helper function for getting layer count from tensor name
std::pair<int, std::string> parse_layer_count(std::string name, int skip) {
    bool found = false;
    bool after_layer = false;
    std::string digit_chars = "";
    std::string after_layer_name = "";
    int count = 0;
    for (char& c : name) {
        if (count < skip) {
            count += 1;
            continue;
        }
        count += 1;
        if (after_layer) {
            after_layer_name += c;
        } else if (std::isdigit(c)) {
            found = true;
            digit_chars += c;
        } else if (!found) {
            
        } else {
            after_layer = true;
            after_layer_name += c;
        }
    }
    if (digit_chars.size() == 0) {
        return std::make_pair(-1, name);
    }
    return std::make_pair(std::stoi(digit_chars), after_layer_name);
}

int search_for_gguf_keys(gguf_context * meta, std::vector<std::string> possible_keys) {
    int gguf_key = -1;
    for (auto key : possible_keys) {
        gguf_key = gguf_find_key(meta, key.c_str());
        if (gguf_key != -1) {
            return gguf_key;
        }
    }
    return gguf_key;
}

void random_uniform_gen(int count, float * tgt, float min, float max) {
    // 说明（性能相关）：Kokoro 的 uv/noise 输入在每次推理前都会生成大量随机数。
    // std::uniform_real_distribution + default_random_engine 在这里会带来明显的 CPU 开销，
    // 同时原实现还存在一个隐患：distribution 是 static 的，后续传入不同的 min/max 并不会生效。
    //
    // 这里使用 thread_local 的 xorshift64* 生成器，配合位级转换快速得到 [0, 1) 的 float，
    // 再线性缩放到 [min, max)。这样在不改变“均匀分布”性质的前提下，显著降低前处理耗时。

    if (count <= 0 || tgt == nullptr) {
        return;
    }

    const float range = max - min;

    auto uniform01 = [](uint32_t x) -> float {
        // 取 23bit 填充 float 的尾数，构造 [1, 2) 的 float，再减 1 得到 [0, 1)。
        const uint32_t bits = (x >> 9) | 0x3F800000u;
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f - 1.0f;
    };

    auto next_u32 = []() -> uint32_t {
        // xorshift64*，每线程独立状态；避免加锁也便于未来并行调用。
        static thread_local uint64_t s = [] {
            std::random_device rd;
            uint64_t seed = (uint64_t(rd()) << 32) ^ uint64_t(rd());
            // 避免 0 种子导致退化。
            return seed ? seed : 0x9E3779B97F4A7C15ull;
        }();

        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        const uint64_t v = s * 2685821657736338717ull;
        return uint32_t(v >> 32);
    };

    for (int i = 0; i < count; i++) {
        const float u = uniform01(next_u32());
        tgt[i] = min + range * u;
    }
}

void random_normal_gen(int count, float * tgt, float mean, float std) {
    static std::default_random_engine e;
    static std::normal_distribution<float> dis(mean, std);
    for (int i = 0; i < count; i++) {
        tgt[i] = dis(e);
    }
}

float round_to_float(double v) {
    return roundf(v * powl(10, 6)) / powl(10, 6);
}

// 说明：生成常量输入（如用于 Vulkan 图中的加/减常数），避免直接使用 host 指针。
static ggml_tensor * tts_const_tensor(ggml_context * ctx, const float * v,
                                      std::vector<tts_graph_const_input> * const_inputs) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    if (const_inputs != nullptr) {
        ggml_set_input(t);
        const_inputs->push_back({t, *v});
    } else {
        t->data = const_cast<float *>(v);
    }
    return t;
}

struct ggml_tensor * reciprocal(ggml_context * ctx, struct ggml_tensor * x,
                                std::vector<tts_graph_const_input> * const_inputs) {
    TTS_ASSERT(x->ne[0] == 1);
    static constexpr float one = 1.0f;
    // 说明：
    // - 旧实现为了得到 "1/x" 会先构造一个与 x 同形状的全 1 张量再做除法。
    // - 在 Vulkan 下用 ggml_repeat 去广播标量常量，有概率触发后端对标量 repeat 的边界问题（表现为输出出现明显的“金属音”失真）。
    // - 这里改为用标准算子生成同形状的全 1：ones = x*0 + 1，再计算 ones/x。
    //   这样避免了标量 repeat，同时仍保持整段计算可被 Vulkan 图执行。
    ggml_tensor * one_t = tts_const_tensor(ctx, &one, const_inputs); // 标量 1
    ggml_tensor * zeros = ggml_scale(ctx, x, 0.0f);                 // 与 x 同形状的全 0
    ggml_tensor * ones  = ggml_add(ctx, zeros, one_t);              // 与 x 同形状的全 1（标量自动广播）
    return ggml_div(ctx, ones, x);
}

// Described in https://arxiv.org/abs/2006.08195
// Snake1d is a common tunable activation function used in the DAC model.
struct ggml_tensor * snake_1d(ggml_context * ctx, struct ggml_tensor * alpha, struct ggml_tensor * a,
                              std::vector<tts_graph_const_input> * const_inputs) {
    assert(a->ne[2] == 1 && a->ne[3] == 1);
    // 说明：
    // Snake1d(a, alpha) = a + (sin(alpha * a)^2) / alpha
    //
    // 这里故意使用 “/ alpha” 而不是先算 reciprocal(alpha) 再乘：
    // - 可避免 Vulkan 路径下对标量常量的 repeat 广播（见 reciprocal() 的说明）；
    // - 算子更少（少一次 reciprocal 分支），在 Vulkan 下也更稳定。
    (void) const_inputs; // 当前实现不再需要通过 const_inputs 注入常量
    ggml_tensor * sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, a, alpha)));
    ggml_tensor * term   = ggml_div(ctx, sin_sq, alpha);
    return ggml_add(ctx, a, term);
}

namespace {
// ----------------------------- 说明（重要） -----------------------------
//
// ggml 0.9.4 的 `ggml_conv_transpose_1d` 当前实现存在两个限制：
// 1) padding 必须为 0
// 2) dilation 必须为 1
//
// 但本项目（尤其是 Kokoro / DAC）历史代码中，存在：
// - padding != 0 的转置卷积（非常常见：用于上采样时对齐输出长度）
// - 带 output_padding / groups 的旧扩展签名（Kokoro 的 pool 是 depthwise 转置卷积）
//
// 为了“侵入性更小”（不去修改 ggml 源码），这里采用兼容封装：
// - groups == 1：调用 ggml 原生转置卷积（固定 padding=0）得到更长输出，再通过 view 裁剪实现 padding/output_padding。
// - groups  > 1：使用 GGML_OP_CUSTOM 在项目侧实现一个 CPU 版本的 grouped 转置卷积，补齐 depthwise 场景。
//
// 注意：custom 路径目前仅实现为 CPU 计算（与本项目 STFT/ISTFT 自定义算子一致）。
struct tts_conv_transpose_1d_op_params {
    ggml_custom_op_params base;
    int32_t s0;
    int32_t p0;
    int32_t d0;
    int32_t output_padding;
    int32_t groups;
};

static inline float tts_read_kernel_val_3d_f32(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
    return *(const float *) ((const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]);
}

static inline float tts_read_kernel_val_3d_f16(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
    const ggml_fp16_t v = *(const ggml_fp16_t *) ((const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2]);
    return ggml_fp16_to_fp32(v);
}

static void tts_compute_conv_transpose_1d_custom(struct ggml_tensor * dst, int ith, int nth, void *) {
    tts_conv_transpose_1d_op_params p{};
    std::memcpy(&p, dst->op_params, sizeof(p));

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel != nullptr);
    GGML_ASSERT(input  != nullptr);

    // 目前只为项目中实际用到的场景兜底：输入为 F32；kernel 为 F16/F32；输出为 F32。
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(input->nb[0] == sizeof(float));

    const int32_t s0 = p.s0;
    const int32_t p0 = p.p0;
    const int32_t d0 = p.d0;
    const int32_t output_padding = p.output_padding;
    const int32_t groups = p.groups;

    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(d0 > 0);
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(output_padding >= 0);
    GGML_ASSERT(groups > 0);

    // 约定 kernel layout 与 ggml 的 conv_transpose_1d 一致：
    // kernel: [K, OC_per_group, IC_total]
    // input : [L_in, IC_total, N]
    // output: [L_out, OC_total, N]，其中 OC_total = OC_per_group * groups
    const int64_t K = kernel->ne[0];
    const int64_t OC_per_group = kernel->ne[1];
    const int64_t IC_total = kernel->ne[2];

    const int64_t L_in  = input->ne[0];
    const int64_t IC_in = input->ne[1];
    const int64_t N     = input->ne[2];

    GGML_ASSERT(kernel->ne[3] == 1);
    GGML_ASSERT(input->ne[3] == 1);
    GGML_ASSERT(IC_in == IC_total);
    GGML_ASSERT(IC_total % groups == 0);
    GGML_ASSERT(OC_per_group > 0);

    const int64_t IC_per_group = IC_total / groups;
    const int64_t OC_total = OC_per_group * groups;

    const int64_t L_out_expected = (L_in - 1) * (int64_t) s0 - 2 * (int64_t) p0 + (int64_t) d0 * (K - 1) + (int64_t) output_padding + 1;
    GGML_ASSERT(dst->ne[0] == L_out_expected);
    GGML_ASSERT(dst->ne[1] == OC_total);
    GGML_ASSERT(dst->ne[2] == N);
    GGML_ASSERT(dst->ne[3] == 1);

    // 任务划分：按输出通道（ne1）切片，避免写冲突。
    const int64_t ocpt = (OC_total + nth - 1) / nth; // out channels per task
    const int64_t oc0  = ocpt * ith;
    const int64_t oc1  = std::min(oc0 + ocpt, OC_total);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc_total = oc0; oc_total < oc1; ++oc_total) {
            float * out_row = (float *) ((char *) dst->data + oc_total * dst->nb[1] + n * dst->nb[2]);
            std::memset(out_row, 0, (size_t) dst->ne[0] * sizeof(float));

            const int64_t g = oc_total / OC_per_group;
            const int64_t oc = oc_total - g * OC_per_group; // oc in group

            const int64_t ic_begin = g * IC_per_group;
            const int64_t ic_end   = ic_begin + IC_per_group;

            for (int64_t ic = ic_begin; ic < ic_end; ++ic) {
                const float * in_row = (const float *) ((const char *) input->data + ic * input->nb[1] + n * input->nb[2]);

                for (int64_t x = 0; x < L_in; ++x) {
                    const float xv = in_row[x];
                    if (xv == 0.0f) {
                        continue;
                    }

                    const int64_t base_y = x * (int64_t) s0 - (int64_t) p0;

                    for (int64_t k = 0; k < K; ++k) {
                        const int64_t y = base_y + k * (int64_t) d0;
                        if (y < 0 || y >= dst->ne[0]) {
                            continue;
                        }

                        const float w = (kernel->type == GGML_TYPE_F32)
                                            ? tts_read_kernel_val_3d_f32(kernel, k, oc, ic)
                                            : tts_read_kernel_val_3d_f16(kernel, k, oc, ic);
                        out_row[y] += xv * w;
                    }
                }
            }
        }
    }
}
} // namespace

bool has_suffix(std::string value, std::string suffix) {
    return value.size() >= suffix.size() && value.compare(value.size()-suffix.size(), suffix.size(), suffix) == 0;
}

bool has_prefix(std::string value, std::string prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

namespace {
// ----------------------------- 说明（重要） -----------------------------
//
// 现状：
// - 旧版 ggml 曾内置 ggml_stft / ggml_istft 以及对应 CPU kernel（用于 Kokoro 的声码器部分）。
// - 你升级到 ggml 0.9.4 后，这组算子已不再存在，因此项目会直接编译失败。
//
// 方案（侵入性更小）：
// - 不去改 ggml 源码；
// - 在本项目侧用 ggml 的 GGML_OP_CUSTOM 自定义算子实现 STFT / ISTFT，
//   让 Kokoro 仍能在 ggml 0.9.4 上工作。
//
// 注意：
// - 该实现目前只提供 CPU 计算逻辑（自定义算子由 ggml-cpu 调用回调函数执行）。
// - 若上层启用多后端调度（如 Metal + CPU），scheduler 会在需要时对 window 等权重做拷贝，
//   保证自定义算子在 CPU 上可读写数据。

// 2*pi 常量（避免依赖 M_PI 在不同平台的宏开关）。
static constexpr float TTS_TWO_PI_F = 6.28318530717958647692f;

// 这是一个简单的 O(N^2) DFT 实现：当 n_fft 不是 2 的幂时，用它作为回退。
static void tts_simple_dft(float * real, float * imag, float * scratch, size_t n_fft, size_t step) {
    const float base_k = -TTS_TWO_PI_F / (float) n_fft;

    for (size_t i = 0; i < n_fft; ++i) {
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (size_t j = 0; j < n_fft; ++j) {
            const float k = base_k * (float) j * (float) i;
            const float c = cosf(k);
            const float s = sinf(k);

            const float xr = real[j * step];
            const float xi = imag[j * step];

            // (xr + i*xi) * (c + i*s)
            acc_r += xr * c - xi * s;
            acc_i += xr * s + xi * c;
        }

        scratch[i * 2 + 0] = acc_r;
        scratch[i * 2 + 1] = acc_i;
    }

    // 写回输出（实部/虚部分离存储）
    for (size_t i = 0; i < n_fft; ++i) {
        real[i * step] = scratch[i * 2 + 0];
        imag[i * step] = scratch[i * 2 + 1];
    }
}

// ----------------------------- FFT（CPU 热点） -----------------------------
// 说明：
// - Kokoro 的声码器需要做 STFT/ISTFT，推理时会反复调用 FFT。
// - 旧版递归 FFT 每个 butterfly 都会调用一次 cosf/sinf（twiddle），CPU 上 trig 成本非常高，
//   这会显著拖慢纯 CPU 推理速度。
// - 这里对“2 的幂长度 FFT”预计算 twiddle 表（cos/sin），推理时只查表不再计算 trig，
//   在不引入第三方库（如 FFTW）的前提下获得明显提速。

// 若 n 是 2 的幂，返回 log2(n)，否则返回 -1。
static int tts_log2_exact_pow2(size_t n) {
    if (n == 0) {
        return -1;
    }
    if ((n & (n - 1)) != 0) {
        return -1;
    }
    int e = 0;
    while (n > 1) {
        n >>= 1;
        ++e;
    }
    return e;
}

struct tts_fft_twiddles {
    // 当前缓存对应的 FFT 长度（必须是 2 的幂）
    size_t n_fft = 0;
    int    log2_n = 0;

    // twiddle 表：level 表示 FFT 子问题规模 n=2^level
    // - cos[level][i] = cos(-2π*i/n)
    // - sin[level][i] = sin(-2π*i/n)
    std::vector<std::vector<float>> cos;
    std::vector<std::vector<float>> sin;
};

static const tts_fft_twiddles & tts_get_fft_twiddles(size_t n_fft) {
    static thread_local tts_fft_twiddles cache{};
    if (cache.n_fft == n_fft) {
        return cache;
    }

    const int log2_n = tts_log2_exact_pow2(n_fft);
    if (log2_n < 0) {
        // 非 2 的幂：不构建 twiddle 表（调用方会走慢路径）
        cache = {};
        cache.n_fft  = n_fft;
        cache.log2_n = -1;
        return cache;
    }

    cache = {};
    cache.n_fft  = n_fft;
    cache.log2_n = log2_n;
    cache.cos.resize((size_t) log2_n + 1);
    cache.sin.resize((size_t) log2_n + 1);

    // level=1 => n=2，...，level=log2_n => n=n_fft
    for (int level = 1; level <= log2_n; ++level) {
        const size_t n = (size_t) 1u << (unsigned) level;
        const size_t half = n / 2;

        cache.cos[level].resize(half);
        cache.sin[level].resize(half);

        const float km = -TTS_TWO_PI_F / (float) n;
        for (size_t i = 0; i < half; ++i) {
            const float k = km * (float) i;
            cache.cos[level][i] = cosf(k);
            cache.sin[level][i] = sinf(k);
        }
    }

    return cache;
}

// 慢路径：递归 Radix-2 FFT（会调用 trig），用于非 2 的幂长度（或极端兼容场景）。
static void tts_radix2_fft_slow(float * real, float * imag, float * scratch, size_t n_fft, size_t step) {
    if (n_fft == 1) {
        return;
    }

    if (n_fft % 2 != 0) {
        // 当长度无法被 2 因子分解时，回退到 O(N^2) 的简单 DFT。
        tts_simple_dft(real, imag, scratch, n_fft, step);
        return;
    }

    // 递归：偶/奇拆分（通过 step*2 达到“跳步访问”的效果，不额外分配数组）
    tts_radix2_fft_slow(real, imag, scratch, n_fft / 2, step * 2);
    tts_radix2_fft_slow(
        (float *) ((char *) real + step * sizeof(float)),
        (float *) ((char *) imag + step * sizeof(float)),
        scratch,
        n_fft / 2,
        step * 2);

    const float km = -TTS_TWO_PI_F / (float) n_fft;

    // butterfly
    for (size_t i = 0; 2 * i < n_fft; ++i) {
        const float k  = km * (float) i;
        const float c  = cosf(k);
        const float s  = sinf(k);

        const float pr = real[i * 2 * step];
        const float pi = imag[i * 2 * step];

        const float qr0 = real[(i * 2 + 1) * step];
        const float qi0 = imag[(i * 2 + 1) * step];

        // (qr0 + i*qi0) * (c + i*s)
        const float qr = qr0 * c - qi0 * s;
        const float qi = qr0 * s + qi0 * c;

        // 结果先写到 scratch，再整体写回（避免覆盖尚未读取的数据）
        scratch[i + n_fft] = pi + qi;
        scratch[i]         = pr + qr;

        scratch[i + (n_fft / 2) + n_fft] = pi - qi;
        scratch[i + (n_fft / 2)]         = pr - qr;
    }

    for (size_t i = 0; i < n_fft; ++i) {
        real[i * step] = scratch[i];
        imag[i * step] = scratch[i + n_fft];
    }
}

// 快路径：仅用于 n_fft 是 2 的幂（level=log2(n_fft)）。
static void tts_radix2_fft_twiddled(
    float * real,
    float * imag,
    float * scratch,
    size_t  step,
    int     level,
    const tts_fft_twiddles & tw) {
    if (level <= 0) {
        return;
    }
    const size_t n = (size_t) 1u << (unsigned) level;

    // 递归：偶/奇拆分（通过 step*2 达到“跳步访问”的效果，不额外分配数组）
    tts_radix2_fft_twiddled(real, imag, scratch, step * 2, level - 1, tw);
    tts_radix2_fft_twiddled(
        (float *) ((char *) real + step * sizeof(float)),
        (float *) ((char *) imag + step * sizeof(float)),
        scratch,
        step * 2,
        level - 1,
        tw);

    const float * cos_tbl = tw.cos[level].data();
    const float * sin_tbl = tw.sin[level].data();

    // butterfly（查表代替 trig）
    for (size_t i = 0; i < n / 2; ++i) {
        const float c  = cos_tbl[i];
        const float s  = sin_tbl[i];

        const float pr = real[i * 2 * step];
        const float pi = imag[i * 2 * step];

        const float qr0 = real[(i * 2 + 1) * step];
        const float qi0 = imag[(i * 2 + 1) * step];

        // (qr0 + i*qi0) * (c + i*s)
        const float qr = qr0 * c - qi0 * s;
        const float qi = qr0 * s + qi0 * c;

        scratch[i + n] = pi + qi;
        scratch[i]     = pr + qr;

        scratch[i + (n / 2) + n] = pi - qi;
        scratch[i + (n / 2)]     = pr - qr;
    }

    for (size_t i = 0; i < n; ++i) {
        real[i * step] = scratch[i];
        imag[i * step] = scratch[i + n];
    }
}

// Radix-2 FFT（Cooley-Tukey）：输入为 real/imag 两个数组。
static void tts_radix2_fft(float * real, float * imag, float * scratch, size_t n_fft, size_t step) {
    if (n_fft == 1) {
        return;
    }

    // 非偶数长度：无法继续二分，回退到 DFT
    if (n_fft % 2 != 0) {
        tts_simple_dft(real, imag, scratch, n_fft, step);
        return;
    }

    const int level = tts_log2_exact_pow2(n_fft);
    if (level >= 0) {
        const tts_fft_twiddles & tw = tts_get_fft_twiddles(n_fft);
        if (tw.log2_n == level) {
            tts_radix2_fft_twiddled(real, imag, scratch, step, level, tw);
            return;
        }
    }

    // 非 2 的幂但仍可继续二分：走慢路径（内部同样会在必要时回退 DFT）。
    tts_radix2_fft_slow(real, imag, scratch, n_fft, step);
}

// STFT 自定义算子的参数（存放在 dst->op_params 里，避免额外堆分配/生命周期管理）。
struct tts_stft_op_params {
    ggml_custom_op_params base;
    int32_t n_fft;
    int32_t hop;
    int32_t abs_and_angle; // 0/1：输出为 (real,imag) 或 (magnitude,angle)
    int32_t one_sided;     // 0/1：是否输出 one-sided（ne0 = n_fft/2+1）
};

// ISTFT 自定义算子的参数（同上）。
struct tts_istft_op_params {
    ggml_custom_op_params base;
    int32_t n_fft;
    int32_t hop;
    int32_t from_abs_and_angle; // 0/1：输入为 (real,imag) 或 (magnitude,angle)
    int32_t one_sided;          // 0/1：输入是否为 one-sided（ne0 = n_fft/2+1）
};

static void tts_compute_stft_custom(struct ggml_tensor * dst, int ith, int nth, void *) {
    tts_stft_op_params p{};
    std::memcpy(&p, dst->op_params, sizeof(p));

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * win  = dst->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(win  != nullptr);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(win->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_fft = p.n_fft;
    const int32_t hop   = p.hop;
    const int32_t half  = n_fft / 2;
    const bool abs_and_angle = p.abs_and_angle != 0;
    const bool one_sided = p.one_sided != 0;

    GGML_ASSERT(n_fft > 0);
    GGML_ASSERT(hop > 0);
    GGML_ASSERT(win->ne[0] == n_fft);
    // 说明：默认 Kokoro 使用 one-sided（只保留 [0, n_fft/2] 频率区间），可以少一半输出元素，
    // 同时避免“view + cont”的额外拷贝。
    const int32_t out_ne0 = one_sided ? (half + 1) : n_fft;
    GGML_ASSERT(dst->ne[0] == out_ne0);
    GGML_ASSERT(dst->ne[3] == 2);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(win->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[0]  == sizeof(float));

    const int64_t in_len  = src0->ne[0];
    const int64_t n_batch = dst->ne[2];
    const int64_t n_frame = dst->ne[1];

    // 每个任务负责一段 frame 区间，写入互不重叠，天然无竞争。
    const int64_t fpt = (n_frame + nth - 1) / nth;  // frames per task
    const int64_t f0  = fpt * ith;
    const int64_t f1  = std::min(f0 + fpt, n_frame);

    // 每个线程复用两块 buffer：
    // - fft_buf：长度 2*n_fft（前半 real，后半 imag），作为 FFT 的工作区；
    // - scratch：长度 2*n_fft，作为 radix2 过程的临时存储。
    // 说明：过去实现直接在 dst 输出缓冲上就地 FFT，但当输出改为 one-sided 后 dst->ne0 < n_fft，
    // 无法直接作为 FFT 工作区，因此改为线程本地缓冲。
    static thread_local std::vector<float> tls_fft_buf;
    static thread_local std::vector<float> tls_scratch;
    if (tls_fft_buf.size() < (size_t) n_fft * 2) {
        tls_fft_buf.resize((size_t) n_fft * 2);
    }
    if (tls_scratch.size() < (size_t) n_fft * 2) {
        tls_scratch.resize((size_t) n_fft * 2);
    }
    float * fft_r  = tls_fft_buf.data();
    float * fft_i  = tls_fft_buf.data() + n_fft;
    float * scratch = tls_scratch.data();

    const float * w = (const float *) win->data;

    for (int64_t b = 0; b < n_batch; ++b) {
        const char * src0_b = (const char *) src0->data + b * src0->nb[1];
        const float * src = (const float *) src0_b;

        for (int64_t fi = f0; fi < f1; ++fi) {
            const int64_t center = fi * (int64_t) hop;

            float * out_r = (float *) ((char *) dst->data + fi * dst->nb[1] + b * dst->nb[2]);
            float * out_i = (float *) ((char *) dst->data + fi * dst->nb[1] + b * dst->nb[2] + dst->nb[3]);

            // 1) 取窗口并做“中心对齐”的反射 padding（与旧 ggml 行为一致）
            // 2) 预先乘上 window（Hann 等）
            for (int32_t i = 0; i < n_fft; ++i) {
                const int64_t ai = center - half + i;
                float sample = 0.0f;
                if (ai < 0) {
                    sample = src[(size_t) (-ai)];
                } else if (ai >= in_len) {
                    const int64_t ri = in_len - (ai - in_len + 1);
                    sample = src[(size_t) ri];
                } else {
                    sample = src[(size_t) ai];
                }

                fft_r[i] = sample * w[i];
                fft_i[i] = 0.0f;
            }

            // 3) FFT（就地变换 fft_r/fft_i）
            tts_radix2_fft(fft_r, fft_i, scratch, (size_t) n_fft, 1);

            // 4) 可选：转换为 (magnitude, angle)
            if (abs_and_angle) {
                for (int32_t i = 0; i < out_ne0; ++i) {
                    const float r  = fft_r[i];
                    const float im = fft_i[i];
                    out_r[i] = sqrtf(r * r + im * im);
                    out_i[i] = atan2f(im, r);
                }
            } else {
                // 非 abs/angle：直接输出 real/imag
                std::memcpy(out_r, fft_r, (size_t) out_ne0 * sizeof(float));
                std::memcpy(out_i, fft_i, (size_t) out_ne0 * sizeof(float));
            }
        }
    }
}

static void tts_compute_istft_custom(struct ggml_tensor * dst, int ith, int nth, void *) {
    tts_istft_op_params p{};
    std::memcpy(&p, dst->op_params, sizeof(p));

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * win  = dst->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(win  != nullptr);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(win->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_fft = p.n_fft;
    const int32_t hop   = p.hop;
    const int32_t half  = n_fft / 2;
    const bool from_abs_and_angle = p.from_abs_and_angle != 0;
    const bool onesided_param = p.one_sided != 0;

    GGML_ASSERT(n_fft > 0);
    GGML_ASSERT(hop > 0);
    GGML_ASSERT(win->ne[0] == n_fft);
    GGML_ASSERT(src0->ne[3] == 2);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(win->nb[0] == sizeof(float));

    const int64_t n_frame = src0->ne[1];
    const int64_t n_batch = src0->ne[2];
    const int64_t out_len = dst->ne[0];

    // onesided：输入频率维只有 half+1，需要按厄米对称重建完整频谱
    const bool onesided = onesided_param;
    if (onesided) {
        GGML_ASSERT(src0->ne[0] == (int64_t) half + 1);
    } else {
        GGML_ASSERT(src0->ne[0] == (int64_t) n_fft);
    }

    // 将输出长度按任务划分，每个任务只写自己负责的区间，避免线程间原子/锁/栅栏。
    const int64_t spt = (out_len + nth - 1) / nth;  // samples per task
    const int64_t t0  = spt * ith;
    const int64_t t1  = std::min(t0 + spt, out_len);

    // 先把自己负责的输出区间清零（每个 batch 独立一段）
    for (int64_t b = 0; b < n_batch; ++b) {
        float * out = (float *) ((char *) dst->data + b * dst->nb[1]);
        for (int64_t t = t0; t < t1; ++t) {
            out[t] = 0.0f;
        }
    }

    // FFT scratch：buffer 用于 radix2 过程，spec 用于重建完整复数频谱（real + imag）
    static thread_local std::vector<float> tls_buffer;
    static thread_local std::vector<float> tls_spec;
    if (tls_buffer.size() < (size_t) n_fft * 2) {
        tls_buffer.resize((size_t) n_fft * 2);
    }
    if (tls_spec.size() < (size_t) n_fft * 2) {
        tls_spec.resize((size_t) n_fft * 2);
    }
    float * buffer   = tls_buffer.data();
    float * spec_r   = tls_spec.data();
    float * spec_i   = tls_spec.data() + n_fft;

    const float * w = (const float *) win->data;
    const float inv_n_fft = 1.0f / (float) n_fft;

    // 为了保证每个输出采样点拿到所有会影响它的 frame，这里按旧 ggml 的经验公式取一个“扩展 frame 范围”。
    const int poa = (half / hop) - 1;
    const int pob = (half / hop) + 1;

    int ir0 = (ith == 0) ? 0 : (int) (t0 / hop) - poa;
    if (ir0 < 0) {
        ir0 = 0;
    }
    int ir1 = (int) std::min<int64_t>((t1 / hop) + pob, n_frame);
    if (ir1 < 0) {
        ir1 = 0;
    }

    for (int64_t b = 0; b < n_batch; ++b) {
        float * out = (float *) ((char *) dst->data + b * dst->nb[1]);

        for (int i1 = ir0; i1 < ir1; ++i1) {
            const float * src_r = (const float *) ((const char *) src0->data + (int64_t) i1 * src0->nb[1] + b * src0->nb[2]);
            const float * src_i = (const float *) ((const char *) src0->data + (int64_t) i1 * src0->nb[1] + b * src0->nb[2] + src0->nb[3]);

            // 重建完整频谱（长度 n_fft）
            if (onesided) {
                // 说明：one-sided 情况只需要计算 [0..half]，其余频点通过厄米对称补齐，
                // 可以把 cosf/sinf 的调用量减少约一半。
                if (from_abs_and_angle) {
                    for (int32_t idx = 0; idx <= half; ++idx) {
                        const float mag   = src_r[idx];
                        const float phase = src_i[idx];
                        const float c = cosf(phase);
                        const float s = sinf(phase);
                        spec_r[idx] = mag * c;
                        spec_i[idx] = mag * s;
                    }
                } else {
                    // real/imag 输入：直接拷贝 one-sided 区间
                    std::memcpy(spec_r, src_r, (size_t) (half + 1) * sizeof(float));
                    std::memcpy(spec_i, src_i, (size_t) (half + 1) * sizeof(float));
                }

                // 补齐镜像频点：X[n-k] = conj(X[k])
                for (int32_t idx = 1; idx < half; ++idx) {
                    spec_r[n_fft - idx] = spec_r[idx];
                    spec_i[n_fft - idx] = -spec_i[idx];
                }
            } else {
                if (from_abs_and_angle) {
                    for (int32_t i = 0; i < n_fft; ++i) {
                        const float mag   = src_r[i];
                        const float phase = src_i[i];
                        spec_r[i] = mag * cosf(phase);
                        spec_i[i] = mag * sinf(phase);
                    }
                } else {
                    std::memcpy(spec_r, src_r, (size_t) n_fft * sizeof(float));
                    std::memcpy(spec_i, src_i, (size_t) n_fft * sizeof(float));
                }
            }

            // IFFT：利用“FFT(·) + 反向索引”的等价关系得到时域信号，再乘 window 并叠加到目标序列
            tts_radix2_fft(spec_r, spec_i, buffer, (size_t) n_fft, 1);

            const int64_t center = (int64_t) i1 * hop;
            for (int32_t i = 0; i < n_fft; ++i) {
                const int32_t base_index = (n_fft - i) % n_fft;       // 对应时域采样位置（0..n_fft-1）
                const int64_t location   = center + (base_index - half);

                if (location < t0 || location >= t1) {
                    continue;
                }
                // 归一化 1/n_fft，并在此处乘上 window（后续会再除以 window_squared_sum 去掉窗函数影响）
                out[location] += (spec_r[i] * inv_n_fft) * w[base_index];
            }
        }
    }
}

// ----------------------------- Vulkan/通用图实现辅助 -----------------------------
// 说明：以下函数用于在“非自定义算子”路径下构建 STFT/ISTFT 计算图，避免 GGML_OP_CUSTOM。
// 参考 whisper.cpp 的 STFT 方案：通过 conv_1d + 预计算基矩阵实现频域变换。

// 近似 atan(z)，假设 |z| <= 1。公式来自常见快速 atan 近似，最大误差约 0.002rad。
static ggml_tensor * tts_atan_approx_unit(ggml_context * ctx, ggml_tensor * z,
                                          std::vector<tts_graph_const_input> * const_inputs) {
    // atan(z) ≈ (π/4)*z - z*(|z|-1)*(0.2447 + 0.0663*|z|)
    // 此处 z >= 0（由外层保证），所以 |z| == z。
    const float k_pi_over_4 = TTS_TWO_PI_F * 0.25f;
    static const float k_c1 = 0.2447f;
    static const float k_c2 = 0.0663f;
    static const float k_neg_one = -1.0f;

    ggml_tensor * t = ggml_add(ctx, ggml_scale(ctx, z, k_c2), tts_const_tensor(ctx, &k_c1, const_inputs)); // 0.2447 + 0.0663*z
    ggml_tensor * z_minus_one = ggml_add(ctx, z, tts_const_tensor(ctx, &k_neg_one, const_inputs));         // z - 1
    ggml_tensor * corr = ggml_mul(ctx, ggml_mul(ctx, z, z_minus_one), t);      // z*(z-1)*(0.2447+0.0663*z)
    return ggml_sub(ctx, ggml_scale(ctx, z, k_pi_over_4), corr);               // (π/4)*z - corr
}

// 近似 atan2(y, x)，避免依赖缺失的 atan2 算子。
static ggml_tensor * tts_atan2_approx(ggml_context * ctx, ggml_tensor * y, ggml_tensor * x,
                                      std::vector<tts_graph_const_input> * const_inputs) {
    const float k_pi = TTS_TWO_PI_F * 0.5f;
    const float k_pi_over_2 = TTS_TWO_PI_F * 0.25f;
    static const float k_eps = 1e-8f; // 防止除 0
    static const float k_one = 1.0f;
    static const float k_neg_one = -1.0f;

    ggml_tensor * abs_y = ggml_abs(ctx, y);
    ggml_tensor * abs_x = ggml_abs(ctx, x);
    ggml_tensor * abs_y_eps = ggml_add(ctx, abs_y, tts_const_tensor(ctx, &k_eps, const_inputs));
    ggml_tensor * abs_x_eps = ggml_add(ctx, abs_x, tts_const_tensor(ctx, &k_eps, const_inputs));

    // mask：|y| > |x|
    ggml_tensor * mask = ggml_step(ctx, ggml_sub(ctx, abs_y_eps, abs_x_eps));

    // z0/z1 保证在 [0, 1]，用于 atan 近似
    ggml_tensor * z0 = ggml_div(ctx, abs_y_eps, abs_x_eps);
    ggml_tensor * z1 = ggml_div(ctx, abs_x_eps, abs_y_eps);
    ggml_tensor * atan0 = tts_atan_approx_unit(ctx, z0, const_inputs);
    ggml_tensor * atan1 = tts_atan_approx_unit(ctx, z1, const_inputs);

    ggml_tensor * base1 = ggml_add(ctx, ggml_neg(ctx, atan1), tts_const_tensor(ctx, &k_pi_over_2, const_inputs)); // π/2 - atan1
    ggml_tensor * one_minus_mask = ggml_add(ctx, ggml_neg(ctx, mask), tts_const_tensor(ctx, &k_one, const_inputs));
    ggml_tensor * base = ggml_add(ctx,
                                  ggml_mul(ctx, atan0, one_minus_mask),
                                  ggml_mul(ctx, base1, mask));

    // sign(y)：用 step(y) 生成 {+1/-1}，避免 y==0 时得到 0（影响 x<0 的象限修正）。
    ggml_tensor * sign_y = ggml_add(ctx, ggml_scale(ctx, ggml_step(ctx, y), 2.0f), tts_const_tensor(ctx, &k_neg_one, const_inputs));
    ggml_tensor * angle = ggml_mul(ctx, base, sign_y);

    // 当 x < 0 时，angle = π*sign(y) - angle
    ggml_tensor * mask_x = ggml_step(ctx, ggml_neg(ctx, x));
    ggml_tensor * pi_sign = ggml_scale(ctx, sign_y, k_pi);
    ggml_tensor * two_angle = ggml_scale(ctx, angle, 2.0f);
    ggml_tensor * delta = ggml_sub(ctx, pi_sign, two_angle);
    ggml_tensor * angle_fixed = ggml_add(ctx, angle, ggml_mul(ctx, mask_x, delta));

    // 当幅度极小（x≈0 且 y≈0）时，直接置 0，避免异常角度干扰后续网络。
    ggml_tensor * mag2 = ggml_add(ctx, ggml_mul(ctx, x, x), ggml_mul(ctx, y, y));
    ggml_tensor * non_zero = ggml_step(ctx, ggml_sub(ctx, mag2, tts_const_tensor(ctx, &k_eps, const_inputs)));
    return ggml_mul(ctx, angle_fixed, non_zero);
}

// 使用反射 padding 索引对输入做 padding，返回形状 [len + 2*pad, 1, batch]。
static ggml_tensor * tts_pad_reflect_with_indices(
    ggml_context * ctx,
    ggml_tensor * a,
    ggml_tensor * pad_indices) {
    GGML_ASSERT(a != nullptr);
    GGML_ASSERT(pad_indices != nullptr);
    GGML_ASSERT(a->ne[2] == 1 && a->ne[3] == 1);
    GGML_ASSERT(pad_indices->type == GGML_TYPE_I32);
    GGML_ASSERT(pad_indices->ne[1] == a->ne[1]);

    if (!ggml_is_contiguous(a)) {
        a = ggml_cont(ctx, a);
    }

    // 将 [len, batch] 视作 [1, len, batch]，便于 get_rows 按“行索引”抽取
    ggml_tensor * a_view = ggml_reshape_3d(ctx, a, 1, a->ne[0], a->ne[1]);
    ggml_tensor * gathered = ggml_get_rows(ctx, a_view, pad_indices); // [1, padded_len, batch]
    return ggml_transpose(ctx, gathered); // [padded_len, 1, batch]
}

}  // namespace

struct ggml_tensor * stft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided) {
    if (window->ne[0] != (int64_t) n_fft) {
        TTS_ABORT("For #stft the window_size, %d, must be equal to n_fft, %d.\n", (int) window->ne[0], (int) n_fft);
    }
    if (a->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32) {
        TTS_ABORT("For #stft inputs must be GGML_TYPE_F32.\n");
    }

    // 输出：shape = [ne0, n_frames, batch, 2]
    // - one_sided=false：ne0=n_fft（完整频谱）
    // - one_sided=true： ne0=n_fft/2+1（只保留 [0..Nyquist]，减少约一半输出与后续拷贝）
    const int64_t n_frames = (int64_t) (a->ne[0] / (int64_t) hop) + 1;
    const int64_t ne0 = one_sided ? (((int64_t) n_fft / 2) + 1) : (int64_t) n_fft;
    const int64_t ne1 = n_frames;
    const int64_t ne2 = a->ne[1];
    const int64_t ne3 = 2;

    ggml_tensor * args[2] = { a, window };

    ggml_tensor * out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    out->op = GGML_OP_CUSTOM;
    out->src[0] = args[0];
    out->src[1] = args[1];

    tts_stft_op_params p{};
    p.base.fun     = tts_compute_stft_custom;
    p.base.n_tasks = GGML_N_TASKS_MAX;
    p.base.userdata = nullptr;
    p.n_fft = (int32_t) n_fft;
    p.hop   = (int32_t) hop;
    p.abs_and_angle = abs_and_angle ? 1 : 0;
    p.one_sided = one_sided ? 1 : 0;
    ggml_set_op_params(out, &p, sizeof(p));

    return out;
}

struct ggml_tensor * istft(ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * window_squared_sum, struct ggml_tensor * window, size_t n_fft, size_t hop, bool abs_and_angle, bool one_sided) {
    const int64_t expected_ne0 = one_sided ? ((int64_t) n_fft / 2) + 1 : (int64_t) n_fft;
    if (a->ne[0] != expected_ne0) {
        TTS_ABORT("For #istft input ne[0]=%d mismatches expected %d (n_fft=%d, one_sided=%d).\n",
                  (int) a->ne[0], (int) expected_ne0, (int) n_fft, (int) one_sided);
    }
    if (window->ne[0] != (int64_t) n_fft) {
        TTS_ABORT("For #istft the window_size, %d, must be equal to n_fft, %d.\n", (int) window->ne[0], (int) n_fft);
    }
    if (a->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32) {
        TTS_ABORT("For #istft inputs must be GGML_TYPE_F32.\n");
    }

    // 输出长度与旧 ggml 一致：(frames - 1) * hop
    const int64_t n_frames = a->ne[1];
    const int64_t out_len  = (n_frames - 1) * (int64_t) hop;

    ggml_tensor * args[2] = { a, window };

    ggml_tensor * out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out_len, a->ne[2], 1, 1);
    out->op = GGML_OP_CUSTOM;
    out->src[0] = args[0];
    out->src[1] = args[1];

    tts_istft_op_params p{};
    p.base.fun     = tts_compute_istft_custom;
    p.base.n_tasks = GGML_N_TASKS_MAX;
    p.base.userdata = nullptr;
    p.n_fft = (int32_t) n_fft;
    p.hop   = (int32_t) hop;
    p.from_abs_and_angle = abs_and_angle ? 1 : 0;
    p.one_sided = one_sided ? 1 : 0;
    ggml_set_op_params(out, &p, sizeof(p));

    // 去掉窗函数影响（与旧实现一致）
    out = ggml_div(ctx, out, window_squared_sum);
    return out;
}

struct ggml_tensor * stft_graph(
    ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * forward_basis,
    struct ggml_tensor * pad_indices,
    std::vector<tts_graph_const_input> * const_inputs,
    size_t n_fft,
    size_t hop,
    bool abs_and_angle,
    bool one_sided) {
    if (a == nullptr || forward_basis == nullptr || pad_indices == nullptr) {
        TTS_ABORT("stft_graph: 输入为空。\n");
    }
    if (!one_sided) {
        TTS_ABORT("stft_graph: 当前仅支持 one_sided=true。\n");
    }
    if (a->type != GGML_TYPE_F32 || forward_basis->type != GGML_TYPE_F32) {
        TTS_ABORT("stft_graph: 输入与基矩阵必须为 F32。\n");
    }
    if (pad_indices->type != GGML_TYPE_I32) {
        TTS_ABORT("stft_graph: pad_indices 必须为 I32。\n");
    }

    const int64_t half = (int64_t) (n_fft / 2);
    const int64_t cutoff = (int64_t) (n_fft / 2 + 1);
    const int64_t padded_len = a->ne[0] + 2 * half;
    if (pad_indices->ne[0] != padded_len || pad_indices->ne[1] != a->ne[1]) {
        TTS_ABORT("stft_graph: pad_indices 维度不匹配（pad_len=%d, batch=%d）。\n",
                  (int) pad_indices->ne[0], (int) pad_indices->ne[1]);
    }
    if (forward_basis->ne[0] != (int64_t) n_fft || forward_basis->ne[1] != 1 || forward_basis->ne[2] != cutoff * 2) {
        TTS_ABORT("stft_graph: forward_basis 维度不匹配（期望 [n_fft,1,%d]）。\n", (int) (cutoff * 2));
    }

    // 1) 反射 padding：利用索引 + get_rows，避免 GGML_OP_CUSTOM / pad_reflect_1d
    ggml_tensor * padded = tts_pad_reflect_with_indices(ctx, a, pad_indices); // [padded_len, 1, batch]
    padded = ggml_cont(ctx, padded); // 说明：conv_1d 更偏好连续输入

    // 2) conv_1d 得到实部/虚部（channels = 2 * cutoff）
    ggml_tensor * stft_raw = ggml_conv_1d(ctx, forward_basis, padded, (int) hop, 0, 1);
    if (stft_raw->ne[1] != cutoff * 2) {
        TTS_ABORT("stft_graph: conv 输出通道数不匹配（got=%d expected=%d）。\n",
                  (int) stft_raw->ne[1], (int) (cutoff * 2));
    }

    // 3) reshape + permute => [cutoff, frames, batch, 2]
    ggml_tensor * stft_4d = ggml_reshape_4d(ctx, stft_raw, stft_raw->ne[0], cutoff, stft_raw->ne[2], 2);
    ggml_tensor * stft_perm = ggml_permute(ctx, stft_4d, 1, 0, 2, 3);

    if (!abs_and_angle) {
        return stft_perm;
    }

    // 4) 转为 (magnitude, angle)
    ggml_tensor * real = ggml_view_3d(ctx, stft_perm,
                                      stft_perm->ne[0], stft_perm->ne[1], stft_perm->ne[2],
                                      stft_perm->nb[1], stft_perm->nb[2], 0);
    ggml_tensor * imag = ggml_view_3d(ctx, stft_perm,
                                      stft_perm->ne[0], stft_perm->ne[1], stft_perm->ne[2],
                                      stft_perm->nb[1], stft_perm->nb[2], stft_perm->nb[3]);

    ggml_tensor * real_sq = ggml_mul(ctx, real, real);
    ggml_tensor * imag_sq = ggml_mul(ctx, imag, imag);
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_add(ctx, real_sq, imag_sq));
    ggml_tensor * ang = tts_atan2_approx(ctx, imag, real, const_inputs);

    ggml_tensor * mag4 = ggml_reshape_4d(ctx, mag, mag->ne[0], mag->ne[1], mag->ne[2], 1);
    ggml_tensor * ang4 = ggml_reshape_4d(ctx, ang, ang->ne[0], ang->ne[1], ang->ne[2], 1);
    return ggml_concat(ctx, mag4, ang4, 3);
}

struct ggml_tensor * istft_graph(
    ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * window_squared_sum,
    struct ggml_tensor * inverse_basis,
    size_t n_fft,
    size_t hop,
    bool abs_and_angle,
    bool one_sided) {
    if (a == nullptr || window_squared_sum == nullptr || inverse_basis == nullptr) {
        TTS_ABORT("istft_graph: 输入为空。\n");
    }
    if (!one_sided) {
        TTS_ABORT("istft_graph: 当前仅支持 one_sided=true。\n");
    }
    if (a->type != GGML_TYPE_F32 || inverse_basis->type != GGML_TYPE_F32) {
        TTS_ABORT("istft_graph: 输入与基矩阵必须为 F32。\n");
    }
    if (a->ne[3] != 2) {
        TTS_ABORT("istft_graph: 输入最后一维必须为 2（实/虚或幅度/相位）。\n");
    }

    const int64_t half = (int64_t) (n_fft / 2);
    const int64_t cutoff = (int64_t) (n_fft / 2 + 1);
    if (a->ne[0] != cutoff) {
        TTS_ABORT("istft_graph: 频率维长度不匹配（ne0=%d，期望=%d）。\n", (int) a->ne[0], (int) cutoff);
    }
    if (inverse_basis->ne[0] != (int64_t) n_fft || inverse_basis->ne[1] != 1 || inverse_basis->ne[2] != cutoff * 2) {
        TTS_ABORT("istft_graph: inverse_basis 维度不匹配（期望 [n_fft,1,%d]）。\n", (int) (cutoff * 2));
    }

    ggml_tensor * part0 = ggml_view_3d(ctx, a, a->ne[0], a->ne[1], a->ne[2], a->nb[1], a->nb[2], 0);
    ggml_tensor * part1 = ggml_view_3d(ctx, a, a->ne[0], a->ne[1], a->ne[2], a->nb[1], a->nb[2], a->nb[3]);

    ggml_tensor * real = nullptr;
    ggml_tensor * imag = nullptr;
    if (abs_and_angle) {
        // 相位来自 atan2 的输出，保持与前向 STFT 一致（imag 可为负）
        real = ggml_mul(ctx, part0, ggml_cos(ctx, part1));
        imag = ggml_mul(ctx, part0, ggml_sin(ctx, part1));
    } else {
        real = part0;
        imag = part1;
    }

    // [freq, frames, batch] -> [frames, freq, batch]
    ggml_tensor * real_t = ggml_transpose(ctx, real);
    ggml_tensor * imag_t = ggml_transpose(ctx, imag);
    ggml_tensor * feat = ggml_concat(ctx, real_t, imag_t, 1); // [frames, 2*freq, batch]
    feat = ggml_cont(ctx, feat);

    // 通过转置卷积做 overlap-add，padding=half 以还原输出长度
    ggml_tensor * out = tts_conv_transpose_1d(ctx, inverse_basis, feat, (int) hop, (int) half, 1);
    out = ggml_reshape_4d(ctx, out, out->ne[0], out->ne[2], 1, 1); // [out_len, batch, 1, 1]

    // 去掉窗函数影响
    return ggml_div(ctx, out, window_squared_sum);
}

struct ggml_tensor * tts_conv_transpose_1d(
    ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * b,
    int s0,
    int p0,
    int d0,
    int output_padding,
    int groups) {
    if (groups <= 0) {
        TTS_ABORT("For #tts_conv_transpose_1d groups must be > 0.\n");
    }
    if (s0 <= 0) {
        TTS_ABORT("For #tts_conv_transpose_1d stride must be > 0.\n");
    }
    if (d0 <= 0) {
        TTS_ABORT("For #tts_conv_transpose_1d dilation must be > 0.\n");
    }
    if (p0 < 0 || output_padding < 0) {
        TTS_ABORT("For #tts_conv_transpose_1d padding/output_padding must be >= 0.\n");
    }

    // groups==1：尽量复用 ggml 内置实现（其目前只支持 p0==0 且 d0==1）
    if (groups == 1) {
        if (d0 != 1) {
            TTS_ABORT("For #tts_conv_transpose_1d (groups==1) ggml 0.9.4 only supports dilation==1.\n");
        }
        if (output_padding > 0 && p0 == 0) {
            // ggml 0.9.4 的 conv_transpose_1d 本身无法“变长”；当 padding==0 且需要 output_padding>0 时，
            // 仅靠 view 裁剪无法补齐尾部新增元素（可能是 0，也可能包含贡献，取决于框架语义）。
            // 当前仓库模型未用到该组合，先显式报错，避免 silent wrong。
            TTS_ABORT("For #tts_conv_transpose_1d output_padding>0 with padding==0 is not supported in ggml 0.9.4 compat path.\n");
        }

        // 先用 ggml 的实现算一个“padding=0”的更长结果，再裁剪实现 padding/output_padding。
        ggml_tensor * raw = ggml_conv_transpose_1d(ctx, a, b, s0, 0 /*p0*/, 1 /*d0*/);

        if (p0 == 0 && output_padding == 0) {
            return raw;
        }

        // 等价裁剪规则：
        // - 对 padding：从左侧裁 p0 个样本、从右侧裁 p0 个样本
        // - 对 output_padding：相比 output_padding==0 的情况，右侧少裁 output_padding 个样本（即让输出更长）
        const int64_t start = (int64_t) p0;
        const int64_t end_crop = (int64_t) p0 - (int64_t) output_padding;
        if (end_crop < 0) {
            TTS_ABORT("For #tts_conv_transpose_1d output_padding must be <= padding in current compat path.\n");
        }
        if (raw->ne[0] < start + end_crop) {
            TTS_ABORT("For #tts_conv_transpose_1d invalid crop: raw_len=%d, start=%d, end_crop=%d.\n",
                      (int) raw->ne[0], (int) start, (int) end_crop);
        }

        const int64_t new_ne0 = raw->ne[0] - start - end_crop;
        ggml_tensor * view = ggml_view_4d(ctx, raw,
                                          /*ne0=*/new_ne0, /*ne1=*/raw->ne[1], /*ne2=*/raw->ne[2], /*ne3=*/raw->ne[3],
                                          /*nb1=*/raw->nb[1], /*nb2=*/raw->nb[2], /*nb3=*/raw->nb[3],
                                          /*offset=*/(size_t) (start * raw->nb[0]));
        return ggml_cont(ctx, view);
    }

    // groups>1：项目侧 CPU 自定义实现（用于 Kokoro depthwise 转置卷积）。
    if (b->type != GGML_TYPE_F32) {
        TTS_ABORT("For #tts_conv_transpose_1d (groups>1) input must be GGML_TYPE_F32.\n");
    }
    if (a->type != GGML_TYPE_F16 && a->type != GGML_TYPE_F32) {
        TTS_ABORT("For #tts_conv_transpose_1d (groups>1) kernel must be GGML_TYPE_F16 or GGML_TYPE_F32.\n");
    }
    if (a->ne[3] != 1 || b->ne[3] != 1) {
        TTS_ABORT("For #tts_conv_transpose_1d only ne[3]==1 tensors are supported.\n");
    }
    if (a->ne[2] != b->ne[1]) {
        TTS_ABORT("For #tts_conv_transpose_1d kernel ne[2](%d) must equal input ne[1](%d).\n", (int) a->ne[2], (int) b->ne[1]);
    }
    if (a->ne[2] % groups != 0) {
        TTS_ABORT("For #tts_conv_transpose_1d input channels (%d) must be divisible by groups (%d).\n", (int) a->ne[2], groups);
    }

    const int64_t out_len = (b->ne[0] - 1) * (int64_t) s0 - 2 * (int64_t) p0 + (int64_t) d0 * (a->ne[0] - 1) + (int64_t) output_padding + 1;
    if (out_len <= 0) {
        TTS_ABORT("For #tts_conv_transpose_1d invalid output length: %d.\n", (int) out_len);
    }

    const int64_t out_channels = a->ne[1] * (int64_t) groups;
    if (out_channels <= 0) {
        TTS_ABORT("For #tts_conv_transpose_1d invalid output channels: %d.\n", (int) out_channels);
    }

    ggml_tensor * out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, out_len, out_channels, b->ne[2]);
    out->op = GGML_OP_CUSTOM;
    out->src[0] = a;
    out->src[1] = b;

    tts_conv_transpose_1d_op_params p{};
    p.base.fun = tts_compute_conv_transpose_1d_custom;
    p.base.n_tasks = GGML_N_TASKS_MAX;
    p.base.userdata = nullptr;
    p.s0 = (int32_t) s0;
    p.p0 = (int32_t) p0;
    p.d0 = (int32_t) d0;
    p.output_padding = (int32_t) output_padding;
    p.groups = (int32_t) groups;
    ggml_set_op_params(out, &p, sizeof(p));

    return out;
}

void hann_window(size_t n_fft, std::vector<float> & tgt) {
    for (int i = 0; i < n_fft; i++) {
        float v = pow(sin(std::numbers::pi * (double)i / (double) n_fft), 2.0);
        tgt.push_back(v);
    }
}

// This is a custom map op for computing noise and relevant voiced sections.
void uv_noise_compute(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata) {
    float voice_threshold = ((float *) c->data)[0];
    float noise_std = ((float *) c->data)[1];
    float sin_amp = ((float *) c->data)[2];
    float sin_amp_div = ((float *) c->data)[3];
    float * rand_init = ((float *) c->data) + 4;

    const int rpt = (b->ne[0] + nth - 1)/nth;
    const int start = ith * rpt;
    const int end = MIN((ith + 1) * rpt, b->ne[0]);

    float * uv_dst = (float *) dst->data;
    float * noise_dst = (float *)((char*)dst->data + dst->nb[2]);
    float * tgt = (float *) b->data;

    for(int bt = 0; bt < b->ne[2]; bt++) {
        for(int r = start; r < end; r++) {
            if (tgt[r] > voice_threshold) {
                for (int h = 0; h < a->ne[1]; h++) {
                    int index = h*dst->ne[0]+r;
                    uv_dst[index] = sin_amp;
                    noise_dst[index] = noise_std * rand_init[index];
                }
            } else {
                for (int h = 0; h < a->ne[1]; h++) {
                    int index = h*dst->ne[0]+r;
                    uv_dst[index] = 0.0f;
                    noise_dst[index] = sin_amp_div * rand_init[index];
                }
            }
        }
    }
}

// This is a custom map op for applying cfg scale. It is used at the terminus of logit generation in Dia.
void cfg_scale(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    const float scale = ((float *) userdata)[0];
    const float max_output = ((float*) userdata)[1];
    const int rpt = (b->ne[0] + nth - 1)/nth;
    const int start = ith * rpt;
    const int end = MIN((ith + 1) * rpt, b->ne[0]);

    float * output = (float *) dst->data;
    float * cond = (float *) a->data;
    float * uncond = (float *) b->data;

    for(int bt = 0; bt < b->ne[2]; bt++) {
        for (int h = 0; h < b->ne[1]; h++) {
            int i = (h * b->ne[0]) + (bt * b->ne[0] * b->ne[1]);
            for(int r = start; r < end; r++) {
                // only let the output heads yield tokens up to EOS
                if (r > max_output) {
                    output[i+r] = -INFINITY;
                }
                const float cr = cond[i+r];
                const float ur = uncond[i+r];
                output[i+r] = cr + scale * (cr - ur);
            }
        }
    }
}

// currently this assumes a center view in which the output vector is reflectively padded by n_fft / 2 on each side.
void compute_window_squared_sum(size_t n_fft, size_t hop, size_t n_frames, float * tgt, float * window) {
    size_t cutoff = n_frames * hop;
    size_t half = n_fft / 2;
    std::memset(tgt, 0, cutoff*sizeof(float));
    // istft applies half / hop steps before the beginning of the sequence. We need to account for these accumulated windows.
    for (int i = 0; i < n_frames + (half / hop); i++) {
        for (int ii = 0; ii < n_fft; ii++) {
            int index = ii + i*hop - half;
            if (index < 0 || index >= cutoff) {
                continue;
            }
            // powf(x, 2) 的开销远高于 x*x，这里直接平方即可。
            const float w = window[ii];
            tgt[index] += w * w;
        }
    }
}

std::vector<std::string> split(std::string target, std::string split_on, bool include_split_characters) {
    std::vector<std::string> output;
    size_t last = 0;

    for (int i = 0; i < target.size(); i++) {
        if (i > last && split_on.find(target[i]) != std::string::npos) {
            std::string part(target.substr(last, i - last));
            output.push_back(part);
            if (include_split_characters) {
                output.push_back(target.substr(i, 1));
            }
            last = i+1;
        } else if (i == last && split_on.find(target[i]) != std::string::npos) {
            if (include_split_characters) {
                output.push_back(target.substr(i, 1));
            }
            last = i+1;
        }
    }
    if (last < target.size()) {
        std::string part(target.substr(last));
        output.push_back(part);
    }

    return output;
}

std::vector<std::string> split(std::string target, const char split_on, bool include_split_characters) {
    std::vector<std::string> output;
    size_t last = 0;

    for (int i = 0; i < target.size(); i++) {
        if (i > last && split_on == target[i]) {
            std::string part(target.substr(last, i - last));
            output.push_back(part);
            if (include_split_characters) {
                output.push_back(target.substr(i, 1));
            }
            last = i+1;
        } else if (i == last && split_on == target[i]) {
            if (include_split_characters) {
                output.push_back(target.substr(i, 1));
            }
            last = i+1;
        }
    }
    if (last < target.size()) {
        std::string part(target.substr(last));
        output.push_back(part);
    }

    return output;
}

std::string strip(std::string target, std::string vals) {
    target.erase(target.begin(), std::find_if(target.begin(), target.end(), [&vals](unsigned char ch) {
        return vals.find(ch) == std::string::npos;
    }));
    target.erase(std::find_if(target.rbegin(), target.rend(), [&vals](unsigned char ch) {
        return vals.find(ch) == std::string::npos;
    }).base(), target.end());
    return target;
}

std::string replace_any(std::string target, std::string to_replace, std::string replacement) {
    for (int i = 0; i < to_replace.size(); i++) {
        size_t position = target.find(to_replace[i]);
        while (position != std::string::npos) {
            target.replace(position, 1, replacement);
            position = target.find(to_replace[i]);
        }
    }
    return target;
}

struct model_tensor_meta compute_tensor_meta(std::string name_prefix, ggml_context * weight_ctx, std::function<void(ggml_tensor*)>* callback) {
    model_tensor_meta meta;
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        if (callback) {
            (*callback)(cur);
        }
        std::string::size_type pos = std::string(cur->name).find(".", 0);
        std::string top_level(std::string(cur->name).substr(0, pos));
        if (top_level == name_prefix) {
            meta.n_tensors += 1;
            meta.n_bytes += ggml_nbytes_pad(cur);
        }
    }
    return meta;
}
