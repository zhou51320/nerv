#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

// Using this simple struct as opposed to a common std::vector allows us to return the cpu buffer
// pointer directly rather than copying the contents of the buffer to a predefined std::vector.
struct tts_response {
	float * data;
	size_t n_outputs = 0;
	uint32_t hidden_size; // this parameter is only currently used by the t5_encoder for which n_outputs corresponds to sequence length;
};

enum tts_arch {
	PARLER_TTS_ARCH = 0,
	KOKORO_ARCH = 1,
	DIA_ARCH = 2,
	ORPHEUS_ARCH = 3,
};

// 文本语言偏好（主要用于 Kokoro 的多语言前端与数字读法选择）。
enum class tts_language : uint8_t {
    ZH = 0,
    EN = 1,
    JA = 2,
};

// 推理后端类型（用于选择 ggml 的设备后端）。
// 说明：
// - CPU：纯 CPU 推理。
// - Metal：Apple 设备上的 Metal 后端（需要编译启用 GGML_METAL）。
// - Vulkan：跨平台 Vulkan 后端（需要编译启用 GGML_VULKAN）。
// - Auto：自动选择可用的 GPU 后端（优先 Metal，其次 Vulkan），若都不可用则回退到 CPU（由上层决定是否允许回退）。
enum class tts_compute_backend {
    CPU = 0,
    METAL,
    VULKAN,
    AUTO,
};

// 推理后端配置。
// 说明：目前主要用于 Vulkan 的设备选择（device=0 表示第 0 个 Vulkan 设备）。
// 未来如需扩展到 CUDA/OpenCL 等后端，也可复用该结构体。
struct tts_backend_config {
    tts_compute_backend backend = tts_compute_backend::CPU;
    int device = 0;
    // 是否优先使用“主机可见”的权重缓冲（主要用于 Vulkan 场景下回退 CPU 时避免读到设备内存）。
    bool prefer_host_buffer = false;
};

const std::map<std::string, tts_arch> SUPPORTED_ARCHITECTURES = {
	{ "parler-tts", PARLER_TTS_ARCH },
	{ "kokoro", KOKORO_ARCH },
	{ "dia", DIA_ARCH },
	{ "orpheus", ORPHEUS_ARCH }
};

/// Given a map from keys to values, creates a new map from values to keys
template<typename K, typename V>
static std::map<V, K> reverse_map(const std::map<K, V>& m) {
    std::map<V, K> r;
    for (const auto& kv : m) {
        r[kv.second] = kv.first;
    }
    return r;
}

const std::map<tts_arch, std::string> ARCHITECTURE_NAMES = reverse_map(SUPPORTED_ARCHITECTURES);

struct generation_configuration {
    generation_configuration(
    	std::string voice = "",
    	int top_k = 50,
    	float temperature = 1.0,
    	float repetition_penalty = 1.0,
        bool use_cross_attn = true,
    	int max_tokens = 0,
    	float top_p = 1.0,
    	bool sample = true,
        tts_language language = tts_language::ZH,
        std::string zh_dict_dir = ""):
        use_cross_attn(use_cross_attn),
        temperature(temperature),
        repetition_penalty(repetition_penalty),
        top_p(top_p),
        top_k(top_k),
        max_tokens(max_tokens),
        voice(std::move(voice)),
        sample(sample),
        zh_dict_dir(std::move(zh_dict_dir)),
        language(language) {};

    bool use_cross_attn;
    float temperature;
    float repetition_penalty;
    float top_p;
    int top_k;
    int max_tokens;
    std::string voice = "";
    bool sample = true;
    // Kokoro 中文前端词典目录（可选）：包含 `pinyin_phrase.txt` / `pinyin.txt`。
    // - 为空：默认尝试使用工作目录下的 `dict/`；若加载失败则回退到内置逐字映射。
    // - "-"：显式禁用词典（便于对比/排查）。
    std::string zh_dict_dir = "";
    tts_language language = tts_language::ZH;
};

struct tts_runner {
	struct ggml_context * ctx = nullptr;
	float sampling_rate = 44100.0f;
	bool supports_voices = false;

    virtual ~tts_runner() = default;

	void init_build(std::vector<uint8_t>* buf_compute_meta);
	void free_build();
};

struct ggml_tensor;
struct tts_model_loader;
struct llama_mmap;

struct tts_generation_runner : tts_runner {
    const reference_wrapper<const tts_model_loader> loader;
    unique_ptr<llama_mmap> buf;
    explicit tts_generation_runner(const tts_model_loader & loader);
    ~tts_generation_runner() override;

    virtual void                assign_weight(const char * name, ggml_tensor & tensor) = 0;
    virtual void                prepare_post_load()                                    = 0;
    virtual vector<string_view> list_voices();
    virtual void                update_conditional_prompt(const char * file_path, const char * prompt);
    virtual void generate(const char * sentence, tts_response & output, const generation_configuration & config) = 0;

    // 可选：返回“音素化后的 prompt”（用于 CLI 调试打印）。
    // 说明：
    // - 默认实现返回 false，表示该 runner 不支持输出音素串；
    // - Kokoro runner 会覆盖该接口，返回其实际使用的 phoneme/token 串（UTF-8）。
    virtual bool try_phonemize(const char * sentence, std::string & out_phonemes, const generation_configuration & config) {
        (void) sentence;
        (void) out_phonemes;
        (void) config;
        return false;
    }

    // 可选：返回“分段后的文本 + 每段对应的音素串”（用于 CLI 更直观地观测分词/多音字/数字/单位等处理是否符合预期）。
    // 默认实现：若 try_phonemize() 可用，则把整段文本作为一个 segment 返回。
    struct phoneme_segment {
        std::string text;
        std::string phonemes;
        bool        is_boundary = false;
    };

    virtual bool try_phonemize_segments(const char * sentence,
                                        std::string & out_phonemes,
                                        std::vector<phoneme_segment> & out_segments,
                                        const generation_configuration & config) {
        out_segments.clear();
        if (!try_phonemize(sentence, out_phonemes, config)) {
            return false;
        }
        phoneme_segment seg;
        seg.text = sentence ? std::string(sentence) : std::string();
        seg.phonemes = out_phonemes;
        seg.is_boundary = false;
        out_segments.push_back(std::move(seg));
        return true;
    }
};

struct test_tts_generation_runner : tts_generation_runner {
    explicit test_tts_generation_runner(const tts_model_loader & loader);

    void assign_weight(const char * name, ggml_tensor & tensor) final;
    void prepare_post_load() final;
};
