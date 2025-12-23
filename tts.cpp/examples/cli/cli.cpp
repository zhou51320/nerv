#include <thread>
#include <vector>
#include <cstdarg>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <shellapi.h>
#endif

#include "../../src/models/kokoro/zh_frontend.h"
#include "../../src/models/loaders.h"
#include "args.h"
#include "common.h"
#include "ggml.h"
#include "playback.h"
#include "vad.h"
#include "write_file.h"
#include "../../src/util.h"

// 说明：CLI 以前在“加载模型 / 生成语音”期间几乎没有任何输出，容易让用户误以为程序没启动或卡死。
// 这里补充关键阶段的流程日志（统一输出到 stderr，避免污染 stdout 的语音/列表输出）。
static void tts_cli_set_unbuffered() {
    // 尽量保证日志“立刻可见”（尤其是在 stdout/stderr 被重定向时）。
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
#ifdef _WIN32
    // Windows 控制台默认代码页为 ANSI，UTF-8 中文会乱码，这里强制切到 UTF-8。
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

static void tts_cli_log(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[tts-cli] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stderr);
}

static bool tts_cli_try_file_size(const std::string & path, uint64_t & out_bytes) {
    std::error_code ec;
    const auto      bytes = std::filesystem::file_size(std::filesystem::path(path), ec);
    if (ec) {
        return false;
    }
    out_bytes = static_cast<uint64_t>(bytes);
    return true;
}

static std::filesystem::path tts_cli_self_exe_dir(const char * argv0) {
    // 说明：用于默认寻找 `dict/`（多音字短语词典），避免用户必须在“特定工作目录”运行 tts-cli。
    // 优先使用 WinAPI 获取真实路径；其它平台退化为解析 argv0。
#ifdef _WIN32
    wchar_t buf[MAX_PATH + 1] = {0};
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        return std::filesystem::path(buf).parent_path();
    }
#endif

    if (argv0 == nullptr || argv0[0] == '\0') {
        return {};
    }

    std::error_code ec;
    std::filesystem::path p = std::filesystem::u8path(argv0);
    p = std::filesystem::absolute(p, ec);
    if (ec) {
        return {};
    }
    return p.parent_path();
}

struct tts_cli_zh_dict_resolution {
    // 传给库侧的值：
    // - ""        ：自动模式（库侧会尝试 "dict" 并在失败时回退内置词典）
    // - "<path>"  ：显式指定外部目录
    // - ":builtin"：强制内置（若编译时启用）
    // - "-"       ：禁用词典
    std::string dir;
    // 打印到日志里的描述（便于用户确认“到底用的是哪份词典”）。
    std::string desc;
};

static bool tts_cli_has_zh_phrase_dict(const std::filesystem::path & dict_dir) {
    // 说明：仅用短语词典是否存在作为判定条件（pinyin_phrase.txt 是 DP 分词 + 多音字消歧的关键）。
    std::error_code ec;
    return std::filesystem::exists(dict_dir / "pinyin_phrase.txt", ec) && !ec;
}

static std::filesystem::path tts_cli_find_zh_dict_near_exe(const std::filesystem::path & exe_dir) {
    // 说明：优先支持“随 exe 打包”的目录结构，但也兼容开发态：
    // - 发行版/便携包：<exe_dir>/dict/pinyin_phrase.txt
    // - 开发构建：build/bin/tts-cli.exe，而词典在仓库根目录 dict/（即 <exe_dir>/../../dict）
    //
    // 实现策略：从 exe_dir 开始向上查找（最多几层）是否存在 dict/。
    if (exe_dir.empty()) {
        return {};
    }

    constexpr int kMaxSearchParents = 8;
    std::filesystem::path cur = exe_dir;
    for (int depth = 0; depth <= kMaxSearchParents && !cur.empty(); ++depth) {
        const std::filesystem::path candidate = cur / "dict";
        if (tts_cli_has_zh_phrase_dict(candidate)) {
            return candidate;
        }

        const std::filesystem::path parent = cur.parent_path();
        if (parent == cur) {
            break;
        }
        cur = parent;
    }
    return {};
}

static tts_cli_zh_dict_resolution tts_cli_resolve_zh_dict_dir(const std::string & user_value, const char * argv0) {
    // 说明：
    // - 用户显式传入 --zh-dict-dir 时：完全尊重其值；
    // - 未传时（自动）：优先 ./dict（工作目录），否则在 exe_dir 及其父目录中查找 dict/，最后回退内置词典（若启用）。
    tts_cli_zh_dict_resolution res{};

    if (!user_value.empty()) {
        res.dir = user_value;
        if (user_value == "-") {
            res.desc = "(disabled)";
        } else if (user_value == ":builtin") {
            res.desc = "(builtin)";
        } else {
            res.desc = user_value;
        }
        return res;
    }

    // 1) 工作目录 ./dict（保持空字符串：让库侧继续走“auto: ./dict -> builtin”的逻辑）
    if (tts_cli_has_zh_phrase_dict(std::filesystem::path("dict"))) {
        res.dir = "";
        res.desc = "(auto: ./dict)";
        return res;
    }

    // 2) exe_dir 及其父目录查找 dict/（兼容 build/bin 与便携包结构）
    const std::filesystem::path exe_dir = tts_cli_self_exe_dir(argv0);
    const std::filesystem::path found = tts_cli_find_zh_dict_near_exe(exe_dir);
    if (!found.empty()) {
        res.dir = found.u8string();
        // 说明：为避免误导，这里直接打印“找到的实际路径”，不再强行标注为 exe_dir/dict。
        res.desc = "(auto: found) " + res.dir;
        return res;
    }

    // 3) 找不到外部词典：交给库侧回退内置词典（若启用）
    res.dir = "";
    res.desc = "(auto: ./dict -> builtin)";
    return res;
}

static double tts_cli_us_to_ms(const int64_t us) {
    return us / 1000.0;
}

static std::string tts_cli_to_lower(std::string v) {
    for (char & c : v) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return v;
}

static const char * tts_cli_bench_prompt() {
    // 说明：内置一段“中英混合、约 200 字”的固定测试文本，方便快速验证：
    // - 中文：停顿/语速/数字读法/单位（℃）
    // - 英文：常见单词与缩写（OpenAI/GPU/Vulkan/C++17）
    // 注意：当 CLI 传入 --bench 时，会强制使用该文本进行推理，并忽略用户通过 -p/--prompt 传入的内容。
    return u8"你好，这是 tts.cpp 的中英混合语音测试，用来检查停顿、重音和数字读法。今天是2025年12月21日，温度23.5°C；1+1=2。Now in English: Hello world! This is a quick benchmark for Kokoro TTS. Please pronounce OpenAI, GPU, Vulkan, and C++17 clearly. Thanks.";
}

static bool tts_cli_try_parse_int(const std::string & text, int & out) {
    // 说明：CLI 参数里会出现类似 "vulkan:0" 的形式，这里提供一个不抛异常的 int 解析工具。
    // - 返回 true：成功解析出一个完整的十进制整数（允许前导空格与正负号）。
    // - 返回 false：格式非法（例如包含非数字字符）。
    int  v = 0;
    char extra = 0;
    if (std::sscanf(text.c_str(), "%d%c", &v, &extra) == 1) {
        out = v;
        return true;
    }
    return false;
}

static bool tts_cli_parse_device_arg(const std::string & value, int default_vulkan_device, tts_backend_config & out) {
    // 说明：`--device` 仅用于测试/对比，常见用户不需要关心该参数。
    // 支持的写法：
    // - cpu
    // - metal
    // - vulkan / vk
    // - vulkan:0 / vk:0（同时指定 Vulkan 设备索引）
    // - auto（交给库侧按可用后端自动选择：优先 Metal，其次 Vulkan）
    const std::string lower = tts_cli_to_lower(value);

    out = {};
    out.device = default_vulkan_device;

    if (lower == "cpu") {
        out.backend = tts_compute_backend::CPU;
        return true;
    }
    if (lower == "metal") {
        out.backend = tts_compute_backend::METAL;
        return true;
    }
    if (lower == "auto") {
        out.backend = tts_compute_backend::AUTO;
        return true;
    }

    // 处理 vulkan[:N] / vk[:N]
    std::string name = lower;
    int device_index = default_vulkan_device;
    const size_t colon = lower.find(':');
    if (colon != std::string::npos) {
        name = lower.substr(0, colon);
        const std::string idx = lower.substr(colon + 1);
        if (idx.empty()) {
            return false;
        }
        if (!tts_cli_try_parse_int(idx, device_index) || device_index < 0) {
            return false;
        }
    }

    if (name == "vulkan" || name == "vk") {
        out.backend = tts_compute_backend::VULKAN;
        out.device = device_index;
        return true;
    }

    return false;
}

static tts_backend_config tts_cli_default_backend_config(int default_vulkan_device) {
    // 说明：默认设备跟随“编译时启用的后端”。
    // - 编译启用 Metal：默认 Metal
    // - 否则编译启用 Vulkan：默认 Vulkan
    // - 都没启用：默认 CPU
    tts_backend_config out{};
    out.device = default_vulkan_device;

#if defined(GGML_USE_METAL)
    out.backend = tts_compute_backend::METAL;
#elif defined(GGML_USE_VULKAN)
    out.backend = tts_compute_backend::VULKAN;
#else
    out.backend = tts_compute_backend::CPU;
#endif

    return out;
}

static bool tts_cli_parse_language(const std::string & value, tts_language & out) {
    const std::string lower = tts_cli_to_lower(value);
    if (lower.empty() || lower == "zh" || lower == "cn" || lower == "chinese") {
        out = tts_language::ZH;
        return true;
    }
    if (lower == "en" || lower == "english") {
        out = tts_language::EN;
        return true;
    }
    if (lower == "ja" || lower == "jp" || lower == "japanese") {
        out = tts_language::JA;
        return true;
    }
    return false;
}

static const char * tts_cli_language_name(tts_language lang) {
    switch (lang) {
        case tts_language::ZH: return "zh";
        case tts_language::EN: return "en";
        case tts_language::JA: return "ja";
    }
    return "zh";
}

static void tts_cli_print_help_short(const char * argv0) {
    // 说明：面向普通用户的“简洁帮助”，避免把大量调参项一次性全部打印出来。
    // 若需要查看完整/高级参数，请使用 --help-all。
    const char * exe = (argv0 != nullptr && argv0[0] != '\0') ? argv0 : "tts-cli";
    std::fprintf(stdout,
                 "用法：\n"
                 "  %s --model-path <模型.gguf> --prompt \"文本\" [选项]\n"
                 "  %s --model-path <模型.gguf> --bench [选项]\n"
                 "  %s --model-path <模型.gguf> --list-voices\n"
                 "\n"
                 "常用选项：\n"
                 "  -sp, --save-path <out.wav>                 输出 wav 路径（默认：TTS.cpp.wav）\n"
                 "  -v,  --voice <name>                        音色（仅支持 voice packs 的模型）\n"
                 "  -l,  --lang <zh|en|ja>                     语言偏好（默认：zh）\n"
                 "  -d,  --device <cpu|vulkan[:N]|metal|auto>  推理后端/设备（默认：跟随编译配置）\n"
                 "  -nt, --n-threads <N>                       CPU 线程数（默认：硬件并发数）\n"
                 "  -zd, --zh-dict-dir <dir|:builtin|->        中文词典目录（默认自动查找）\n"
                 "  --help-all                                 显示完整/高级参数\n"
                 "\n"
                 "示例：\n"
                 "  %s --model-path model.gguf -p \"好好学习\"\n"
                 "  %s --model-path model.gguf -p \"Hello\" --device cpu\n",
                 exe,
                 exe,
                 exe,
                 exe,
                 exe);
    std::fflush(stdout);
}

static bool tts_cli_is_ascii_space_only(const std::string & s) {
    // 说明：调试输出里避免对纯空白加括号，保持可读性。
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            return false;
        }
    }
    return true;
}

static bool tts_cli_is_ascii_wordish(const std::string & s) {
    // 说明：用于识别英文/数字片段，避免调试输出出现一个字母一个括号的噪声。
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 0x80) {
            return false;
        }
        const bool is_alpha = (uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z');
        const bool is_digit = (uc >= '0' && uc <= '9');
        if (!is_alpha && !is_digit && uc != '_') {
            return false;
        }
    }
    return true;
}

static bool tts_cli_is_ascii_digit_only(const std::string & s) {
    // 说明：用于判断是否为纯数字片段，避免把小数点当作“词内点号”发音。
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (uc < '0' || uc > '9') {
            return false;
        }
    }
    return true;
}

static bool tts_cli_is_dot_symbol(const std::string & s) {
    // 说明：识别 ASCII 点号与全角点号。
    return s == "." || s == "．";
}

static void tts_cli_print_phoneme_debug(tts_generation_runner * runner,
                                        const std::string & prompt,
                                        const generation_configuration & config) {
    // 说明：默认打印“原文 + 音素串 + 分词(词(音素))”，便于快速定位：
    // - 小数点/单位是否被正确归一化；
    // - 多音字是否命中短语词典；
    // - 卷舌/舌尖元音等细节是否生效。
    if (runner == nullptr) {
        return;
    }

    std::string phonemes;
    std::vector<tts_generation_runner::phoneme_segment> segments;
    if (!runner->try_phonemize_segments(prompt.c_str(), phonemes, segments, config)) {
        return;
    }

    tts_cli_log("原文：%s", prompt.c_str());
    tts_cli_log("音素：%s", phonemes.c_str());

    std::string pretty;
    pretty.reserve(phonemes.size() + segments.size() * 6);
    std::string pending_ascii_word;
    std::string pending_ascii_phonemes;
    std::string dot_phonemes;
    std::string at_phonemes;
    auto flush_ascii_word = [&] {
        if (pending_ascii_word.empty()) {
            return;
        }
        pretty.append(pending_ascii_word);
        if (!pending_ascii_phonemes.empty()) {
            pretty.push_back('(');
            pretty.append(pending_ascii_phonemes);
            pretty.push_back(')');
        }
        pending_ascii_word.clear();
        pending_ascii_phonemes.clear();
    };
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto & seg = segments[i];
        if (seg.is_boundary) {
            // 说明：连续英文/数字边界段合并为“词级括号”，标点/空白直接原样输出。
            if (tts_cli_is_ascii_wordish(seg.text)) {
                pending_ascii_word.append(seg.text);
                if (!seg.phonemes.empty()) {
                    pending_ascii_phonemes.append(seg.phonemes);
                } else {
                    pending_ascii_phonemes.append(seg.text);
                }
                continue;
            }
            const bool prev_wordish = (i > 0) &&
                                      segments[i - 1].is_boundary &&
                                      tts_cli_is_ascii_wordish(segments[i - 1].text);
            const bool next_wordish = (i + 1 < segments.size()) &&
                                      segments[i + 1].is_boundary &&
                                      tts_cli_is_ascii_wordish(segments[i + 1].text);
            const bool prev_digit = prev_wordish && tts_cli_is_ascii_digit_only(segments[i - 1].text);
            const bool next_digit = next_wordish && tts_cli_is_ascii_digit_only(segments[i + 1].text);
            const bool should_read_dot = (config.language == tts_language::ZH) &&
                                         tts_cli_is_dot_symbol(seg.text) &&
                                         prev_wordish &&
                                         next_wordish &&
                                         !(prev_digit && next_digit);
            const bool should_read_at = (config.language == tts_language::ZH) && seg.text == "@";

            flush_ascii_word();
            pretty.append(seg.text);
            if (should_read_at) {
                if (at_phonemes.empty()) {
                    at_phonemes = kokoro_zh::text_to_zh_phonemes("艾特", config.zh_dict_dir);
                }
                if (!at_phonemes.empty()) {
                    pretty.push_back('(');
                    pretty.append(at_phonemes);
                    pretty.push_back(')');
                }
            } else if (should_read_dot) {
                if (dot_phonemes.empty()) {
                    dot_phonemes = kokoro_zh::text_to_zh_phonemes("点", config.zh_dict_dir);
                }
                if (!dot_phonemes.empty()) {
                    pretty.push_back('(');
                    pretty.append(dot_phonemes);
                    pretty.push_back(')');
                }
            }
            continue;
        }
        flush_ascii_word();
        pretty.append(seg.text);
        pretty.push_back('(');
        pretty.append(seg.phonemes);
        pretty.push_back(')');
        pretty.push_back(' ');
    }
    flush_ascii_word();
    if (!pretty.empty() && pretty.back() == ' ') {
        pretty.pop_back();
    }
    tts_cli_log("分词：%s", pretty.c_str());
}

class tts_timing_printer {
    const int64_t start_us{[] {
        tts_time_init_once();
        return ggml_time_us();
    }()};
public:
    ~tts_timing_printer() {
        const int64_t end_us{ggml_time_us()};
        // 目前只输出端到端总耗时；后续如需更细粒度可参考 llama_print_timings 的拆分方式。
        fprintf(stderr, "[tts-cli] total time = %.2f ms\n", (end_us - start_us) / 1000.0f);
        fflush(stderr);
    }
};

static int main_impl(int argc, const char ** argv) {
    tts_cli_set_unbuffered();
    tts_cli_log("启动中：开始解析参数/初始化（首次加载模型可能需要较长时间）");

    const tts_timing_printer _{};
    float default_temperature = 1.0f;
    // 默认线程数改为“硬件并发数”（最少 1）。
    // 说明：Kokoro CPU 推理主要由卷积/矩阵运算主导，通常能更好地利用全部逻辑核；
    // 若需降低资源占用，可通过 --n-threads 显式设置较小值。
    int default_n_threads = std::max((int) std::thread::hardware_concurrency(), 1);
    int default_top_k = 50;
    int default_max_tokens = 0;
    float default_repetition_penalty = 1.0f;
    float default_top_p = 1.0f;
    arg_list args;
    args.add_argument(bool_arg("--help-all", "(OPTIONAL) Print full/advanced help and exit."));
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1, Dia, or Kokoro.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED unless --bench) The text prompt for which to generate audio in quotation markers.", "-p", false));
    args.add_argument(bool_arg("--bench", "(OPTIONAL) Use the built-in mixed zh/en benchmark prompt and ignore '--prompt'/'-p'.", "-b"));
    args.add_argument(string_arg("--save-path", "(OPTIONAL) The path to save the audio output to in a .wav format. Defaults to TTS.cpp.wav", "-sp", false, "TTS.cpp.wav"));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 1.0.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware concurrency (min 1). If hardware concurrency cannot be determined then it defaults to 1.", "-nt", false, &default_n_threads));
    args.add_argument(int_arg("--topk", "(OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.", "-r", false, &default_repetition_penalty));
    args.add_argument(string_arg("--device",
                                 "(OPTIONAL) Compute device/backend for testing: cpu/metal/vulkan[:N]/auto. Empty=build default.",
                                 "-d",
                                 false,
                                 ""));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--conditional-prompt", "(OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.", "-cp", false));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
    args.add_argument(string_arg("--voice", "(OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.", "-v", false, ""));
    args.add_argument(string_arg("--lang", "(OPTIONAL) Language preference for digit reading / CJK frontend: zh, en, ja. Defaults to zh.", "-l", false, "zh"));
    args.add_argument(string_arg("--zh-dict-dir",
                                 "(OPTIONAL) Kokoro zh dict directory (pinyin_phrase.txt/pinyin.txt). Empty=auto (try ./dict; else search exe_dir/parents for dict/; then builtin if enabled); ':builtin' forces builtin; '-' disables.",
                                 "-zd",
                                 false,
                                 ""));
    args.add_argument(bool_arg("--list-voices", "(OPTIONAL) List available voices for the selected model and exit.", "-lv"));
    args.add_argument(bool_arg("--vad", "(OPTIONAL) whether to apply voice inactivity detection (VAD) and strip silence form the end of the output (particularly useful for Parler TTS). By default, no VAD is applied.", "-va"));
    args.add_argument(int_arg("--max-tokens", "(OPTIONAL) The max audio tokens or token batches to generate where each represents approximates 11 ms of audio. Only applied to Dia generation. If set to zero as is its default then the default max generation size. Warning values under 15 are not supported.", "-mt", false, &default_max_tokens));
    args.add_argument(float_arg("--top-p", "(OPTIONAL) the sum of probabilities to sample over. Must be a value between 0.0 and 1.0. Defaults to 1.0.", "-tp", false, &default_top_p));
    register_play_tts_response_args(args);

    const int64_t t_parse_start_us = ggml_time_us();
    args.parse(argc, argv);
    if (args.for_help) {
        tts_cli_log("检测到 --help：打印简洁帮助并退出（如需完整参数用 --help-all）");
        tts_cli_print_help_short((argc > 0) ? argv[0] : nullptr);
        exit(0);
    }
    if (args.get_bool_param("--help-all")) {
        tts_cli_log("检测到 --help-all：打印完整帮助并退出");
        args.help();
        exit(0);
    }
    args.validate();
    const int64_t t_parse_end_us = ggml_time_us();
    tts_cli_log("参数解析完成：%.2f ms", tts_cli_us_to_ms(t_parse_end_us - t_parse_start_us));

    const bool bench = args.get_bool_param("--bench");
    if (!args.get_bool_param("--list-voices") && !bench && args.get_string_param("--prompt").empty()) {
        fprintf(stderr, "argument '--prompt' is required unless '--bench' is set.\n");
        exit(1);
    }

    std::string conditional_prompt = args.get_string_param("--conditional-prompt");
    std::string text_encoder_path = args.get_string_param("--text-encoder-path");
    if (conditional_prompt.size() > 0 && text_encoder_path.size() <= 0) {
        fprintf(stderr, "The '--text-encoder-path' must be specified when '--condtional-prompt' is passed.\n");
        exit(1);
    }

    if (*args.get_float_param("--top-p") > 1.0f || *args.get_float_param("--top-p") <= 0.0f) {
        fprintf(stderr, "The '--top-p' value must be between 0.0 and 1.0. It was set to '%.6f'.\n", *args.get_float_param("--top-p"));
        exit(1);
    }

    tts_language language = tts_language::ZH;
    if (!tts_cli_parse_language(args.get_string_param("--lang"), language)) {
        fprintf(stderr, "The '--lang' value must be 'zh', 'en' or 'ja'. It was set to '%s'.\n",
                args.get_string_param("--lang").c_str());
        exit(1);
    }

    const std::string                zh_dict_dir_raw = args.get_string_param("--zh-dict-dir");
    const tts_cli_zh_dict_resolution zh_dict_dir = tts_cli_resolve_zh_dict_dir(zh_dict_dir_raw, (argc > 0) ? argv[0] : nullptr);

    const generation_configuration config{
        args.get_string_param("--voice"),
        *args.get_int_param("--topk"),
        *args.get_float_param("--temperature"),
        *args.get_float_param("--repetition-penalty"),
        !args.get_bool_param("--no-cross-attn"),
        *args.get_int_param("--max-tokens"),
        *args.get_float_param("--top-p"),
        true,
        language,
        zh_dict_dir.dir};

    // ----------------------------
    // 选择推理后端（CPU / Metal / Vulkan）
    // ----------------------------
    // 说明：
    // - `--device` 仅用于测试/对比，日常用户无需关注；
    // - 未指定时，默认值跟随“编译时启用的后端”（编译启用 Vulkan 则默认 Vulkan，启用 Metal 则默认 Metal）；
    const int default_vulkan_device = 0;
    tts_backend_config backend = tts_cli_default_backend_config(default_vulkan_device);

    const std::string device_arg = args.get_string_param("--device");
    if (!device_arg.empty()) {
        if (!tts_cli_parse_device_arg(device_arg, default_vulkan_device, backend)) {
            fprintf(stderr, "argument '--device' must be one of: cpu/metal/vulkan[:N]/auto.\n");
            exit(1);
        }
    } else {
        backend = tts_cli_default_backend_config(default_vulkan_device);
    }

    {
        const bool        list_voices = args.get_bool_param("--list-voices");
        const std::string model_path  = args.get_string_param("--model-path");
        const std::string save_path   = args.get_string_param("--save-path");
        std::string device_name;
        switch (backend.backend) {
            case tts_compute_backend::CPU:    device_name = "cpu"; break;
            case tts_compute_backend::METAL:  device_name = "metal"; break;
            case tts_compute_backend::VULKAN: device_name = "vulkan:" + std::to_string(backend.device); break;
            case tts_compute_backend::AUTO:   device_name = "auto"; break;
        }

        uint64_t model_bytes = 0;
        const bool has_size = !model_path.empty() && model_path.rfind("test:", 0) != 0 &&
                              tts_cli_try_file_size(model_path, model_bytes);

        tts_cli_log("运行配置：threads=%d device=%s vad=%s list_voices=%s lang=%s zh_dict=%s",
                    *args.get_int_param("--n-threads"),
                    device_name.c_str(),
                    args.get_bool_param("--vad") ? "on" : "off",
                    list_voices ? "yes" : "no",
                    tts_cli_language_name(language),
                    zh_dict_dir.desc.c_str());
        if (has_size) {
            tts_cli_log("模型文件：%s (%.2f MiB)", model_path.c_str(), model_bytes / 1024.0 / 1024.0);
        } else {
            tts_cli_log("模型文件：%s", model_path.c_str());
        }
        if (!list_voices) {
            tts_cli_log("输出：%s%s", save_path.c_str(), args.get_bool_param("--play") ? "（如启用 --play 则优先播放）" : "");
        }
    }

    const int64_t t_load_start_us = ggml_time_us();
    tts_cli_log("开始加载模型（GGUF）...");

    unique_ptr<tts_generation_runner> runner{runner_from_file(args.get_string_param("--model-path").c_str(),
                                                              *args.get_int_param("--n-threads"),
                                                              config,
                                                              backend)};
    const int64_t t_load_end_us = ggml_time_us();
    tts_cli_log("模型加载完成：arch=%s sampling_rate=%.0fHz supports_voices=%s (%.2f ms)",
                runner->loader.get().arch,
                runner->sampling_rate,
                runner->supports_voices ? "yes" : "no",
                tts_cli_us_to_ms(t_load_end_us - t_load_start_us));

    if (args.get_bool_param("--list-voices")) {
        if (!runner->supports_voices) {
            fprintf(stderr, "The selected model does not support voices.\n");
            exit(1);
        }
        tts_cli_log("开始列出可用音色...");
        size_t voice_count = 0;
        for (const auto & v : runner->list_voices()) {
            printf("%.*s\n", (int) v.size(), v.data());
            ++voice_count;
        }
        tts_cli_log("音色列表输出完成：%zu 个", voice_count);
        exit(0);
    }

    if (!conditional_prompt.empty()) {
        const int64_t t_cp_start_us = ggml_time_us();
        tts_cli_log("开始更新 conditional prompt（text_encoder=%s prompt_bytes=%zu）...",
                    text_encoder_path.c_str(),
                    conditional_prompt.size());
        runner->update_conditional_prompt(text_encoder_path.c_str(), conditional_prompt.c_str());
        const int64_t t_cp_end_us = ggml_time_us();
        tts_cli_log("conditional prompt 更新完成：%.2f ms", tts_cli_us_to_ms(t_cp_end_us - t_cp_start_us));
    }
    tts_response data;

    // 说明：--bench 的设计目标是“一键用固定文本跑通一次推理”，因此这里强制覆盖 prompt，避免 -p/--prompt 影响基准对比的一致性。
    const std::string prompt = bench ? std::string(tts_cli_bench_prompt()) : args.get_string_param("--prompt");

    const int64_t t_gen_start_us = ggml_time_us();
    tts_cli_log("开始生成语音：voice=\"%s\" top_k=%d top_p=%.3f temp=%.3f rep_pen=%.3f",
                config.voice.c_str(),
                config.top_k,
                config.top_p,
                config.temperature,
                config.repetition_penalty);

    // 默认打印音素调试信息（stderr），不影响生成的 wav 文件。
    tts_cli_print_phoneme_debug(runner.get(), prompt, config);
    runner->generate(prompt.c_str(), data, config);
    const int64_t t_gen_end_us = ggml_time_us();
    if (data.n_outputs > 0 && runner->sampling_rate > 0.0f) {
        const double audio_s = static_cast<double>(data.n_outputs) / runner->sampling_rate;
        tts_cli_log("生成完成：samples=%zu audio=%.3fs (%.2f ms)", data.n_outputs, audio_s, tts_cli_us_to_ms(t_gen_end_us - t_gen_start_us));
    } else {
        tts_cli_log("生成完成：samples=%zu (%.2f ms)", data.n_outputs, tts_cli_us_to_ms(t_gen_end_us - t_gen_start_us));
    }
    if (data.n_outputs == 0) {
        fprintf(stderr, "Got empty response for prompt, '%s'.\n", prompt.c_str());
        exit(1);
    }
    if (args.get_bool_param("--vad")) {
        const int64_t t_vad_start_us = ggml_time_us();
        tts_cli_log("应用 VAD：去除尾部静音...");
        apply_energy_voice_inactivity_detection(data, runner->sampling_rate);
        const int64_t t_vad_end_us = ggml_time_us();
        tts_cli_log("VAD 完成：%.2f ms", tts_cli_us_to_ms(t_vad_end_us - t_vad_start_us));
    }

    const bool played = play_tts_response(args, data, runner->sampling_rate);
    if (played) {
        tts_cli_log("播放完成");
    } else {
        tts_cli_log("开始写入 wav 文件...");
        write_audio_file(data, args.get_string_param("--save-path"), runner->sampling_rate);
    }
    static_cast<void>(!runner.release()); // TODO：runner 的析构目前仍不稳定，暂时 release 避免异常退出
    return 0;
}

int main(int argc, const char ** argv) {
#ifdef _WIN32
    // Windows 下确保 argv 为 UTF-8，避免中文等非 ASCII prompt 解析成乱码。
    int wargc = 0;
    wchar_t ** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv || wargc <= 0) {
        return main_impl(argc, argv);
    }

    std::vector<std::string> argv_utf8_storage;
    argv_utf8_storage.reserve(wargc);

    std::vector<const char *> argv_utf8;
    argv_utf8.reserve(wargc);

    for (int i = 0; i < wargc; ++i) {
        const wchar_t * warg = wargv[i] ? wargv[i] : L"";
        const int       size = WideCharToMultiByte(CP_UTF8, 0, warg, -1, nullptr, 0, nullptr, nullptr);
        std::string     arg;
        if (size > 0) {
            arg.resize(static_cast<size_t>(size));
            WideCharToMultiByte(CP_UTF8, 0, warg, -1, arg.data(), size, nullptr, nullptr);
            if (!arg.empty() && arg.back() == '\0') {
                arg.pop_back();
            }
        }
        argv_utf8_storage.push_back(std::move(arg));
        argv_utf8.push_back(argv_utf8_storage.back().c_str());
    }

    LocalFree(wargv);
    return main_impl(wargc, argv_utf8.data());
#else
    return main_impl(argc, argv);
#endif
}
