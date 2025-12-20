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

static double tts_cli_us_to_ms(const int64_t us) {
    return us / 1000.0;
}

static std::string tts_cli_to_lower(std::string v) {
    for (char & c : v) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return v;
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
    // 默认线程数改为“硬件并发的一半”（最少 1）。
    // 说明：很多 CPU 的 hardware_concurrency() 返回的是“逻辑核数”（含超线程），
    // 直接拉满线程在部分平台上反而会增加调度/缓存竞争开销，端到端延迟更差；
    // 取一半通常更接近“物理核数”，对推理更稳。
    int default_n_threads = std::max((int) std::thread::hardware_concurrency() / 2, 1);
    int default_top_k = 50;
    int default_max_tokens = 0;
    float default_repetition_penalty = 1.0f;
    float default_top_p = 1.0f;
    int default_vulkan_device = 0;
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1, Dia, or Kokoro.", "-mp", true));
    args.add_argument(string_arg("--prompt", "(REQUIRED) The text prompt for which to generate audio in quotation markers.", "-p", false));
    args.add_argument(string_arg("--save-path", "(OPTIONAL) The path to save the audio output to in a .wav format. Defaults to TTS.cpp.wav", "-sp", false, "TTS.cpp.wav"));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 1.0.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to half of hardware concurrency (min 1). If hardware concurrency cannot be determined then it defaults to 1.", "-nt", false, &default_n_threads));
    args.add_argument(int_arg("--topk", "(OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.", "-r", false, &default_repetition_penalty));
    args.add_argument(string_arg("--device",
                                 "(OPTIONAL) Compute device/backend for testing: cpu/metal/vulkan[:N]/auto. Empty=build default.",
                                 "-d",
                                 false,
                                 ""));
    // 兼容旧参数（已弃用）：建议统一使用 --device
    args.add_argument(bool_arg("--use-metal", "(DEPRECATED) Use '--device metal' instead.", "-m"));
    args.add_argument(bool_arg("--use-vulkan", "(DEPRECATED) Use '--device vulkan' instead.", "-vk"));
    args.add_argument(int_arg("--vulkan-device", "(OPTIONAL) Vulkan device index (default: 0).", "-vd", false, &default_vulkan_device));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--conditional-prompt", "(OPTIONAL) A distinct conditional prompt to use for generating. If none is provided the preencoded prompt is used. '--text-encoder-path' must be set to use conditional generation.", "-cp", false));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
    args.add_argument(string_arg("--voice", "(OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.", "-v", false, ""));
    args.add_argument(string_arg("--lang", "(OPTIONAL) Language preference for digit reading / CJK frontend: zh, en, ja. Defaults to zh.", "-l", false, "zh"));
    args.add_argument(string_arg("--zh-dict-dir",
                                 "(OPTIONAL) Kokoro zh dict directory (pinyin_phrase.txt/pinyin.txt). Empty=auto (try ./dict then builtin if enabled); ':builtin' forces builtin; '-' disables.",
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
        tts_cli_log("检测到 --help：打印帮助并退出");
        args.help();
        exit(0);
    }
    args.validate();
    const int64_t t_parse_end_us = ggml_time_us();
    tts_cli_log("参数解析完成：%.2f ms", tts_cli_us_to_ms(t_parse_end_us - t_parse_start_us));

    if (!args.get_bool_param("--list-voices") && args.get_string_param("--prompt").empty()) {
        fprintf(stderr, "argument '--prompt' is required.\n");
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
        args.get_string_param("--zh-dict-dir")};

    // ----------------------------
    // 选择推理后端（CPU / Metal / Vulkan）
    // ----------------------------
    // 说明：
    // - `--device` 仅用于测试/对比，日常用户无需关注；
    // - 未指定时，默认值跟随“编译时启用的后端”（编译启用 Vulkan 则默认 Vulkan，启用 Metal 则默认 Metal）；
    // - 兼容旧参数：--use-metal / --use-vulkan（已弃用）。
    tts_backend_config backend{};
    backend.device = *args.get_int_param("--vulkan-device");

    const std::string device_arg = args.get_string_param("--device");
    const bool        use_metal  = args.get_bool_param("--use-metal");
    const bool        use_vulkan = args.get_bool_param("--use-vulkan");

    if (!device_arg.empty()) {
        if (use_metal || use_vulkan) {
            fprintf(stderr, "参数冲突：--device 与 --use-metal/--use-vulkan 不能同时使用。\n");
            exit(1);
        }
        if (!tts_cli_parse_device_arg(device_arg, backend.device, backend)) {
            fprintf(stderr, "argument '--device' must be one of: cpu/metal/vulkan[:N]/auto.\n");
            exit(1);
        }
    } else {
        if (use_metal && use_vulkan) {
            fprintf(stderr, "参数冲突：--use-metal 与 --use-vulkan 不能同时使用。\n");
            exit(1);
        }
        if (use_metal) {
            backend.backend = tts_compute_backend::METAL;
        } else if (use_vulkan) {
            backend.backend = tts_compute_backend::VULKAN;
        } else {
            backend = tts_cli_default_backend_config(backend.device);
        }
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

        const std::string zh_dict_dir = args.get_string_param("--zh-dict-dir");
        std::string       zh_dict_desc;
        if (zh_dict_dir.empty()) {
            zh_dict_desc = "(auto: ./dict -> builtin)";
        } else if (zh_dict_dir == "-") {
            zh_dict_desc = "(disabled)";
        } else if (zh_dict_dir == ":builtin") {
            zh_dict_desc = "(builtin)";
        } else {
            zh_dict_desc = zh_dict_dir;
        }
        tts_cli_log("运行配置：threads=%d device=%s vad=%s list_voices=%s lang=%s zh_dict=%s",
                    *args.get_int_param("--n-threads"),
                    device_name.c_str(),
                    args.get_bool_param("--vad") ? "on" : "off",
                    list_voices ? "yes" : "no",
                    tts_cli_language_name(language),
                    zh_dict_desc.c_str());
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

    const int64_t t_gen_start_us = ggml_time_us();
    tts_cli_log("开始生成语音：voice=\"%s\" top_k=%d top_p=%.3f temp=%.3f rep_pen=%.3f",
                config.voice.c_str(),
                config.top_k,
                config.top_p,
                config.temperature,
                config.repetition_penalty);
    runner->generate(args.get_string_param("--prompt").c_str(), data, config);
    const int64_t t_gen_end_us = ggml_time_us();
    if (data.n_outputs > 0 && runner->sampling_rate > 0.0f) {
        const double audio_s = static_cast<double>(data.n_outputs) / runner->sampling_rate;
        tts_cli_log("生成完成：samples=%zu audio=%.3fs (%.2f ms)", data.n_outputs, audio_s, tts_cli_us_to_ms(t_gen_end_us - t_gen_start_us));
    } else {
        tts_cli_log("生成完成：samples=%zu (%.2f ms)", data.n_outputs, tts_cli_us_to_ms(t_gen_end_us - t_gen_start_us));
    }
    if (data.n_outputs == 0) {
        fprintf(stderr, "Got empty response for prompt, '%s'.\n", args.get_string_param("--prompt").c_str());
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
