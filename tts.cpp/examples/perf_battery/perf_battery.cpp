#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "../../src/models/loaders.h"
#include "args.h"
#include "common.h"

namespace {

// 基准测试用例（以 Kokoro 为主，覆盖中/英/混合场景）
struct bench_case {
    const char * name;
    const char * text;
};

const std::vector<bench_case> kBenchCases = {
    { "zh_short",  "你好，欢迎使用本地语音合成。" },
    { "zh_medium", "今天的天气很好，我们一起去公园散步，然后喝一杯热茶。" },
    { "zh_long",   "这是一个较长的中文句子，用来测试在本地设备上进行语音合成时的性能与稳定性。" },
    { "en_short",  "Hello, welcome to the offline TTS benchmark." },
    { "en_medium", "The quick brown fox jumps over the lazy dog while the rain taps the window." },
    { "mix",       "你好 world, this is a mixed language test for latency and clarity." },
};

// 统一的时间测量（ms）
double now_ms() {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

double mean(const std::vector<double> & series) {
    if (series.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : series) {
        sum += v;
    }
    return sum / static_cast<double>(series.size());
}

// 简单的分位数计算（p 在 [0, 1]）
double percentile(std::vector<double> series, double p) {
    if (series.empty()) {
        return 0.0;
    }
    if (p <= 0.0) {
        return *std::min_element(series.begin(), series.end());
    }
    if (p >= 1.0) {
        return *std::max_element(series.begin(), series.end());
    }
    std::sort(series.begin(), series.end());
    const double idx = p * (series.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(lo + 1, series.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return series[lo] * (1.0 - frac) + series[hi] * frac;
}

// 为了保持 benchmark 结果简洁，如未指定则自动关闭内部 timings 输出
void bench_disable_internal_timings() {
    const char * v = std::getenv("TTS_TIMINGS");
    if (v && v[0] != '\0') {
        return;
    }
#ifdef _WIN32
    _putenv_s("TTS_TIMINGS", "0");
#else
    setenv("TTS_TIMINGS", "0", 1);
#endif
}

std::string to_lower(std::string v) {
    for (char & c : v) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return v;
}

bool parse_backend(const std::string & backend_name, int vulkan_device, tts_backend_config & out) {
    const std::string name = to_lower(backend_name);
    out.device = vulkan_device;
    if (name == "cpu") {
        out.backend = tts_compute_backend::CPU;
        return true;
    }
    if (name == "metal") {
        out.backend = tts_compute_backend::METAL;
        return true;
    }
    if (name == "vulkan") {
        out.backend = tts_compute_backend::VULKAN;
        return true;
    }
    if (name == "auto") {
        out.backend = tts_compute_backend::AUTO;
        return true;
    }
    return false;
}

std::string backend_to_string(const tts_backend_config & cfg) {
    switch (cfg.backend) {
        case tts_compute_backend::CPU:
            return "cpu";
        case tts_compute_backend::METAL:
            return "metal";
        case tts_compute_backend::VULKAN:
            return "vulkan:" + std::to_string(cfg.device);
        case tts_compute_backend::AUTO:
            return "auto";
    }
    return "unknown";
}

struct case_stats {
    double sum_ms = 0.0;
    double sum_audio_s = 0.0;
    int count = 0;
};

} // namespace

int main(int argc, const char ** argv) {
    bench_disable_internal_timings();

    int default_n_threads = std::max((int) std::thread::hardware_concurrency(), 1);
    int default_repeat = 1;
    int default_warmup = 1;
    int default_vulkan_device = 0;
    arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file.", "-mp", true));
    args.add_argument(string_arg("--voice", "(OPTIONAL) The voice to use to generate the audio (voice pack models only).", "-v", false, ""));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware concurrency.", "-nt", false, &default_n_threads));
    args.add_argument(string_arg("--backend", "(OPTIONAL) cpu/metal/vulkan/auto. Defaults to cpu.", "-b", false, "cpu"));
    args.add_argument(int_arg("--vulkan-device", "(OPTIONAL) Vulkan device index (default: 0).", "-vd", false, &default_vulkan_device));
    args.add_argument(int_arg("--repeat", "(OPTIONAL) Repeat count per case. Defaults to 1.", "-r", false, &default_repeat));
    args.add_argument(int_arg("--warmup", "(OPTIONAL) Warmup count before benchmarking. Defaults to 1.", "-w", false, &default_warmup));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();

    if (*args.get_int_param("--repeat") < 1) {
        fprintf(stderr, "argument '--repeat' must be >= 1.\n");
        return 1;
    }
    if (*args.get_int_param("--warmup") < 0) {
        fprintf(stderr, "argument '--warmup' must be >= 0.\n");
        return 1;
    }

    tts_backend_config backend{};
    if (!parse_backend(args.get_string_param("--backend"), *args.get_int_param("--vulkan-device"), backend)) {
        fprintf(stderr, "argument '--backend' must be one of: cpu/metal/vulkan/auto.\n");
        return 1;
    }

    // 使用一套固定的 generation 配置，保证不同后端/版本可直接对比
    const generation_configuration config{
        args.get_string_param("--voice"),
        50,
        1.0f,
        1.0f,
        true,
        0,
        1.0f,
        true
    };

    const double load_start_ms = now_ms();
    unique_ptr<tts_generation_runner> runner{runner_from_file(
        args.get_string_param("--model-path").c_str(),
        *args.get_int_param("--n-threads"),
        config,
        backend)};
    const double load_end_ms = now_ms();

    const double load_ms = load_end_ms - load_start_ms;
    const double sampling_rate = runner->sampling_rate > 0.0f ? runner->sampling_rate : 44100.0f;

    fprintf(stdout, "tts-bench\n");
    fprintf(stdout, "arch: %s\n", runner->loader.get().arch);
    fprintf(stdout, "backend: %s\n", backend_to_string(backend).c_str());
    fprintf(stdout, "threads: %d\n", *args.get_int_param("--n-threads"));
    fprintf(stdout, "voice: %s\n", config.voice.empty() ? "(default)" : config.voice.c_str());
    fprintf(stdout, "cases: %zu, repeat: %d, warmup: %d\n", kBenchCases.size(), *args.get_int_param("--repeat"), *args.get_int_param("--warmup"));
    fprintf(stdout, "load_ms: %.2f\n", load_ms);

    // warmup：不记录结果，减少头部冷启动时间影响
    for (int i = 0; i < *args.get_int_param("--warmup"); ++i) {
        const bench_case & bc = kBenchCases[static_cast<size_t>(i) % kBenchCases.size()];
        tts_response warmup_resp{};
        runner->generate(bc.text, warmup_resp, config);
    }

    std::vector<case_stats> per_case(kBenchCases.size());
    std::vector<double> all_ms;
    std::vector<double> all_rtf;
    double total_ms = 0.0;
    double total_audio_s = 0.0;
    int total_runs = 0;

    for (int r = 0; r < *args.get_int_param("--repeat"); ++r) {
        for (size_t i = 0; i < kBenchCases.size(); ++i) {
            const bench_case & bc = kBenchCases[i];
            tts_response response{};
            const double t_start = now_ms();
            runner->generate(bc.text, response, config);
            const double t_end = now_ms();
            if (response.n_outputs == 0) {
                fprintf(stderr, "empty response for case '%s'.\n", bc.name);
                return 1;
            }

            const double gen_ms = t_end - t_start;
            const double audio_s = static_cast<double>(response.n_outputs) / sampling_rate;
            const double rtf = audio_s > 0.0 ? (gen_ms / 1000.0) / audio_s : 0.0;

            per_case[i].sum_ms += gen_ms;
            per_case[i].sum_audio_s += audio_s;
            per_case[i].count += 1;

            all_ms.push_back(gen_ms);
            all_rtf.push_back(rtf);
            total_ms += gen_ms;
            total_audio_s += audio_s;
            ++total_runs;
        }
    }

    fprintf(stdout, "\n%-12s %12s %10s %8s\n", "case", "time_ms", "audio_s", "rtf");
    for (size_t i = 0; i < kBenchCases.size(); ++i) {
        const case_stats & s = per_case[i];
        const double avg_ms = s.count > 0 ? s.sum_ms / s.count : 0.0;
        const double avg_audio = s.count > 0 ? s.sum_audio_s / s.count : 0.0;
        const double avg_rtf = avg_audio > 0.0 ? (avg_ms / 1000.0) / avg_audio : 0.0;
        fprintf(stdout, "%-12s %12.2f %10.3f %8.3f\n", kBenchCases[i].name, avg_ms, avg_audio, avg_rtf);
    }

    const double mean_ms = mean(all_ms);
    const double p50_ms = percentile(all_ms, 0.50);
    const double p90_ms = percentile(all_ms, 0.90);
    const double mean_rtf = mean(all_rtf);
    const double total_rtf = total_audio_s > 0.0 ? (total_ms / 1000.0) / total_audio_s : 0.0;

    fprintf(stdout, "\nsummary\n");
    fprintf(stdout, "runs: %d\n", total_runs);
    fprintf(stdout, "total_ms: %.2f\n", total_ms);
    fprintf(stdout, "total_audio_s: %.3f\n", total_audio_s);
    fprintf(stdout, "mean_ms: %.2f  p50_ms: %.2f  p90_ms: %.2f\n", mean_ms, p50_ms, p90_ms);
    fprintf(stdout, "mean_rtf: %.3f  total_rtf: %.3f\n", mean_rtf, total_rtf);

    static_cast<void>(!runner.release()); // TODO：runner 析构目前仍不稳定，暂时 release 避免异常退出
    return 0;
}
