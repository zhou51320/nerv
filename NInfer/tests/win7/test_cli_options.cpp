#include "apps/cli/options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void rejects(std::vector<std::string> arguments, const char* expected) {
    try {
        (void)parse(std::move(arguments));
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "wrong argument rejection");
        return;
    }
    throw std::runtime_error("unsupported option was accepted");
}

} // namespace

int main() {
    try {
        const auto options = parse({"ninfer", "model.ninfer", "--prompt", "Hello"});
        require(!options.use_cuda_graph, "Win7 must default to no CUDA Graph");
        require(options.speculative.backend == ninfer::SpeculativeBackend::None,
                "Win7 must default to no speculative decoding");
        require(!options.enable_vision, "Win7 must default to text only");
        require(options.max_context == 2048 && options.kv_capacity.explicit_tokens == 2048,
                "unexpected baseline context capacity");
        require(options.prefill_chunk == 256, "unexpected baseline prefill chunk");
        require(options.kv_cache == ninfer::KvCacheStorage::Int8Group64,
                "unexpected baseline KV format");
        const ninfer::EngineOptions engine;
        require(!engine.use_cuda_graph && engine.max_concurrency == 1,
                "Engine defaults disagree with baseline CLI");
        require(engine.prefill_chunk == options.prefill_chunk && engine.kv_cache == options.kv_cache,
                "Engine allocation defaults disagree with CLI");

        const auto custom = parse({"ninfer", "model.ninfer", "--messages", "text.json",
                                   "--max-context", "4096", "--prefill-chunk", "128",
                                   "--kv-dtype", "bf16", "--greedy", "--no-cuda-graph"});
        require(custom.kv_capacity.explicit_tokens == 4096 && custom.prefill_chunk == 128,
                "explicit text configuration was not preserved");
        require(custom.kv_cache == ninfer::KvCacheStorage::BFloat16 &&
                    custom.sampling.temperature == 0.0F,
                "explicit sampling/KV configuration was not preserved");
        require(parse({"ninfer", "--help"}).help_requested, "help requires a model");
        rejects({"ninfer", "model.ninfer", "--prompt", "Hello", "--vision"}, "text-only");
        rejects({"ninfer", "model.ninfer", "--prompt", "Hello", "--spec", "mtp",
                 "--draft-tokens", "3"}, "Win7 baseline");
        rejects({"ninfer", "model.ninfer", "--prompt", "Hello", "--spec", "dflash",
                 "--draft-tokens", "3"}, "Win7 baseline");
        rejects({"ninfer", "model.ninfer", "--prompt", "Hello", "--prefill-chunk", "17"},
                "multiple of 128");
        const auto help = ninfer::cli::usage_text("ninfer");
        require(help.find("Win7 baseline") != std::string::npos, "help omits the build profile");
        require(help.find("--vision") == std::string::npos, "text-only help advertises vision");
        require(help.find("--spec mtp") == std::string::npos, "baseline help advertises MTP");
        std::cout << "Win7 CLI baseline tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
