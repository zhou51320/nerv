#include "loaders.h"

#include <cstring>
#include <cstdint>
#include <unordered_map>

#include "../util.h"
#include "common.h"
#include "ggml-iterator.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-mmap.h"

namespace {
    using loader_map = unordered_map<string_view, reference_wrapper<const tts_model_loader>>;

    loader_map & get_loader_registry() {
        // 重要：使用“函数内静态变量”来避免静态初始化顺序问题（static initialization order fiasco）。
        //
        // 现象：在 MinGW/GCC 下，不同翻译单元（.cpp）中的全局/静态对象初始化顺序是不确定的。
        // 某些模型的 loader 以全局对象形式存在（例如 dia_loader 等），它们在构造时会向注册表写入。
        // 如果注册表本身也是全局对象，就可能出现“loader 先构造、注册表后构造”的情况，导致 UB，
        // 常见表现就是在 unordered_map::emplace 内部触发除零而崩溃（GDB 显示 SIGFPE）。
        static loader_map registry;
        return registry;
    }
}  // namespace

tts_model_loader::tts_model_loader(const char * arch, bool is_test) : arch{ arch }, is_test{ is_test } {
    auto & loaders = get_loader_registry();
    loaders.emplace(arch, ref(*this));
}

void dia_register();
void dummy_register();
void kokoro_register();
void orpheus_register();
void parler_register();

[[maybe_unused]] static bool loaders = [] {
    dia_register();
    dummy_register();
    kokoro_register();
    orpheus_register();
    parler_register();
    return true;
}();

// 说明：目前 ggml 已支持多种后端（如 CPU / Metal / Vulkan）。
// 历史参数 cpu_only 仅表示是否强制使用 CPU；当 cpu_only=false 时，会根据当前线程的后端配置选择加速后端。
static unique_ptr<tts_generation_runner> runner_from_file_impl(const char * fname, int n_threads,
                                                               const generation_configuration & config, bool cpu_only) {
    auto &     loaders = get_loader_registry();
    string_view fname_sv{ fname };
    if (fname_sv.starts_with("test:")) {
        fname_sv.remove_prefix(sizeof("test:") - 1);
        const auto found{loaders.find(fname_sv)};
        if (found == loaders.end()) {
            GGML_ABORT("Unknown test model/backend %s\n", fname);
        }
        return found->second.get().from_file(nullptr, nullptr, 0, false, config);
    }
    static const bool use_mmap{ !getenv("OLLAMA_NO_MMAP") };  // TODO(danielzgtg) temporary, will be --no-mmap later
    unique_ptr<llama_mmap> in_mmap{};
    if (use_mmap) {
        llama_file in_map_file{ fname, "r" };
        in_mmap = make_unique<llama_mmap>(&in_map_file);
    }
    ggml_context * weight_ctx{};
    gguf_context * meta_ctx = gguf_init_from_file(fname, {
                                                             .no_alloc{ use_mmap },
                                                             .ctx{ &weight_ctx },
                                                         });
    if (!meta_ctx) {
        GGML_ABORT("gguf_init_from_file failed for file %s\n", fname);
    }
    if (use_mmap) {
        const int64_t n{ gguf_get_n_tensors(&*meta_ctx) };
        int64_t       i{};
        void *    in_buffer{ static_cast<char *>(in_mmap->addr()) + gguf_get_data_offset(meta_ctx) };
        for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
            GGML_ASSERT(i < n);
            GGML_ASSERT(!strcmp(cur.name, gguf_get_tensor_name(&*meta_ctx, static_cast<int>(i))));
            cur.data = static_cast</*const*/ char *>(in_buffer) + gguf_get_tensor_offset(&*meta_ctx, static_cast<int>(i));
            ++i;
        }
    }
    const int          arch_key = gguf_find_key(meta_ctx, "general.architecture");
    const char * const arch{ gguf_get_val_str(meta_ctx, arch_key) };
    const auto         found = loaders.find(arch);
    if (found == loaders.end()) {
        GGML_ABORT("Unknown architecture %s\n", arch);
    }
    const auto &                      loader{ found->second.get() };
    unique_ptr<tts_generation_runner> runner{ loader.from_file(meta_ctx, weight_ctx, n_threads, cpu_only, config) };
    // TODO(mmwillet): change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor & cur : ggml_tensor_iterator{ *weight_ctx }) {
        if (!cur.data) {
            continue;
        }
        if (!*cur.name) {
            // handles the top level meta tensor
            continue;
        }
        runner->assign_weight(cur.name, cur);
    }
    runner->prepare_post_load();
    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    GGML_ASSERT(&runner->loader.get() == &loader);
    runner->buf = move(in_mmap);
    return runner;
}

unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads,
                                                   const generation_configuration & config, bool cpu_only) {
    // 兼容旧接口：cpu_only=false 时，默认尝试自动选择可用的 GPU 后端（优先 Metal，其次 Vulkan）。
    tts_backend_config backend{};
    backend.backend = cpu_only ? tts_compute_backend::CPU : tts_compute_backend::AUTO;
    backend.device  = 0;
    return runner_from_file(fname, n_threads, config, backend);
}

unique_ptr<tts_generation_runner> runner_from_file(const char * fname, int n_threads,
                                                   const generation_configuration & config,
                                                   const tts_backend_config & backend) {
    // 说明：通过 guard 保证同一线程内的子模型加载（如 Parler 的 T5 encoder）使用一致的后端选择。
    tts_backend_config_guard guard{backend};
    const bool cpu_only = backend.backend == tts_compute_backend::CPU;
    return runner_from_file_impl(fname, n_threads, config, cpu_only);
}
