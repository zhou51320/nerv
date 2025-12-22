#include "tts_model.h"
#include "llama-mmap.h"

#include <cstdlib>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "models/loaders.h"

namespace {

static bool tts_env_truthy_local(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

static bool tts_env_truthy_default_local(const char * name, bool default_value) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return default_value;
    }
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 && std::strcmp(v, "false") != 0;
}

static double tts_env_double_default_local(const char * name, double default_value) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return default_value;
    }
    char * end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v) {
        return default_value;
    }
    return x;
}

static bool tts_backend_is_vulkan_local(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    const char * name = ggml_backend_name(backend);
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const std::string_view sv{name};
    // 说明：ggml 的 Vulkan backend 名称通常以 "Vulkan" / "vulkan" 开头（可能带设备编号）。
    return tts_starts_with(sv, "Vulkan") || tts_starts_with(sv, "vulkan");
}

} // namespace

void append_to_response(tts_response & response, tts_response & to_append) {
    float * new_data = (float *) malloc((response.n_outputs + to_append.n_outputs) * sizeof(float));
    if (response.n_outputs > 0) {
        std::memcpy(new_data, response.data, response.n_outputs*sizeof(float));
    }
    if (to_append.n_outputs > 0) {
        float * next_loc = new_data + response.n_outputs;
        std::memcpy(next_loc, to_append.data, to_append.n_outputs*sizeof(float));
    }
    response.data = new_data;
    response.n_outputs += to_append.n_outputs;
}

/*
 * Pulls output_size to prepped buffer 'output' from 'output_node' tensor. If no buffer is passed will default to the existing output buffer present
 * on runner_context.
 */
void runner_context::get_ggml_node_data(struct ggml_tensor * output_node, float * output, size_t output_size, ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        buffer = buf_output;
    }
    if (ggml_backend_buffer_get_size(buffer) < output_size) {
        TTS_ABORT("Output buffer overflow of %d / %d for output node '%s'\n", output_size, ggml_backend_buffer_get_size(buffer), ggml_get_name(output_node));
    } else if (ggml_nbytes(output_node) < output_size) {
        TTS_ABORT("Output node, '%s', with %d bytes is too small for #ggml_backend_tensor_get_async with size of %d.\n", ggml_get_name(output_node), ggml_nbytes(output_node), output_size);
    }
    ggml_backend_t backend_res = nullptr;
    if (sched) {
        backend_res = ggml_backend_sched_get_tensor_backend(sched, output_node);
    } else {
        // 说明：CPU-only 使用 galloc 时，不存在 sched；输出张量一定位于 CPU 后端。
        // 兼容兜底：若存在 backend（加速后端）但没有 sched，则优先使用 backend。
        backend_res = backend ? backend : backend_cpu;
    }
    if (!backend_res) {
        TTS_ABORT("get_ggml_node_data: backend_res is null (sched=%p backend=%p backend_cpu=%p)", (void *) sched, (void *) backend, (void *) backend_cpu);
    }
    ggml_backend_tensor_get_async(backend_res, output_node, output, 0, output_size);
}

void runner_context::set_threads() {
    if (backend != nullptr) {
#ifdef GGML_USE_METAL
        // 说明：
        // 旧版 ggml 曾暴露过 ggml_backend_metal_set_n_cb() 用于设置 Metal 命令缓冲数量，
        // 但在新版 ggml 中该接口已不再对外提供（仅在 ggml 内部使用）。
        // 这里不再做该项设置，保持使用 ggml 默认策略。
#endif
    }
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(backend_cpu, n_threads);
        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new(&ttp);
        ggml_backend_cpu_set_threadpool(backend_cpu, threadpool);
    }
}

void runner_context::build_schedule(size_t max_nodes) {
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();

    // ------------------------- CPU-only：绕开 scheduler -------------------------
    // 说明：
    // - ggml_backend_sched_new(graph_size) 会按 graph_size 预分配/管理一批内部数组与 context buffer；
    // - 对于 Kokoro 这种图规模较大的模型，graph_size 往往会被设置得很保守，导致 scheduler 试图申请“十几 GB”的连续内存；
    // - 在 ARM Linux 等无 swap 或严格 overcommit 的机器上，这会在“模型加载阶段”直接触发 ENOMEM，进而 ggml_init 断言崩溃。
    // - CPU-only 场景其实不需要 scheduler（没有多后端拆图/拷贝需求），用 ggml_gallocr 按实际图分配即可。
    if (backend == nullptr) {
        if (sched) {
            ggml_backend_sched_free(sched);
            sched = nullptr;
        }
        if (!galloc) {
            galloc = ggml_gallocr_new(backend_cpu_buffer);
            if (!galloc) {
                TTS_ABORT("CPU-only 创建 ggml_gallocr 失败。");
            }
        }
        return;
    }

    // 加速后端（Metal/Vulkan）：仍使用 scheduler 负责拆图/多 buffer 分配与同步
    if (galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }
    if (backend != nullptr) {
        // 说明：无论是 Metal 还是 Vulkan（以及未来更多后端），都通过通用接口获取默认 buffer type。
        backend_buffer = ggml_backend_get_default_buffer_type(backend);
        if (!backend_buffer) {
            TTS_ABORT("无法从 ggml backend 获取默认 buffer type。");
        }
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_buffer, backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend, backend_cpu};
        // 说明：ggml 0.9.4 起 ggml_backend_sched_new 增加了 op_offload 参数；此处保持与旧行为一致（关闭）。
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 2, max_nodes, false, false);
    } else {
        std::vector<ggml_backend_buffer_type_t> bufs = {backend_cpu_buffer};
        std::vector<ggml_backend_t> backs = {backend_cpu};
        sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_nodes, false, false);
    }
}

bool runner_context::prep_schedule(struct ggml_cgraph * gf) {
    // ------------------------- CPU-only：galloc 预分配（可选） -------------------------
    // 说明：该函数主要用于 “prepare_post_load” 阶段对 worst-case 图做一次可选预分配，减少首次推理时的 realloc。
    // CPU-only 下我们不使用 scheduler，因此在这里改为使用 ggml_gallocr_reserve(_n_size)。
    if (!sched) {
        if (backend == nullptr && backend_cpu != nullptr && galloc != nullptr) {
            const bool cpu_prealloc = tts_env_truthy_default_local("TTS_CPU_PREALLOC", /*default_value=*/true);
            if (!cpu_prealloc) {
                return true;
            }

            size_t sizes[1] = {0};
            ggml_gallocr_reserve_n_size(galloc, gf, /*node_buffer_ids=*/nullptr, /*leaf_buffer_ids=*/nullptr, sizes);
            const size_t cpu_need = sizes[0];

            const double max_gib = tts_env_double_default_local("TTS_CPU_PREALLOC_MAX_GIB", 2.0);
            if (max_gib > 0.0) {
                const double gib = 1024.0 * 1024.0 * 1024.0;
                const size_t limit = (size_t) (max_gib * gib);
                if (cpu_need > limit) {
                    fprintf(stderr,
                            "[tts] CPU 预分配需求 %.2f GiB 超过阈值 %.2f GiB，跳过预分配并在运行时按实际图分配。\n",
                            cpu_need / gib,
                            max_gib);
                    return true;
                }
            }

            const bool ok = ggml_gallocr_reserve(galloc, gf);
            if (!ok) {
                // 说明：worst-case 预分配失败并不致命，后续运行时仍可按实际输入图重新分配。
                fprintf(stderr,
                        "[tts] CPU 预分配失败，跳过并在运行时按实际图分配。\n");
                return true;
            }
            return true;
        }
        return false;
    }

    // ------------------------- CPU-only 预分配保护 -------------------------
    // 说明：
    // - ggml_backend_sched_reserve() 会尝试为“当前图（可能是 worst-case 图）”一次性预分配一大块连续内存；
    // - 在 CPU-only 模式下，这可能触发数 GiB 级别的申请（甚至失败），导致启动变慢、系统内存压力增大，
    //   极端情况下还可能引发后续 ggml_init 内部分配失败（ctx->mem_buffer == NULL）。
    // - 因此这里给 CPU-only 的预分配增加一个“软上限”：超过阈值则跳过预分配，让运行时按实际输入图分配。
    // - 兼容开关：
    //   - TTS_CPU_PREALLOC=0 彻底关闭 CPU 预分配；
    //   - TTS_CPU_PREALLOC_MAX_GIB=xxx 设置阈值（默认 2 GiB；设为 0 表示不设上限）。
    if (backend == nullptr && backend_cpu != nullptr) {
        const bool cpu_prealloc = tts_env_truthy_default_local("TTS_CPU_PREALLOC", /*default_value=*/true);
        if (!cpu_prealloc) {
            return true;
        }

        size_t sizes[2] = {0, 0};
        ggml_backend_sched_reserve_size(sched, gf, sizes);
        const size_t cpu_need = sizes[0];

        const double max_gib = tts_env_double_default_local("TTS_CPU_PREALLOC_MAX_GIB", 2.0);
        if (max_gib > 0.0) {
            const double gib = 1024.0 * 1024.0 * 1024.0;
            const size_t limit = (size_t) (max_gib * gib);
            if (cpu_need > limit) {
                fprintf(stderr,
                        "[tts] CPU 预分配需求 %.2f GiB 超过阈值 %.2f GiB，跳过预分配并在运行时按实际图分配。\n",
                        cpu_need / gib,
                        max_gib);
                return true;
            }
        }
    }

    // 说明：Vulkan/Metal 的“最坏图”预分配可能远超显存，启动阶段直接触发 OOM。
    // 先做不分配的尺寸测量，若需求明显超出显存则跳过预分配，
    // 让运行时按实际输入图再分配，避免无意义的大额申请。
    if (backend != nullptr && backend_buffer != nullptr) {
        // 说明：Kokoro 等模型的“最坏图”预分配在 Vulkan 上经常得不偿失：
        // - 可能触发巨额 buffer 申请（甚至超过 VkPhysicalDeviceLimits::maxStorageBufferRange / maxBufferSize）；
        // - 即便最终失败也会显著拉长 CLI 启动时间；
        // 因此默认跳过 Vulkan 的预分配；如需开启可设置：TTS_VK_PREALLOC=1
        if (tts_backend_is_vulkan_local(backend) && !tts_env_truthy_local("TTS_VK_PREALLOC")) {
            return true;
        }

        size_t sizes[2] = {0, 0};
        ggml_backend_sched_reserve_size(sched, gf, sizes);

        const size_t backend_need = sizes[0];

        // 说明：部分 Vulkan 设备虽然“显存/共享内存总量”很大，但单个 buffer 的最大尺寸受驱动/硬件限制。
        // 预分配时 ggml 往往会尝试一次性申请一个大 buffer；若超过该上限会直接失败。
        // 这里提前用 ggml 提供的 max_size 做剪枝，避免发生“先失败再回退”的高额启动开销。
        const size_t backend_max_size = ggml_backend_get_max_size(backend);
        if (backend_max_size > 0 && backend_need > backend_max_size) {
            const double gib = 1024.0 * 1024.0 * 1024.0;
            fprintf(stderr,
                    "[tts] 加速后端预分配需求 %.2f GiB 超过后端单 buffer 上限 %.2f GiB，跳过预分配并在运行时按实际图分配。\n",
                    backend_need / gib,
                    backend_max_size / gib);
            return true;
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props{};
        if (dev != nullptr) {
            ggml_backend_dev_get_props(dev, &props);
        }

        const size_t mem_total = props.memory_total;
        const size_t mem_free = props.memory_free;
        const bool exceed_total = mem_total > 0 && backend_need > mem_total;
        const bool exceed_free = !exceed_total && mem_free > 0 && backend_need > mem_free;
        const bool mem_unknown = backend_need > 0 && mem_total == 0 && mem_free == 0;

        if (mem_unknown) {
            // 说明：部分 Vulkan/Metal 驱动无法返回显存信息；此时不做预分配，避免未知预算下直接 OOM。
            fprintf(stderr,
                    "[tts] 加速后端显存信息不可用，跳过预分配并在运行时按实际图分配。\n");
            return true;
        }

        if (exceed_total || exceed_free) {
            const double gib = 1024.0 * 1024.0 * 1024.0;
            fprintf(stderr,
                    "[tts] 加速后端预分配需求 %.2f GiB 超过设备显存(可用 %.2f / 总量 %.2f GiB)，跳过预分配并在运行时按实际图分配。\n",
                    backend_need / gib,
                    mem_free / gib,
                    mem_total / gib);
            return true;
        }
    }

    const bool ok = ggml_backend_sched_reserve(sched, gf);
    if (!ok) {
        // 说明：预分配失败时重置调度器，避免后续分配处于不一致状态。
        fprintf(stderr,
                "[tts] 加速后端预分配失败，已重置调度器并改为运行时分配。\n");
        ggml_backend_sched_reset(sched);
    }
    return ok;
}

void runner_context::reset_graph() {
    // 说明：scheduler 在每次推理前/后通常会 reset，避免下次推理时状态不一致。
    // CPU-only + galloc 不需要显式 reset（alloc_graph 内部会 reset vbuffer），这里做成统一封装便于调用方不分支。
    if (sched) {
        ggml_backend_sched_reset(sched);
    }
}

bool runner_context::alloc_graph(struct ggml_cgraph * gf) {
    if (sched) {
        return ggml_backend_sched_alloc_graph(sched, gf);
    }
    if (galloc) {
        return ggml_gallocr_alloc_graph(galloc, gf);
    }
    return false;
}

enum ggml_status runner_context::compute_graph_async(struct ggml_cgraph * gf) {
    if (sched) {
        return ggml_backend_sched_graph_compute_async(sched, gf);
    }

    // 说明：CPU-only 使用 galloc 分配图张量后，直接用 CPU backend 执行即可。
    ggml_backend_t exec_backend = backend ? backend : backend_cpu;
    if (!exec_backend) {
        return GGML_STATUS_ALLOC_FAILED;
    }
    return ggml_backend_graph_compute_async(exec_backend, gf);
}

void runner_context::prep_output_buffer(size_t new_size) {
    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output) : 0;
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
            buf_output = nullptr;
            logits = nullptr;
        }
        buf_output = ggml_backend_buft_alloc_buffer(backend_cpu_buffer, new_size);
    }
    logits = (float *) ggml_backend_buffer_get_base(buf_output);
}

void runner_context::sync() {
    if (sched) {
        // 说明：异步后端（如 Vulkan/Metal）在 compute_async / tensor_get_async 后可能仍在执行，
        // 必须先同步，避免 reset 释放仍在使用的 buffer 引发崩溃或数据未就绪。
        ggml_backend_sched_synchronize(sched);
        return;
    }

    // 说明：CPU-only（galloc）同样走统一的同步接口；CPU backend 通常是同步执行，但这里保持一致性与安全性。
    if (backend != nullptr) {
        ggml_backend_synchronize(backend);
    }
    if (backend_cpu != nullptr) {
        ggml_backend_synchronize(backend_cpu);
    }
}

void tts_runner::init_build(std::vector<uint8_t>* buf_compute_meta) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta->size(),
        /*.mem_buffer =*/ buf_compute_meta->data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);
}

void tts_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

tts_generation_runner::tts_generation_runner(const tts_model_loader & loader) : loader{ ref(loader) } {}

tts_generation_runner::~tts_generation_runner() {}

std::vector<std::string_view> tts_generation_runner::list_voices() {
    GGML_ABORT("The architecture '%s' does not support #list_voices.", loader.get().arch);
}

void tts_generation_runner::update_conditional_prompt(const char * file_path, const char * prompt) {
    GGML_ABORT("The architecture '%s' does not support update_conditional_prompt.", loader.get().arch);
}

test_tts_generation_runner::test_tts_generation_runner(const tts_model_loader & loader) :
    tts_generation_runner{ loader } {
    GGML_ASSERT(loader.is_test);
}

void test_tts_generation_runner::assign_weight(const char *, ggml_tensor &) {
    GGML_ABORT("Assumed loader.is_test");
}

void test_tts_generation_runner::prepare_post_load() {
    GGML_ABORT("Assumed loader.is_test");
}

void tts_model::prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size) {
    // currently DAC is only supported on cpu because the ops are not implemented on other devices;
    if (cpu_only) {
        backend = ggml_backend_cpu_init();
        buffer = ggml_backend_cpu_buffer_type();
    } else {
        // 说明：根据当前线程的后端配置，初始化加速后端（Metal/Vulkan）。
        backend = tts_backend_init_accel();
        buffer = backend ? ggml_backend_get_default_buffer_type(backend) : nullptr;

        // 说明：部分模型在 Vulkan 下会回退 CPU 计算，为避免 CPU 直接读取设备内存导致崩溃，
        // 可选择“主机可见”的权重缓冲（若后端支持）。
        const tts_backend_config cfg = tts_get_backend_config();
        if (backend != nullptr && cfg.backend == tts_compute_backend::VULKAN && cfg.prefer_host_buffer) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (host_buft != nullptr) {
                buffer = host_buft;
                fprintf(stderr, "[tts] Vulkan 权重使用主机可见缓冲，避免 CPU 回退时访问设备内存。\n");
            }
        }

        if (!backend || !buffer) {
            const tts_backend_config cfg = tts_get_backend_config();
            const char * backend_name = "unknown";
            switch (cfg.backend) {
                case tts_compute_backend::CPU:    backend_name = "cpu";    break;
                case tts_compute_backend::METAL:  backend_name = "metal";  break;
                case tts_compute_backend::VULKAN: backend_name = "vulkan"; break;
                case tts_compute_backend::AUTO:   backend_name = "auto";   break;
            }

            TTS_ABORT("初始化 GPU 后端失败（backend=%s device=%d）。请确认编译时启用了对应后端（GGML_METAL/GGML_VULKAN），或使用 CPU 推理。",
                      backend_name,
                      cfg.device);
        }
    }
    size_t ctx_size = ggml_tensor_overhead() * (tensor_meta.n_tensors * size_offset);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);

    // 说明：不同后端对 buffer 内部的 tensor 起始地址有不同的对齐要求：
    // - CPU：通常是 32 字节对齐；
    // - Vulkan：要求满足 VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment（常见为 256）。
    //
    // 这里的权重加载逻辑属于“手动打包到单一大 buffer”，因此必须预留对齐 padding 的额外空间，
    // 否则 Vulkan 后端在绑定 storage buffer 时会因为 offset 未对齐而触发断言/崩溃。
    const size_t alignment = ggml_backend_buft_get_alignment(buffer);
    const size_t extra_pad = alignment > 1 ? alignment * (size_t) tensor_meta.n_tensors : 0;
    const size_t buf_bytes = tensor_meta.n_bytes + (size_t) dedicated_add_on_size + extra_pad;

    buf = ggml_backend_buft_alloc_buffer(buffer, buf_bytes);
    if (!buf) {
        TTS_ABORT("分配模型权重 buffer 失败：requested=%zu bytes (weights=%zu + dedicated=%u + pad=%zu)",
                  buf_bytes,
                  (size_t) tensor_meta.n_bytes,
                  dedicated_add_on_size,
                  extra_pad);
    }
}

void tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
	TTS_ABORT("%s received name, %s, tensor without being defined. %s must be defined for all implementations of tts_model. \n", __func__, name.c_str(), __func__);
}

static inline size_t tts_align_up(size_t v, size_t alignment) {
    if (alignment <= 1) {
        return v;
    }
    const size_t mask = alignment - 1;
    return (v + mask) & ~mask;
}

void tts_model::alloc_tensor(struct ggml_tensor * tensor, const char * debug_name) {
    if (!tensor) {
        TTS_ABORT("%s: tensor 为空", __func__);
    }
    if (!buf) {
        TTS_ABORT("%s: buf 未初始化（请先调用 prep_buffers_and_context）", __func__);
    }

    // 统一使用当前 buffer type 的对齐要求，避免 Vulkan/Metal 的 offset 对齐问题。
    const size_t alignment = ggml_backend_buffer_get_alignment(buf);
    offset = tts_align_up(offset, alignment);

    const size_t size = ggml_nbytes(tensor);
    const size_t cap  = ggml_backend_buffer_get_size(buf);
    if (offset + size > cap) {
        TTS_ABORT("%s: buffer 溢出：need=%zu cap=%zu (tensor=%s)",
                  __func__,
                  offset + size,
                  cap,
                  debug_name ? debug_name : "(unnamed)");
    }

    tensor->buffer = buf;
    tensor->data   = (void *) ((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    if (debug_name && *debug_name) {
        ggml_set_name(tensor, debug_name);
    }
    offset += size;
}

void tts_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    // 说明：这是项目侧“手动将权重打包进单一大 buffer”的关键路径。
    // 必须保证每个 tensor 的起始地址满足后端对齐要求，否则 Vulkan 会在绑定 storage buffer 时崩溃。
    alloc_tensor(tensor, target ? target->name : nullptr);

    const size_t size = target ? ggml_nbytes(target) : ggml_nbytes(tensor);
    if (target) {
        ggml_backend_tensor_set(tensor, target->data, 0, size);
    }
}

void tts_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset, uint32_t dedicated_add_on_size) {
    tensor_meta = compute_tensor_meta(model_prefix, load_context, compute_tensor_meta_cb);
    prep_buffers_and_context(cpu_only, size_offset, dedicated_add_on_size);
}

size_t tts_model::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void tts_model::free() {
    if (ctx) {
        ggml_free(ctx);
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}
