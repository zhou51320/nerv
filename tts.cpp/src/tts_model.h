#ifndef tts_model_h
#define tts_model_h

#include <cstring>
#include <functional>
#include "util.h"
#include "common.h"

using namespace std;

void append_to_response(tts_response & response, tts_response & to_append);

using tensor_meta_callback = std::function<void(ggml_tensor*)>*;

struct runner_context {
    runner_context(int n_threads): n_threads(n_threads) {};
    virtual ~runner_context() {
        // 注意：部分后端/对象在某些配置下可能为 nullptr（例如纯 CPU、或提前退出未完成初始化）。
        // 为避免析构阶段因为对 nullptr 调用 free 接口而崩溃，这里做防御性判空。
        if (sched) {
            ggml_backend_sched_free(sched);
            sched = nullptr;
        }
        if (threadpool) {
            ggml_threadpool_free(threadpool);
            threadpool = nullptr;
        }
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
            buf_output = nullptr;
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
            backend_cpu = nullptr;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
    // TODO: extend the backend and buffer support out to all devices
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t backend_buffer = nullptr;

    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buffer = nullptr;
    
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_buffer_t buf_output = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    float * logits = nullptr;
    int n_threads;

    void get_ggml_node_data(struct ggml_tensor * output_tensor, float * output, size_t output_size, ggml_backend_buffer_t buffer = nullptr);
    void set_threads();
    void build_schedule(size_t max_nodes);
    bool prep_schedule(ggml_cgraph * gf);
    void prep_output_buffer(size_t new_size);
};

struct tts_model {
    struct model_tensor_meta tensor_meta;

    // this is the current byte offset into the model's buffer.
    size_t offset = 0;

    bool use_cross_attn = true;
    
    ggml_backend_buffer_type_t buffer = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // it is quite common for implementations of tts_model to need to update attributes or perform distinct operations
    // when computing the tensor meta of the loaded model. This callback allows this as it will receive each processed tensor.
    tensor_meta_callback compute_tensor_meta_cb = nullptr;

    struct ggml_context * ctx;
    
    void prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset = 1.4, uint32_t dedicated_add_on_size = 0);

    // 将 tensor 放入当前模型的 backend buffer 中，并按后端要求做对齐。
    // 说明：
    // - Vulkan 的 StorageBuffer offset 需要满足 minStorageBufferOffsetAlignment（通常为 256 字节）；
    // - Metal/CPU 也有各自的对齐要求；这里统一使用 ggml_backend_buffer_get_alignment() 获取。
    // - 该函数只负责“分配/定位”tensor 的 data 指针，不做数据拷贝。
    void alloc_tensor(struct ggml_tensor * tensor, const char * debug_name = nullptr);

    void set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target);
    size_t max_nodes();
    void assign_weight(std::string name, ggml_tensor * tensor);
    void free();
};

#endif
