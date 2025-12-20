#pragma once

#include <cstdint>
#include <utility>

#include "ggml.h"
#include "gguf.h"

class gguf_key_iterator {
    const gguf_context * const ctx;
    const int64_t              n_kv;
    int64_t                    i{};

  public:
    explicit gguf_key_iterator(const gguf_context & ctx) : ctx{ &ctx }, n_kv{ gguf_get_n_kv(&ctx) } {}

    std::pair<int64_t, const char *> operator*() const { return { i, gguf_get_key(ctx, static_cast<int>(i)) }; }

    gguf_key_iterator & operator++() {
        ++i;
        return *this;
    }

    gguf_key_iterator begin() const {
        auto result{ *this };
        result.i = 0;
        return result;
    }

    gguf_key_iterator end() const {
        auto result{ *this };
        result.i = n_kv;
        return result;
    }

    // C++17 兼容：默认化比较运算符（= default）是 C++20 才支持的语法。
    // 这里手写 == / !=，以便 range-for 能正常使用 begin != end 迭代。
    bool operator==(const gguf_key_iterator & other) const noexcept {
        return ctx == other.ctx && i == other.i;
    }

    bool operator!=(const gguf_key_iterator & other) const noexcept {
        return !(*this == other);
    }
};

class ggml_tensor_iterator {
    const ggml_context * const ctx;
    ggml_tensor *              cur;

  public:
    explicit ggml_tensor_iterator(const ggml_context & ctx) : ctx{ &ctx }, cur{ ggml_get_first_tensor(&ctx) } {}

    ggml_tensor & operator*() const { return *cur; }

    ggml_tensor_iterator & operator++() {
        cur = ggml_get_next_tensor(ctx, cur);
        return *this;
    }

    ggml_tensor_iterator begin() const {
        auto result{ *this };
        result.cur = ggml_get_first_tensor(ctx);
        return result;
    }

    ggml_tensor_iterator end() const {
        auto result{ *this };
        result.cur = nullptr;
        return result;
    }

    // C++17 兼容：默认化比较运算符（= default）是 C++20 才支持的语法。
    // 这里手写 == / !=，以便 range-for 能正常使用 begin != end 迭代。
    bool operator==(const ggml_tensor_iterator & other) const noexcept {
        return ctx == other.ctx && cur == other.cur;
    }

    bool operator!=(const ggml_tensor_iterator & other) const noexcept {
        return !(*this == other);
    }
};
