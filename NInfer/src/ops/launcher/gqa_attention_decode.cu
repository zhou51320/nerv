// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/common/dispatch.h"
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t gqa_small_t_launch_capacity(GqaExecutionEnvelope envelope, std::int32_t tokens,
                                         DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            gqa_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
        static_cast<__nv_bfloat16*>(cache_v.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    auto launch = [&](auto warps_per_cta_tag, auto min_blocks_per_sm_tag, auto key_block_tag, auto dynamic_arena_tag) {
        constexpr int WarpsPerCta = decltype(warps_per_cta_tag)::value;
        constexpr int MinBlocksPerSm = decltype(min_blocks_per_sm_tag)::value;
        constexpr int KeyBlock = decltype(key_block_tag)::value;
        constexpr bool DynamicArena = decltype(dynamic_arena_tag)::value;
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kGqaHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena, MultiBatch, Masked, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch(DispatchValue<24>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
        } else if (implementation_window <= 2054) {
            launch(DispatchValue<12>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
        } else if (implementation_window <= 8198) {
            launch(DispatchValue<12>{}, DispatchValue<1>{}, DispatchValue<64>{}, DispatchValue<true>{});
        } else {
            launch(DispatchValue<6>{}, DispatchValue<2>{}, DispatchValue<32>{}, DispatchValue<false>{});
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch(DispatchValue<32>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
            } else if (implementation_window <= 1029) {
                launch(DispatchValue<16>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
            } else {
                launch(DispatchValue<8>{}, DispatchValue<2>{}, DispatchValue<32>{}, DispatchValue<false>{});
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch(DispatchValue<24>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
            } else if (implementation_window <= 1029) {
                launch(DispatchValue<24>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
            } else if (implementation_window <= 4096) {
                launch(DispatchValue<12>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
            } else {
                launch(DispatchValue<6>{}, DispatchValue<2>{}, DispatchValue<32>{}, DispatchValue<false>{});
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch(DispatchValue<16>{}, DispatchValue<1>{}, DispatchValue<32>{}, DispatchValue<false>{});
        } else {
            launch(DispatchValue<8>{}, DispatchValue<2>{}, DispatchValue<32>{}, DispatchValue<false>{});
        }
    } else {
        launch(DispatchValue<8>{}, DispatchValue<2>{}, DispatchValue<32>{}, DispatchValue<false>{});
    }
    CUDA_CHECK(cudaGetLastError());
}

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    PagedKVBatchLayerView result{};        result.k_pages = cache.k_pages;
        result.v_pages = cache.v_pages;
        result.k_scale_pages = cache.k_scale_pages;
        result.v_scale_pages = cache.v_scale_pages;
        result.block_tables = cache.block_table.view({cache.block_table.ne[0], 1});
        result.head_dim = cache.head_dim;
        result.num_kv_heads = cache.num_kv_heads;
        result.dtype = cache.dtype;
        result.quant_group = cache.quant_group;
    return result;
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 || (cache_dtype != DType::BF16 && cache_dtype != DType::I8) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    if (q_heads == Gqa27Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa27Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == Gqa35Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa35Geometry>(envelope, tokens, cache_dtype);
    }
    throw std::invalid_argument("gqa_attention split capacity: unsupported head geometry");
}

template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                      Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                      cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits =
        gqa_small_t_launch_capacity<Geometry>(envelope, invocation.width, cache.dtype);

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define NINFER_GQA_SMALL_T_DISPATCH(TOKENS, WARPS)                                                 \
    do {                                                                                           \
        const auto launch_profile = [&](auto multi_batch_tag, auto masked_tag) { \
            constexpr bool MultiBatch = decltype(multi_batch_tag)::value; \
            constexpr bool Masked = decltype(masked_tag)::value;                          \
            if (cache.dtype == DType::I8) {                                                        \
                launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked>(                      \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);     \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            }                                                                                      \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile(DispatchValue<false>{}, DispatchValue<true>{});                                 \
            } else {                                                                               \
                launch_profile(DispatchValue<false>{}, DispatchValue<false>{});                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile(DispatchValue<true>{}, DispatchValue<true>{});                                      \
        } else {                                                                                   \
            launch_profile(DispatchValue<true>{}, DispatchValue<false>{});                                     \
        }                                                                                          \
    } while (0)

    switch (invocation.width) {
    case 1:
        NINFER_GQA_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_GQA_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_GQA_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_GQA_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_GQA_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_GQA_SMALL_T_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef NINFER_GQA_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kGqaHeadDim, kDChunk),
                           invocation.width * invocation.batch_size);
    const auto launch_reduce = [&](auto int8_tag, auto multi_batch_tag, auto masked_tag, auto offset_tag) {
        constexpr bool Int8 = decltype(int8_tag)::value;
        constexpr bool MultiBatch = decltype(multi_batch_tag)::value;
        constexpr bool Masked = decltype(masked_tag)::value;
        constexpr bool Offset = decltype(offset_tag)::value;
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                   Offset>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    const bool masked         = invocation.valid_columns != nullptr;
    const auto launch_profile = [&](auto int8_tag, auto multi_batch_tag, auto masked_tag) {
        constexpr bool Int8 = decltype(int8_tag)::value;
        constexpr bool MultiBatch = decltype(multi_batch_tag)::value;
        constexpr bool Masked = decltype(masked_tag)::value;
        if (invocation.column_begin == 0) {
            launch_reduce(DispatchValue<Int8>{}, DispatchValue<MultiBatch>{}, DispatchValue<Masked>{}, DispatchValue<false>{});
        } else {
            launch_reduce(DispatchValue<Int8>{}, DispatchValue<MultiBatch>{}, DispatchValue<Masked>{}, DispatchValue<true>{});
        }
    };
    const auto launch_for_dtype = [&](auto int8_tag) {
        constexpr bool Int8 = decltype(int8_tag)::value;
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_profile(DispatchValue<Int8>{}, DispatchValue<false>{}, DispatchValue<true>{});
            } else {
                launch_profile(DispatchValue<Int8>{}, DispatchValue<false>{}, DispatchValue<false>{});
            }
        } else if (masked) {
            launch_profile(DispatchValue<Int8>{}, DispatchValue<true>{}, DispatchValue<true>{});
        } else {
            launch_profile(DispatchValue<Int8>{}, DispatchValue<true>{}, DispatchValue<false>{});
        }
    };
    if (cache.dtype == DType::I8) {
        launch_for_dtype(DispatchValue<true>{});
    } else {
        launch_for_dtype(DispatchValue<false>{});
    }
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    GqaSmallTInvocation invocation{};        invocation.valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns;
        invocation.table_rows = &table_rows;
        invocation.full_width = q.ne[2];
        invocation.column_begin = column_begin;
        invocation.width = width;
        invocation.batch_size = q.ne[3];
    
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream) {
    const GqaCachedInput input{};
    GqaSmallTInvocation invocation{};        invocation.valid_columns = nullptr;
        invocation.table_rows = nullptr;
        invocation.full_width = q.ne[2];
        invocation.column_begin = 0;
        invocation.width = q.ne[2];
        invocation.batch_size = 1;
    
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

} // namespace ninfer::ops::detail
