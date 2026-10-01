#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

/**
 * Non-owning, single-sequence view consumed by growing-cache Ops.
 *
 * Physical plane order is fixed by the owning homogeneous pool and validated by the consuming
 * Op. block_table is one contiguous I32 logical-block row.
 */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
};

/**
 * Non-owning multi-sequence view consumed by batched growing-cache Ops.
 *
 * Physical planes and the complete block-table matrix are shared by every logical row in one
 * invocation. block_tables is contiguous I32 [logical_pages, table_rows]; the consuming Op
 * receives its per-row table selectors separately.
 */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
};

} // namespace ninfer
