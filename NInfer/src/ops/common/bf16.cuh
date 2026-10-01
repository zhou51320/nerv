#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace ninfer::ops {

// CUDA 11.7 hides the vector BF16 intrinsics from sm_75. Widen each stored lane exactly;
// do not convert through FP16, whose exponent range cannot represent general BF16 values.
__host__ __device__ inline float2 bf16x2_to_float2(__nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat1622float2(value);
#else
    return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
#endif
}

__host__ __device__ inline __nv_bfloat162 bf16x2_sub_rn(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hsub2_rn(a, b);
#else
    const float2 av = bf16x2_to_float2(a);
    const float2 bv = bf16x2_to_float2(b);
    return __floats2bfloat162_rn(av.x - bv.x, av.y - bv.y);
#endif
}

} // namespace ninfer::ops
