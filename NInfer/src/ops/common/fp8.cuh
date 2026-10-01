#pragma once

#include <cuda_runtime.h>

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {

// E4M3FN has bias 7, subnormals, signed zero, and NaN at magnitude 0x7f (no infinity).
// Every finite value is exactly representable in FP32 and BF16.
__host__ __device__ inline float decode_fp8_e4m3(std::uint8_t storage) {
    const int magnitude = storage & 0x7f;
    if (magnitude == 0x7f) { return nanf(""); }
    const int exponent = magnitude >> 3;
    const int mantissa = magnitude & 7;
    const float value = exponent == 0 ? static_cast<float>(mantissa) * (1.0F / 512.0F)
                                      : ldexpf(static_cast<float>(8 + mantissa), exponent - 10);
    return (storage & 0x80) != 0 ? -value : value;
}

__host__ __device__ inline float2 decode_fp8_e4m3x2(std::uint16_t storage) {
#if CUDART_VERSION >= 11080
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<float2>(value);
#else
    return make_float2(decode_fp8_e4m3(static_cast<std::uint8_t>(storage)),
                       decode_fp8_e4m3(static_cast<std::uint8_t>(storage >> 8)));
#endif
}

} // namespace ninfer::ops::detail
