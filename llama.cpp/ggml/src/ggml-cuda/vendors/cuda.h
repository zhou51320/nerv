#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#else
// CUDA 10.2 does not ship cuda_bf16.h. Provide a minimal compatibility shim
// so legacy toolchains can compile code paths that mention nv_bfloat* types.
typedef half  nv_bfloat16;
typedef half2 nv_bfloat162;

static inline __host__ __device__ nv_bfloat16 __float2bfloat16(float x) {
    return __float2half(x);
}

static inline __host__ __device__ float __bfloat162float(nv_bfloat16 x) {
    return __half2float(x);
}

static inline __host__ __device__ nv_bfloat162 __float22bfloat162_rn(float2 x) {
    return __float22half2_rn(x);
}
#endif // CUDART_VERSION >= 11000

#if CUDART_VERSION >= 12050
#include <cuda_fp8.h>
#endif // CUDART_VERSION >= 12050

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#ifndef CUDA_R_16BF
#define CUDA_R_16BF CUDA_R_16F
#endif // CUDA_R_16BF
#endif // CUDART_VERSION < 11020
