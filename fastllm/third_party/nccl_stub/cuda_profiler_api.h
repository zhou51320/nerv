#ifndef FASTLLM_NCCL_STUB_CUDA_PROFILER_API_H
#define FASTLLM_NCCL_STUB_CUDA_PROFILER_API_H

// nerv: fallback header for build environments (e.g. CUDA redist packages)
// that do not ship cuda_profiler_api.h. The multicuda code includes this
// header but makes no profiler calls, so a minimal decl is enough.

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaProfilerInitialize(const char *configFile,
                                   const char *outputFile,
                                   unsigned int outputMode) { return 0; }
cudaError_t cudaProfilerStart(void) { return 0; }
cudaError_t cudaProfilerStop(void) { return 0; }

#ifdef __cplusplus
}
#endif

#endif