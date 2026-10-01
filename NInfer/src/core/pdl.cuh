#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

namespace detail {

template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t launch_classic(const LaunchConfig& launch, bool cooperative,
                                               void (*kernel)(KernelArgs...), CallArgs&&... args) {
    // Materialize the kernel's parameter types before taking addresses. Call arguments can have
    // different types (e.g. nullptr or a double literal) from the actual CUDA parameter storage.
    auto invoke = [&](KernelArgs... parameters) {
        std::array<void*, sizeof...(KernelArgs)> pointers{{static_cast<void*>(&parameters)...}};
        if (cooperative) {
            return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), launch.grid,
                                               launch.block, pointers.data(),
                                               launch.dynamic_smem_bytes, launch.stream);
        }
        return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), launch.grid, launch.block,
                                pointers.data(), launch.dynamic_smem_bytes, launch.stream);
    };
    return invoke(std::forward<CallArgs>(args)...);
}

} // namespace detail

template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_cooperative(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    return detail::launch_classic(launch, true, kernel, std::forward<CallArgs>(args)...);
}

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies(). On older devices/toolkits, ordinary stream ordering provides the fence.
#if defined(NINFER_SM75) || defined(NINFER_SM86) || CUDART_VERSION < 12000
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    return detail::launch_classic(launch, false, kernel, std::forward<CallArgs>(args)...);
}

__device__ __forceinline__ void trigger_dependents() {}
__device__ __forceinline__ void wait_for_dependencies() {}
#else
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

__device__ __forceinline__ void trigger_dependents() { cudaTriggerProgrammaticLaunchCompletion(); }
__device__ __forceinline__ void wait_for_dependencies() { cudaGridDependencySynchronize(); }
#endif

} // namespace ninfer::pdl
