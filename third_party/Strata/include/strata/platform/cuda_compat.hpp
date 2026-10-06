#pragma once

#include <cuda_runtime.h>

// CUDA 12 exposes the three-argument cudaGraphInstantiate(exec, graph, flags)
// form. CUDA 11.7/11.8 retain the older five-argument form, so keep one call
// site that builds against both runtime headers.
inline cudaError_t strata_cuda_graph_instantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                                  unsigned long long flags = 0) {
#if defined(CUDART_VERSION) && CUDART_VERSION < 12000
    cudaGraphNode_t error_node = nullptr;
    char log_buffer[1024] = {};
    return ::cudaGraphInstantiate(exec, graph, &error_node, log_buffer, flags);
#else
    return ::cudaGraphInstantiate(exec, graph, flags);
#endif
}
