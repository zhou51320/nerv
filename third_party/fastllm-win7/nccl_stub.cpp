// nerv: NCCL stub for Windows builds of fastllm (see include/nccl.h).
#define FASTLLM_WIN7_NCCL_STUB_IMPL
#include "nccl.h"

extern "C" {

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *) {
    if (comms != nullptr) {
        for (int i = 0; i < ndev; ++i) {
            comms[i] = nullptr;
        }
    }
    return ncclSystemError;
}

ncclResult_t ncclCommDestroy(ncclComm_t) { return ncclSuccess; }
ncclResult_t ncclCommAbort(ncclComm_t) { return ncclSuccess; }

ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t *asyncError) {
    if (asyncError != nullptr) {
        *asyncError = ncclSystemError;
    }
    return ncclSystemError;
}

ncclResult_t ncclGroupStart(void) { return ncclSystemError; }
ncclResult_t ncclGroupEnd(void) { return ncclSystemError; }

ncclResult_t ncclSend(const void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
    return ncclSystemError;
}

ncclResult_t ncclRecv(void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
    return ncclSystemError;
}

ncclResult_t ncclBroadcast(const void *, void *, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
    return ncclSystemError;
}

ncclResult_t ncclAllReduce(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) {
    return ncclSystemError;
}

ncclResult_t ncclAllGather(const void *, void *, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
    return ncclSystemError;
}

ncclResult_t ncclReduce(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t) {
    return ncclSystemError;
}

const char *ncclGetErrorString(ncclResult_t) {
    return "NCCL is not available on Windows builds of fastllm (multi-GPU unsupported)";
}

} // extern "C"
