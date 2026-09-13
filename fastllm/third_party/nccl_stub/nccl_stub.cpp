// nerv: Windows-only NCCL stub implementation.
// Included in the fastllm build on WIN32 so the multicuda compilation units
// link; every call fails so multi-GPU features error out at runtime.

#include "nccl.h"

#include <stddef.h>

static const char kStubError[] =
    "NCCL stub: multi-GPU (NCCL) is not supported on Windows builds of fastllm";

extern "C" {

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    (void)devlist;
    if (comms != nullptr) {
        for (int i = 0; i < ndev; ++i) {
            comms[i] = nullptr;
        }
    }
    return ncclUnhandledError;
}

ncclResult_t ncclGroupStart() { return ncclUnhandledError; }

ncclResult_t ncclGroupEnd() { return ncclUnhandledError; }

ncclResult_t ncclSend(const void *buff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)buff; (void)count; (void)datatype; (void)peer; (void)comm; (void)stream;
    return ncclUnhandledError;
}

ncclResult_t ncclRecv(void *buff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)buff; (void)count; (void)datatype; (void)peer; (void)comm; (void)stream;
    return ncclUnhandledError;
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)count; (void)datatype; (void)root;
    (void)comm; (void)stream;
    return ncclUnhandledError;
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)count; (void)datatype; (void)op;
    (void)comm; (void)stream;
    return ncclUnhandledError;
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)sendcount; (void)datatype;
    (void)comm; (void)stream;
    return ncclUnhandledError;
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff; (void)recvbuff; (void)count; (void)datatype; (void)op;
    (void)root; (void)comm; (void)stream;
    return ncclUnhandledError;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    (void)comm;
    return ncclUnhandledError;
}

const char *ncclGetErrorString(ncclResult_t result) {
    (void)result;
    return kStubError;
}

} // extern "C"