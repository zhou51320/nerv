#ifndef FASTLLM_NCCL_STUB_NCCL_H
#define FASTLLM_NCCL_STUB_NCCL_H

// nerv: minimal NCCL interface stub for Windows builds.
// Real NCCL does not exist on Windows; the stub lets the multicuda
// compilation units build/link, and every stub call fails at runtime so
// multi-GPU features report an error instead of crashing.

#include <cuda_runtime.h>
#include <stddef.h>

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 8
} ncclResult_t;

typedef struct ncclComm *ncclComm_t;

typedef enum {
    ncclInt8 = 0,
    ncclChar = 1,
    ncclUint8 = 2,
    ncclInt32 = 3,
    ncclInt = 3,
    ncclUint32 = 4,
    ncclUint = 4,
    ncclInt64 = 5,
    ncclUint64 = 6,
    ncclFloat64 = 7,
    ncclFloat32 = 8,
    ncclFloat = 8,
    ncclFloat16 = 9,
    ncclHalf = 9,
    ncclBfloat16 = 10,
    ncclBfloat = 10
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3
} ncclRedOp_t;

#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();
ncclResult_t ncclSend(const void *buff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void *buff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
const char *ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif

#endif