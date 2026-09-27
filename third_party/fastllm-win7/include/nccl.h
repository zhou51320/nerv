#ifndef FASTLLM_WIN7_NCCL_H
#define FASTLLM_WIN7_NCCL_H

// nerv: minimal NCCL interface for Windows builds of fastllm.
// NCCL does not exist on Windows; every call fails at runtime, so multi-GPU
// tensor parallel reports an error while single-GPU inference is unaffected.

#include <stddef.h>

#ifdef FASTLLM_WIN7_NCCL_STUB_IMPL
typedef struct CUstream_st *cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7
} ncclResult_t;

typedef struct ncclComm *ncclComm_t;

typedef enum {
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2, ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, ncclHalf = 6,
    ncclFloat32 = 7, ncclFloat = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16 = 9
} ncclDataType_t;

typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 } ncclRedOp_t;

#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);
ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);
const char *ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif

#endif
