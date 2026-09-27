// nerv: inert DiskDevice for Windows builds of fastllm.
// The upstream disk offload backend depends on mmap/pread/O_DIRECT. This stub
// keeps the device linkable but registers no disk operators, so disk offload
// is unavailable while GPU/CPU inference is unaffected.
#include "devices/disk/diskdevice.h"
#include "utils.h"

#include <cstring>

namespace fastllm {
    DiskMoeCacheStats GetDiskMoeCacheStats() { return DiskMoeCacheStats(); }
    void TrimDiskMoeCache() {}
    void ReleaseDiskMoeCache(const Data *) {}

    DiskDevice::DiskDevice() {
        this->deviceType = "disk";
    }

    bool DiskDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t[size];
        return true;
    }

    bool DiskDevice::Free(void *ret) {
        delete[] (uint8_t*)ret;
        return true;
    }

    bool DiskDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        if (dst != src && dst != nullptr && src != nullptr) {
            memcpy(dst, src, size);
        }
        return true;
    }

    bool DiskDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        if (dst != src && dst != nullptr && src != nullptr) {
            memcpy(dst, src, size);
        }
        return true;
    }
}
