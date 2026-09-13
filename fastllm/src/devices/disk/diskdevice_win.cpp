#include "devices/disk/diskdevice.h"
#include "utils/utils.h"

#include <cstring>

// nerv: the Linux disk-weight backend depends on POSIX memory mapping
// (mmap/pread/O_DIRECT) that is not available on Windows.  Keep the class
// linkable so model code that instantiates DiskDevice still builds, but leave
// it inert: no disk operators are registered and any weight load fails fast.
namespace fastllm {
    DiskDevice::DiskDevice() {
        this->deviceType = "disk";
        WarnInFastLLM("disk device is not supported on this platform\n");
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