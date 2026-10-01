#pragma once

#if !defined(NINFER_DISABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

#include <string>
#include <utility>

namespace ninfer {

class NvtxRange {
public:
#if defined(NINFER_DISABLE_NVTX)
    explicit NvtxRange(const char*) noexcept {}
    explicit NvtxRange(const std::string&) noexcept {}
    ~NvtxRange() = default;
#else
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }

    explicit NvtxRange(std::string name) : name_(std::move(name)) { nvtxRangePushA(name_.c_str()); }

    ~NvtxRange() { nvtxRangePop(); }
#endif

    NvtxRange(const NvtxRange&)            = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&&)                 = delete;
    NvtxRange& operator=(NvtxRange&&)      = delete;

#if !defined(NINFER_DISABLE_NVTX)
private:
    std::string name_;
#endif
};

} // namespace ninfer
