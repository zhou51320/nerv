#pragma once

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdio>
#include <cstddef>
#include <vector>

namespace strata::kernels::cpu::detail {

// Hard affinity is only for pool-owned threads, which terminate with the pool. A caller's implicit
// all-group affinity cannot be restored from PreviousGroupAffinity, so the host uses CPU Sets below.
inline bool set_thread_group_affinity(int core, int worker, GROUP_AFFINITY* previous = nullptr) {
    if (core < 0) return false;
    GROUP_AFFINITY target{};
    target.Group = (WORD) (core / 64);
    target.Mask = KAFFINITY(1) << (core & 63);
    if (SetThreadGroupAffinity(GetCurrentThread(), &target, previous)) return true;
    std::fprintf(stderr, "strata cpu pool: SetThreadGroupAffinity for worker %d (group %u, mask 0x%llx) failed: %lu; previous affinity kept\n",
                 worker, (unsigned) target.Group, (unsigned long long) target.Mask, (unsigned long) GetLastError());
    return false;
}

inline bool get_thread_cpu_sets(std::vector<ULONG>& ids) {
    using GetThreadSelectedCpuSetsFn = BOOL(WINAPI*)(HANDLE, PULONG, ULONG, PULONG);
    static const GetThreadSelectedCpuSetsFn get_sets = reinterpret_cast<GetThreadSelectedCpuSetsFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadSelectedCpuSets"));
    if (!get_sets) return false;
    ULONG count = 0;
    if (!get_sets(GetCurrentThread(), nullptr, 0, &count) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    ids.resize(count);
    if (count == 0) return true;
    if (!get_sets(GetCurrentThread(), ids.data(), count, &count)) return false;
    ids.resize(count);
    return true;
}

inline bool set_thread_cpu_sets(const std::vector<ULONG>& ids) {
    using SetThreadSelectedCpuSetsFn = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
    static const SetThreadSelectedCpuSetsFn set_sets = reinterpret_cast<SetThreadSelectedCpuSetsFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadSelectedCpuSets"));
    if (!set_sets) return false;
    return set_sets(GetCurrentThread(), ids.empty() ? nullptr : ids.data(), (ULONG) ids.size()) != 0;
}

inline bool cpu_sets_available() {
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    return kernel && GetProcAddress(kernel, "SetThreadSelectedCpuSets") != nullptr &&
           GetProcAddress(kernel, "GetThreadSelectedCpuSets") != nullptr &&
           GetProcAddress(kernel, "GetSystemCpuSetInformation") != nullptr;
}

// CPU Set IDs are opaque: find the entry by both group and group-relative processor number.
inline bool cpu_set_for_core(int core, ULONG& id) {
    using GetSystemCpuSetInformationFn = BOOL(WINAPI*)(void*, ULONG, PULONG, HANDLE, ULONG);
    static const GetSystemCpuSetInformationFn get_info = reinterpret_cast<GetSystemCpuSetInformationFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetSystemCpuSetInformation"));
    if (!get_info) return false;
    // The Win10 SDK hides SYSTEM_CPU_SET_INFORMATION when targeting Win7.
    // We only need the stable prefix and the group-relative processor index.
    struct CpuSetInfo {
        ULONG Size;
        ULONG Type;
        ULONG64 Id;
        USHORT Group;
        UCHAR LogicalProcessorIndex;
        UCHAR CoreIndex;
        UCHAR LastLevelCacheIndex;
        UCHAR NumaNodeIndex;
        UCHAR EfficiencyClass;
        UCHAR AllFlags;
        ULONG Reserved;
    };
    ULONG bytes = 0;
    if (!get_info(nullptr, 0, &bytes, GetCurrentProcess(), 0) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    std::vector<unsigned char> buffer(bytes);
    if (bytes && !get_info(buffer.data(), bytes, &bytes, GetCurrentProcess(), 0)) return false;
    constexpr size_t header_size = sizeof(ULONG) * 2;
    for (size_t offset = 0; offset + header_size <= bytes;) {
        const auto* info = reinterpret_cast<const CpuSetInfo*>(buffer.data() + offset);
        if (info->Size < header_size || info->Size > bytes - offset) break;
        if (info->Type == 0 && info->Size >= sizeof(CpuSetInfo) && info->Group == core / 64 &&
            info->LogicalProcessorIndex == (core & 63)) {
            id = (ULONG) info->Id;
            return true;
        }
        offset += info->Size;
    }
    SetLastError(ERROR_NOT_FOUND);
    return false;
}

}  // namespace strata::kernels::cpu::detail
#endif
