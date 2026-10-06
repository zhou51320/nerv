#pragma once

// Small declarations for optional Windows APIs whose SDK declarations are
// hidden when targeting Win7.  The layout matches the documented
// WIN32_MEMORY_RANGE_ENTRY structure and is only used with GetProcAddress.
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

struct StrataWin32MemoryRangeEntry {
    PVOID VirtualAddress;
    SIZE_T NumberOfBytes;
};
#endif
