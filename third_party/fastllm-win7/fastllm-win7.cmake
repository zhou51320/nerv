# nerv: Win7 / MSVC compatibility layer for upstream fastllm.
#
# Injected right after `project(fastllm)` via
#   -DCMAKE_PROJECT_fastllm_INCLUDE=<this file>
# so that fastllm/ can stay a pure upstream copy and be replaced on update.
if (NOT WIN32)
    return()
endif()

set(FASTLLM_WIN7_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Upstream's Windows POST_BUILD for fastllm_tools uses the VS-only
# $(Configuration) macro, which makes Ninja reject the whole build.ninja.
# We don't ship fastllm_tools, so drop that step and forward everything else.
function(add_custom_command)
    if (ARGC GREATER 1 AND ARGV0 STREQUAL "TARGET" AND ARGV1 STREQUAL "fastllm_tools")
        return()
    endif()
    _add_custom_command(${ARGV})
endfunction()

# CUDA import libs (cublas, cublasLt, cuda) are linked by bare name upstream.
cmake_policy(SET CMP0074 NEW)
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_LIBRARY_DIR)
    link_directories("${CUDAToolkit_LIBRARY_DIR}")
endif()

# MSVC lacks the GCC extensions used by a few CPU translation units:
#  - __int128 only holds device memory byte budgets (fits in 64 bits)
#  - __attribute__((aligned/always_inline/noinline)) are hints only
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:/D__int128=__int64>"
    "$<$<COMPILE_LANGUAGE:CXX>:/D__attribute__(x)=>"
)

# gguf.cpp uses C++20 designated initializers; the project defaults to C++17.
set_source_files_properties(third_party/gguf/gguf.cpp PROPERTIES COMPILE_OPTIONS "/std:c++20")

# amx.cpp includes <sys/syscall.h> unconditionally but only calls syscall()
# under __AMX_TILE__, which MSVC never defines.
set_source_files_properties(src/devices/cpu/amx.cpp PROPERTIES
    INCLUDE_DIRECTORIES "${FASTLLM_WIN7_DIR}/shim")

# NCCL does not exist on Windows. A target named `nccl` makes the upstream
# `target_link_libraries(... nccl ...)` resolve to this single-GPU stub.
add_library(nccl STATIC "${FASTLLM_WIN7_DIR}/nccl_stub.cpp")
target_include_directories(nccl PUBLIC "${FASTLLM_WIN7_DIR}/include")

# The disk offload backend is built on mmap/pread; swap in an inert stub.
function(fastllm_win7_fixup_sources)
    foreach (tgt IN ITEMS fastllm fastllm_tools)
        if (TARGET ${tgt})
            get_target_property(srcs ${tgt} SOURCES)
            list(FILTER srcs EXCLUDE REGEX "src/devices/disk/diskdevice\\.cpp$")
            list(APPEND srcs "${FASTLLM_WIN7_DIR}/diskdevice_win.cpp")
            set_property(TARGET ${tgt} PROPERTY SOURCES ${srcs})
        endif()
    endforeach()
endfunction()
cmake_language(DEFER CALL fastllm_win7_fixup_sources)
