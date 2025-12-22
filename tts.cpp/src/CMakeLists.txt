# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif ()
endif ()

# TTS

add_library(tts
        ../include/args.h
        tokenizer.cpp
        sampler.cpp
        sampler.h
        util.cpp
        util.h
        args.cpp
        tts_model.cpp
        tts_model.h
)

target_include_directories(tts PUBLIC . ../include ../ggml/src/)

target_link_libraries(tts PUBLIC ggml)

# 性能（CPU）：
# ggml-cpu 目标在 GGML_NATIVE=ON 时会启用 `-march=native`（或等价的指令集开关），
# 但本项目侧的 CPU-only 自定义算子（如 Snake1d/conv_transpose/stft 等）也在 `tts` 目标里编译，
# 如果这里不跟随开启，会导致这些自定义算子无法使用 AVX/FMA 等指令，出现“ggml 很快但自定义算子拖后腿”的情况。
# 因此当 GGML_NATIVE 开启时，让 `tts` 目标也使用同样的 native 优化（不额外改变可移植性：最终产物本就依赖 ggml-cpu 的 native 指令集）。
if (GGML_NATIVE)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(tts PRIVATE -march=native)
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(tts PRIVATE LLAMA_BUILD)
    target_compile_definitions(tts PUBLIC LLAMA_SHARED)
endif ()

add_subdirectory(decoder)
add_subdirectory(models)
