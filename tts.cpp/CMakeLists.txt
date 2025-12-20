
cmake_minimum_required(VERSION 3.14)
project("tts.cpp" C CXX)
include(CheckIncludeFileCXX)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if (MSVC)
    unset(GGML_WIN_VER CACHE)
else()
    set(GGML_WIN_VER "0x601" CACHE STRING "ggml: Windows version")
endif()

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include(git-vars)

option(BUILD_SHARED_LIBS "build shared libraries" OFF)

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# 兼容较长的 GGUF tensor 名称：
# - ggml 默认 GGML_MAX_NAME=64，最多只能存 63 个字符（还要预留 '\0'）。
# - 部分 Kokoro GGUF 的 tensor 名称长度会达到 64，导致 gguf_init_from_file_impl 报错并无法加载模型。
add_compile_definitions(GGML_MAX_NAME=128)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()

# debug
option(TTS_ALL_WARNINGS           "TTS.cpp: enable all compiler warnings"                   ON)
option(TTS_ALL_WARNINGS_3RD_PARTY "TTS.cpp: enable all compiler warnings in 3rd party libs" OFF)

# build
option(TTS_FATAL_WARNINGS "TTS.cpp: enable -Werror flag" OFF)

# 运行时静态链接（默认开启：MinGW / Linux）
# 目标：生成的可执行文件默认不再依赖 libstdc++ / libgomp 等动态库，便于分发。
set(TTS_STATIC_RUNTIME_DEFAULT OFF)
if (MINGW OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TTS_STATIC_RUNTIME_DEFAULT ON)
endif()
option(TTS_STATIC_RUNTIME "TTS.cpp: link libstdc++/libgcc/libgomp statically (GNU only)" ${TTS_STATIC_RUNTIME_DEFAULT})

if (TTS_STATIC_RUNTIME AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # 让 FindOpenMP 优先选择静态 OpenMP 运行库（例如 libgomp.a），避免生成物依赖 libgomp-1.dll / libgomp.so。
    # 必须在任何 find_package(OpenMP) 之前设置（ggml-cpu 会调用 find_package(OpenMP)）。
    set(OpenMP_USE_STATIC_LIBS TRUE)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.a
        OUTPUT_VARIABLE _TTS_LIBGOMP_STATIC
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT _TTS_LIBGOMP_STATIC OR _TTS_LIBGOMP_STATIC STREQUAL "libgomp.a")
        message(FATAL_ERROR "TTS_STATIC_RUNTIME=ON 但未能定位静态 OpenMP 运行库 libgomp.a（请安装/补全 GCC 的静态运行库）")
    endif()
    set(OpenMP_gomp_LIBRARY "${_TTS_LIBGOMP_STATIC}" CACHE FILEPATH "OpenMP gomp (static)" FORCE)
    unset(_TTS_LIBGOMP_STATIC)

    # 注意：这里使用 check_cxx_compiler_flag 做“编译+链接”探测；如果缺少静态库（.a），探测也会失败。
    check_cxx_compiler_flag("-static-libstdc++" TTS_HAS_STATIC_LIBSTDCXX)
    check_cxx_compiler_flag("-static-libgcc"    TTS_HAS_STATIC_LIBGCC)

    if (NOT TTS_HAS_STATIC_LIBSTDCXX)
        message(FATAL_ERROR "TTS_STATIC_RUNTIME=ON 但当前工具链不支持/缺少静态 libstdc++（尝试安装对应的静态库或关闭 TTS_STATIC_RUNTIME）")
    endif()
    if (NOT TTS_HAS_STATIC_LIBGCC)
        message(FATAL_ERROR "TTS_STATIC_RUNTIME=ON 但当前工具链不支持/缺少静态 libgcc（尝试安装对应的静态库或关闭 TTS_STATIC_RUNTIME）")
    endif()
    add_link_options(
        $<$<LINK_LANGUAGE:CXX>:-static-libstdc++>
        $<$<LINK_LANGUAGE:CXX>:-static-libgcc>
    )
endif()

# extra artifacts
option(TTS_BUILD_EXAMPLES "TTS.cpp: build examples" ON)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# override ggml options
set(GGML_ALL_WARNINGS       ${TTS_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS     ${TTS_FATAL_WARNINGS})


# build lib
add_subdirectory(ggml)

# Windows (MinGW) 兼容性说明：
# ggml 0.9.4 在 Windows 分支中会尝试使用 ThreadPowerThrottling 相关 API/结构体。
# 某些 MinGW-w64 发行版的头文件并不包含这些较新的定义，导致编译报错。
# 这里将 ggml 的目标 Windows 版本下限固定到 0x0601（Windows 7），从而跳过该段代码路径。
if (WIN32 AND NOT MSVC)
    foreach(_tts_ggml_tgt IN ITEMS ggml-base ggml-cpu ggml)
        if (TARGET ${_tts_ggml_tgt})
            target_compile_definitions(${_tts_ggml_tgt} PRIVATE _WIN32_WINNT=0x0601 WINVER=0x0601)
        endif()
    endforeach()
    unset(_tts_ggml_tgt)
endif()

add_subdirectory(ggml-patches)
add_subdirectory(src)

# install tts

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TTS_BUILD_NUMBER        ${BUILD_NUMBER})
set(TTS_BUILD_COMMIT        ${BUILD_COMMIT})
set(TTS_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(TTS_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(TTS_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(TTS_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

# form copied from llama.cpp see explanation there.
set(GGML_TRANSIENT_DEFINES)
get_target_property(GGML_DIRECTORY ggml SOURCE_DIR)
get_directory_property(GGML_DIR_DEFINES DIRECTORY ${GGML_DIRECTORY} COMPILE_DEFINITIONS)
if (GGML_DIR_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_DIR_DEFINES})
endif()
get_target_property(GGML_TARGET_DEFINES ggml COMPILE_DEFINITIONS)
if (GGML_TARGET_DEFINES)
    list(APPEND GGML_TRANSIENT_DEFINES ${GGML_TARGET_DEFINES})
endif()
get_target_property(GGML_LINK_LIBRARIES ggml LINK_LIBRARIES)

# all public headers
set(TTS_PUBLIC_HEADERS
    include/common.h
    include/tts.h
    include/audio_file.h
    include/args.h
    include/phonemizer.h)

set_target_properties(tts PROPERTIES PUBLIC_HEADER "${TTS_PUBLIC_HEADERS}")
install(TARGETS tts LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tts-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tts-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tts
    PATH_VARS TTS_INCLUDE_INSTALL_DIR
              TTS_LIB_INSTALL_DIR
              TTS_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tts-version.cmake
    VERSION ${TTS_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tts-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/parer-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tts)

configure_file(cmake/tts.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/tts.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tts.pc"
        DESTINATION lib/pkgconfig)

# examples

if (TTS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(tests)
