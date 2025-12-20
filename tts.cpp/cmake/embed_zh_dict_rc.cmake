# 说明：将中文拼音词典以 Windows 资源（RCDATA）形式打包进最终可执行文件。
# - 这样可规避 MSVC 对超大字符串常量的限制（C2026: 字符串太大）。
# - 资源会被链接到 exe 的 .rsrc 段，运行时通过 WinAPI 读取并以 string_view 视图返回。

if (NOT DEFINED INPUT_PHRASE OR NOT DEFINED INPUT_SINGLE OR NOT DEFINED OUTPUT_RC)
    message(FATAL_ERROR "embed_zh_dict_rc.cmake: need -DINPUT_PHRASE=... -DINPUT_SINGLE=... -DOUTPUT_RC=...")
endif()

if (NOT EXISTS "${INPUT_PHRASE}")
    message(FATAL_ERROR "embed_zh_dict_rc.cmake: phrase dict not found: ${INPUT_PHRASE}")
endif()

if (NOT EXISTS "${INPUT_SINGLE}")
    message(FATAL_ERROR "embed_zh_dict_rc.cmake: single dict not found: ${INPUT_SINGLE}")
endif()

get_filename_component(_TTS_OUT_DIR "${OUTPUT_RC}" DIRECTORY)
file(MAKE_DIRECTORY "${_TTS_OUT_DIR}")

# 说明：RC 文件中的路径字符串按 C 风格解析，反斜杠可能触发转义；这里统一转成正斜杠路径以避免转义问题。
file(TO_CMAKE_PATH "${INPUT_PHRASE}" _TTS_PHRASE_PATH)
file(TO_CMAKE_PATH "${INPUT_SINGLE}" _TTS_SINGLE_PATH)

file(WRITE "${OUTPUT_RC}" "#pragma code_page(65001)\n")
file(APPEND "${OUTPUT_RC}" "// 此文件由 CMake 脚本自动生成（cmake/embed_zh_dict_rc.cmake），请勿手改。\n\n")
file(APPEND "${OUTPUT_RC}" "\"EVA_TTS_ZH_PHRASE_DICT\" RCDATA \"${_TTS_PHRASE_PATH}\"\n")
file(APPEND "${OUTPUT_RC}" "\"EVA_TTS_ZH_SINGLE_DICT\" RCDATA \"${_TTS_SINGLE_PATH}\"\n")

unset(_TTS_OUT_DIR)
unset(_TTS_PHRASE_PATH)
unset(_TTS_SINGLE_PATH)

