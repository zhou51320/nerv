# 说明：把中文拼音词典以 raw string literal 形式生成 C++ 源文件并编译进二进制。
# 目的：运行时不再依赖外部 dict 文件，方便移植/分发；同时保留外部覆盖能力。

if (NOT DEFINED INPUT_PHRASE OR NOT DEFINED INPUT_SINGLE OR NOT DEFINED OUTPUT_CPP)
    message(FATAL_ERROR "embed_zh_dict.cmake: need -DINPUT_PHRASE=... -DINPUT_SINGLE=... -DOUTPUT_CPP=...")
endif()

if (NOT EXISTS "${INPUT_PHRASE}")
    message(FATAL_ERROR "embed_zh_dict.cmake: phrase dict not found: ${INPUT_PHRASE}")
endif()

if (NOT EXISTS "${INPUT_SINGLE}")
    message(FATAL_ERROR "embed_zh_dict.cmake: single dict not found: ${INPUT_SINGLE}")
endif()

file(READ "${INPUT_PHRASE}" _TTS_PHRASE_CONTENT)
file(READ "${INPUT_SINGLE}" _TTS_SINGLE_CONTENT)

# 说明：raw string literal 的分隔符必须“不会出现在内容里”，否则会提前闭合导致编译失败。
# 这里用内容 MD5 生成一个几乎不可能冲突的分隔符，并做一次简单冲突检查。
string(MD5 _TTS_PHRASE_MD5 "${_TTS_PHRASE_CONTENT}")
string(MD5 _TTS_SINGLE_MD5 "${_TTS_SINGLE_CONTENT}")
# 说明：C++ raw string delimiter 的长度上限为 16（标准要求），因此这里只取 MD5 前 8 位。
string(SUBSTRING "${_TTS_PHRASE_MD5}" 0 8 _TTS_PHRASE_MD5_8)
string(SUBSTRING "${_TTS_SINGLE_MD5}" 0 8 _TTS_SINGLE_MD5_8)
set(_TTS_PHRASE_DELIM "TTSZHP${_TTS_PHRASE_MD5_8}")
set(_TTS_SINGLE_DELIM "TTSZHS${_TTS_SINGLE_MD5_8}")

string(FIND "${_TTS_PHRASE_CONTENT}" ")${_TTS_PHRASE_DELIM}\"" _TTS_PHRASE_CONFLICT)
if (NOT _TTS_PHRASE_CONFLICT EQUAL -1)
    message(FATAL_ERROR "embed_zh_dict.cmake: raw string delimiter conflict (phrase); please change delimiter strategy")
endif()

string(FIND "${_TTS_SINGLE_CONTENT}" ")${_TTS_SINGLE_DELIM}\"" _TTS_SINGLE_CONFLICT)
if (NOT _TTS_SINGLE_CONFLICT EQUAL -1)
    message(FATAL_ERROR "embed_zh_dict.cmake: raw string delimiter conflict (single); please change delimiter strategy")
endif()

get_filename_component(_TTS_OUT_DIR "${OUTPUT_CPP}" DIRECTORY)
file(MAKE_DIRECTORY "${_TTS_OUT_DIR}")

file(WRITE "${OUTPUT_CPP}" "// 此文件由 CMake 脚本自动生成（cmake/embed_zh_dict.cmake），请勿手改。\n")
file(APPEND "${OUTPUT_CPP}" "#include <string_view>\n")
file(APPEND "${OUTPUT_CPP}" "#include \"models/kokoro/zh_dict_builtin.h\"\n\n")
file(APPEND "${OUTPUT_CPP}" "namespace kokoro_zh {\n\n")

file(APPEND "${OUTPUT_CPP}" "static const char kZhPhraseDictUtf8[] = R\"${_TTS_PHRASE_DELIM}(\n")
file(APPEND "${OUTPUT_CPP}" "${_TTS_PHRASE_CONTENT}")
file(APPEND "${OUTPUT_CPP}" "\n)${_TTS_PHRASE_DELIM}\";\n\n")

file(APPEND "${OUTPUT_CPP}" "static const char kZhSingleDictUtf8[] = R\"${_TTS_SINGLE_DELIM}(\n")
file(APPEND "${OUTPUT_CPP}" "${_TTS_SINGLE_CONTENT}")
file(APPEND "${OUTPUT_CPP}" "\n)${_TTS_SINGLE_DELIM}\";\n\n")

file(APPEND "${OUTPUT_CPP}" "std::string_view zh_builtin_pinyin_phrase_utf8() {\n")
file(APPEND "${OUTPUT_CPP}" "    return std::string_view{kZhPhraseDictUtf8, sizeof(kZhPhraseDictUtf8) - 1};\n")
file(APPEND "${OUTPUT_CPP}" "}\n\n")
file(APPEND "${OUTPUT_CPP}" "std::string_view zh_builtin_pinyin_single_utf8() {\n")
file(APPEND "${OUTPUT_CPP}" "    return std::string_view{kZhSingleDictUtf8, sizeof(kZhSingleDictUtf8) - 1};\n")
file(APPEND "${OUTPUT_CPP}" "}\n\n")

file(APPEND "${OUTPUT_CPP}" "} // namespace kokoro_zh\n")

unset(_TTS_PHRASE_CONTENT)
unset(_TTS_SINGLE_CONTENT)
unset(_TTS_PHRASE_MD5)
unset(_TTS_SINGLE_MD5)
unset(_TTS_PHRASE_MD5_8)
unset(_TTS_SINGLE_MD5_8)
unset(_TTS_PHRASE_DELIM)
unset(_TTS_SINGLE_DELIM)
unset(_TTS_PHRASE_CONFLICT)
unset(_TTS_SINGLE_CONFLICT)
unset(_TTS_OUT_DIR)
