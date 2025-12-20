#pragma once

#include <string_view>

namespace kokoro_zh {

#if defined(TTS_ZH_DICT_BUILTIN)
// 说明：内置中文词典（UTF-8 文本）。
// - phrase：短语 -> “空格分隔的拼音序列”（每个拼音通常带声调数字）
// - single：单字 -> 拼音（作为 `zh_pinyin_data` 缺失时的兜底）
//
// 这些数据由 CMake 在构建时从 `dict/pinyin_phrase.txt` / `dict/pinyin.txt` 生成并编译进二进制，
// 运行时无需依赖外部文件，方便移植；同时仍支持通过 `--zh-dict-dir` 覆盖/禁用。
std::string_view zh_builtin_pinyin_phrase_utf8();
std::string_view zh_builtin_pinyin_single_utf8();
#endif

} // namespace kokoro_zh

