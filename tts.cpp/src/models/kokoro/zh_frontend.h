#pragma once

#include <string>

// Kokoro 中文（普通话）G2P：将中文文本转换为 Kokoro 可识别的“单字符 token 串”。
//
// 输出约定：
// - 返回 UTF-8 字符串；其中“每个 UTF-8 *字符*”对应 Kokoro 的一个 token；
// - 中文部分主要使用 Bopomofo（注音符号）+ 少量占位汉字 + 声调数字（1~5）；
// - 会尽量保留空白与常见标点（并将中文全角标点归一化为 ASCII），以便 Kokoro 断句。
//
// 说明：
// - 当提供 `dict_dir`（或默认 `dict/` 下存在词典）时，会使用短语拼音词典做多音字消歧、并应用更接近口语的变调/儿化/轻声规则；
// - 若词典不可用，则回退到内置“逐字映射 + 极简变调”的轻量实现（不依赖外部资源）。

namespace kokoro_zh {

std::string text_to_zh_phonemes(const std::string & text, const std::string & dict_dir = "");

} // namespace kokoro_zh
