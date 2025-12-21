#pragma once

#include <string>
#include <vector>

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

struct zh_debug_item {
    // 归一化后的“原文片段”（通常是一个“词/短语”或一个边界符号）。
    std::string text;
    // 对应的 Kokoro token 串（UTF-8，单字符 token + 声调数字）。
    std::string phonemes;
    // 是否为边界符号（空白/标点等）。边界符号在调试打印时通常不需要括号包裹。
    bool is_boundary = false;
};

struct zh_debug_result {
    // 数字归一化后的文本（例如 23.5 -> 二十三点五）。
    std::string normalized_text;
    // 最终输出的 Kokoro token 串。
    std::string phonemes;
    // 逐字对齐信息（便于观测多音字/声调/卷舌等问题）。
    std::vector<zh_debug_item> items;
};

std::string text_to_zh_phonemes(const std::string & text, const std::string & dict_dir = "");

// 调试版：除了返回音素串，还会提供“归一化文本 + 逐字对齐”的信息。
zh_debug_result text_to_zh_phonemes_debug(const std::string & text, const std::string & dict_dir = "");

} // namespace kokoro_zh
