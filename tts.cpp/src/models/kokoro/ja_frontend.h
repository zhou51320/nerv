#pragma once

#include <string>

namespace kokoro_ja {

// 日文前端（轻量版）：
// - 目标：把日文片段（主要是假名：ひらがな/カタカナ）转换为 Kokoro 可用的“音素字符序列”。
// - 说明：这里不引入 OpenJTalk/MeCab 等第三方依赖，因此“汉字→读音”的自动推断能力有限。
//   建议输入尽量使用假名，或在汉字后用括号提供读音（示例：`東京(とうきょう)`）。
std::string text_to_ja_phonemes(const std::string & text);

} // namespace kokoro_ja

