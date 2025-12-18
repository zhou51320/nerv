#pragma once

#include <cstdint>
#include <string_view>

namespace kokoro_zh {

struct zh_syllable_base {
    std::string_view initial;
    std::string_view final;
    uint8_t          tone = 0;
};

bool lookup_zh_syllable(uint32_t codepoint, zh_syllable_base & out);

} // namespace kokoro_zh
