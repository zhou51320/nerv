#include "multilingual.h"

#include "common.h"
#include "phonemizer.h"
#include "ja_frontend.h"
#include "zh_frontend.h"

#include <cctype>
#include <cstdint>
#include <string_view>

namespace {

static bool utf8_decode_next(const std::string & s, size_t & offset, uint32_t & out_cp) {
    if (offset >= s.size()) {
        return false;
    }

    const uint8_t c0 = static_cast<uint8_t>(s[offset]);
    if (c0 < 0x80) {
        out_cp = c0;
        offset += 1;
        return true;
    }

    if ((c0 & 0xE0) == 0xC0 && offset + 1 < s.size()) {
        const uint8_t c1 = static_cast<uint8_t>(s[offset + 1]);
        if ((c1 & 0xC0) == 0x80) {
            out_cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
            offset += 2;
            return true;
        }
    }

    if ((c0 & 0xF0) == 0xE0 && offset + 2 < s.size()) {
        const uint8_t c1 = static_cast<uint8_t>(s[offset + 1]);
        const uint8_t c2 = static_cast<uint8_t>(s[offset + 2]);
        if (((c1 & 0xC0) == 0x80) && ((c2 & 0xC0) == 0x80)) {
            out_cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
            offset += 3;
            return true;
        }
    }

    if ((c0 & 0xF8) == 0xF0 && offset + 3 < s.size()) {
        const uint8_t c1 = static_cast<uint8_t>(s[offset + 1]);
        const uint8_t c2 = static_cast<uint8_t>(s[offset + 2]);
        const uint8_t c3 = static_cast<uint8_t>(s[offset + 3]);
        if (((c1 & 0xC0) == 0x80) && ((c2 & 0xC0) == 0x80) && ((c3 & 0xC0) == 0x80)) {
            out_cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            offset += 4;
            return true;
        }
    }

    out_cp = 0xFFFD;
    offset += 1;
    return true;
}

static bool is_ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool is_ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool is_ascii_word(uint8_t c, bool digits_as_english) {
    return is_ascii_alpha(c) || (digits_as_english && is_ascii_digit(c));
}

static bool is_english_segment_char(uint8_t c, bool digits_as_english) {
    // 说明：language=EN 时，数字归入英文片段；language=ZH 时，数字交给中文前端做中文读法。
    if (is_ascii_word(c, digits_as_english)) {
        return true;
    }
    if (c == ' ' || c == '\'' || c == '-') {
        return true;
    }
    return digits_as_english && (c == '.' || c == ',');
}

static bool is_japanese_kana(uint32_t cp) {
    // 说明：只做常见范围检测即可（平假名/片假名/片假名扩展）。
    if (cp >= 0x3040 && cp <= 0x309F) { // Hiragana
        return true;
    }
    if (cp >= 0x30A0 && cp <= 0x30FF) { // Katakana
        return true;
    }
    if (cp >= 0x31F0 && cp <= 0x31FF) { // Katakana Phonetic Extensions
        return true;
    }
    return false;
}

static bool contains_japanese_kana(const std::string & text) {
    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        utf8_decode_next(text, offset, cp);
        if (is_japanese_kana(cp)) {
            return true;
        }
    }
    return false;
}

} // namespace

bool kokoro_contains_cjk(const std::string & text) {
    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        utf8_decode_next(text, offset, cp);
        // 说明：这里的 “CJK” 用于决定是否启用 Kokoro 的多语言前端，包含：汉字 + 日文假名。
        if ((cp >= 0x4E00 && cp <= 0x9FFF) || is_japanese_kana(cp)) {
            return true;
        }
    }
    return false;
}

std::string kokoro_phonemize_multilingual(const std::string & text,
                                          phonemizer * fallback_en_phonemizer,
                                          tts_language language,
                                          const std::string & zh_dict_dir) {
    std::string out;
    out.reserve(text.size() * 2);

    const bool digits_as_english = language == tts_language::EN;
    const bool force_zh = language == tts_language::ZH;
    const bool force_ja = language == tts_language::JA;

    bool first = true;
    size_t i = 0;
    while (i < text.size()) {
        const uint8_t c = static_cast<uint8_t>(text[i]);

        if (c < 0x80 && is_ascii_word(c, digits_as_english)) {
            // 英文片段：一直读到非英文片段字符为止（language=EN 时允许数字进入）。
            const size_t start = i;
            bool has_word = false;
            while (i < text.size()) {
                const uint8_t cc = static_cast<uint8_t>(text[i]);
                if (cc < 0x80 && is_english_segment_char(cc, digits_as_english)) {
                    has_word = has_word || is_ascii_word(cc, digits_as_english);
                    ++i;
                } else {
                    break;
                }
            }

            if (!has_word) {
                continue;
            }

            std::string segment = text.substr(start, i - start);
            std::string phonemes = fallback_en_phonemizer ? fallback_en_phonemizer->text_to_phonemes(segment) : "";
            if (!phonemes.empty()) {
                if (!first && !out.empty() && out.back() != ' ') {
                    out.push_back(' ');
                }
                out.append(phonemes);
                first = false;
            }
            continue;
        }

        // 非英文片段：一直读到下一个 ASCII 字母/数字（按 language 配置）为止。
        const size_t start = i;
        while (i < text.size()) {
            const uint8_t cc = static_cast<uint8_t>(text[i]);
            if (cc < 0x80 && is_ascii_word(cc, digits_as_english)) {
                break;
            }
            ++i;
        }

        std::string segment = text.substr(start, i - start);
        const bool use_ja = force_ja || (!force_zh && contains_japanese_kana(segment));
        std::string phonemes = use_ja ? kokoro_ja::text_to_ja_phonemes(segment)
                                      : kokoro_zh::text_to_zh_phonemes(segment, zh_dict_dir);
        if (!phonemes.empty()) {
            if (!first && !out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
            out.append(phonemes);
            first = false;
        }
    }

    return out;
}
