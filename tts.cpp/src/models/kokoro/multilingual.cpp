#include "multilingual.h"

#include "phonemizer.h"
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

static bool is_english_segment_char(uint8_t c) {
    return is_ascii_alpha(c) || c == ' ' || c == '\'' || c == '-';
}

} // namespace

bool kokoro_contains_cjk(const std::string & text) {
    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        utf8_decode_next(text, offset, cp);
        if (cp >= 0x4E00 && cp <= 0x9FFF) {
            return true;
        }
    }
    return false;
}

std::string kokoro_phonemize_multilingual(const std::string & text, phonemizer * fallback_en_phonemizer) {
    std::string out;
    out.reserve(text.size() * 2);

    bool first = true;
    size_t i = 0;
    while (i < text.size()) {
        const uint8_t c = static_cast<uint8_t>(text[i]);

        if (c < 0x80 && is_ascii_alpha(c)) {
            // English segment: read until a non-english-segment character.
            const size_t start = i;
            bool has_alpha = false;
            while (i < text.size()) {
                const uint8_t cc = static_cast<uint8_t>(text[i]);
                if (cc < 0x80 && is_english_segment_char(cc)) {
                    has_alpha = has_alpha || is_ascii_alpha(cc);
                    ++i;
                } else {
                    break;
                }
            }

            if (!has_alpha) {
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

        // Non-English segment: read until the next ASCII alpha.
        const size_t start = i;
        while (i < text.size()) {
            const uint8_t cc = static_cast<uint8_t>(text[i]);
            if (cc < 0x80 && is_ascii_alpha(cc)) {
                break;
            }
            ++i;
        }

        std::string segment = text.substr(start, i - start);
        std::string phonemes = kokoro_zh::text_to_zh_phonemes(segment);
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

