#include "zh_frontend.h"

#include "zh_pinyin_data.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace {

static constexpr std::string_view kUnk = "❓";

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

    // 2-byte sequence
    if ((c0 & 0xE0) == 0xC0 && offset + 1 < s.size()) {
        const uint8_t c1 = static_cast<uint8_t>(s[offset + 1]);
        if ((c1 & 0xC0) == 0x80) {
            out_cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
            offset += 2;
            return true;
        }
    }

    // 3-byte sequence
    if ((c0 & 0xF0) == 0xE0 && offset + 2 < s.size()) {
        const uint8_t c1 = static_cast<uint8_t>(s[offset + 1]);
        const uint8_t c2 = static_cast<uint8_t>(s[offset + 2]);
        if (((c1 & 0xC0) == 0x80) && ((c2 & 0xC0) == 0x80)) {
            out_cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
            offset += 3;
            return true;
        }
    }

    // 4-byte sequence
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

    // Invalid; consume 1 byte to make progress.
    out_cp = 0xFFFD;
    offset += 1;
    return true;
}

static void utf8_append(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

static uint32_t normalize_punctuation(uint32_t cp) {
    // Map common CJK / fullwidth punctuation to ASCII or to a token supported by Kokoro.
    switch (cp) {
        case 0x3000: return ' '; // IDEOGRAPHIC SPACE
        case 0x3001: return ','; // 、 -> ,
        case 0x3002: return '.'; // 。 -> .
        case 0xFF0C: return ','; // ， -> ,
        case 0xFF1B: return ';'; // ； -> ;
        case 0xFF1A: return ':'; // ： -> :
        case 0xFF01: return '!'; // ！ -> !
        case 0xFF1F: return '?'; // ？ -> ?
        case 0xFF08: return '('; // （ -> (
        case 0xFF09: return ')'; // ） -> )
        default:     return cp;
    }
}

static std::string_view zh_map(std::string_view token) {
    // Single-character tokens
    if (token.size() == 1) {
        const char c = token[0];
        switch (c) {
            case ' ': return " ";
            case '1': return "1";
            case '2': return "2";
            case '3': return "3";
            case '4': return "4";
            case '5': return "5";
            case ';': return ";";
            case ':': return ":";
            case ',': return ",";
            case '.': return ".";
            case '!': return "!";
            case '?': return "?";
            case '/': return "/";
            case '(': return "(";
            case ')': return ")";
            case 'R': return "R";
            case 'a': return "ㄚ";
            case 'b': return "ㄅ";
            case 'c': return "ㄘ";
            case 'd': return "ㄉ";
            case 'e': return "ㄜ";
            case 'f': return "ㄈ";
            case 'g': return "ㄍ";
            case 'h': return "ㄏ";
            case 'i': return "ㄧ";
            case 'j': return "ㄐ";
            case 'k': return "ㄎ";
            case 'l': return "ㄌ";
            case 'm': return "ㄇ";
            case 'n': return "ㄋ";
            case 'o': return "ㄛ";
            case 'p': return "ㄆ";
            case 'q': return "ㄑ";
            case 'r': return "ㄖ";
            case 's': return "ㄙ";
            case 't': return "ㄊ";
            case 'u': return "ㄨ";
            case 'v': return "ㄩ";
            case 'x': return "ㄒ";
            case 'z': return "ㄗ";
            default:  return kUnk;
        }
    }

    // Multi-character tokens
    if (token == "zh")   return "ㄓ";
    if (token == "ch")   return "ㄔ";
    if (token == "sh")   return "ㄕ";
    if (token == "ai")   return "ㄞ";
    if (token == "ei")   return "ㄟ";
    if (token == "ao")   return "ㄠ";
    if (token == "ou")   return "ㄡ";
    if (token == "an")   return "ㄢ";
    if (token == "en")   return "ㄣ";
    if (token == "ang")  return "ㄤ";
    if (token == "eng")  return "ㄥ";
    if (token == "er")   return "ㄦ";
    if (token == "ie")   return "ㄝ";
    if (token == "ii")   return "ㄭ";
    if (token == "iii")  return "十";
    if (token == "ve")   return "月";
    if (token == "ia")   return "压";
    if (token == "ian")  return "言";
    if (token == "iang") return "阳";
    if (token == "iao")  return "要";
    if (token == "in")   return "阴";
    if (token == "ing")  return "应";
    if (token == "iong") return "用";
    if (token == "iou")  return "又";
    if (token == "ong")  return "中";
    if (token == "ua")   return "穵";
    if (token == "uai")  return "外";
    if (token == "uan")  return "万";
    if (token == "uang") return "王";
    if (token == "uei")  return "为";
    if (token == "uen")  return "文";
    if (token == "ueng") return "瓮";
    if (token == "uo")   return "我";
    if (token == "van")  return "元";
    if (token == "vn")   return "云";

    return kUnk;
}

struct zh_token {
    enum class type { syllable, boundary };
    type                    t = type::boundary;
    uint32_t                cp = 0; // original (for syllable) or normalized punctuation (for boundary)
    kokoro_zh::zh_syllable_base syl{};
};

static bool is_boundary_cp(uint32_t cp) {
    switch (cp) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case ',':
        case '.':
        case '!':
        case '?':
        case ';':
        case ':':
        case '(':
        case ')':
        case '/':
        case 0x2014: // —
        case 0x2026: // …
        case 0x201C: // “
        case 0x201D: // ”
            return true;
        default:
            return false;
    }
}

} // namespace

namespace kokoro_zh {

std::string text_to_zh_phonemes(const std::string & text) {
    // Tokenize into syllables + boundary codepoints.
    std::vector<zh_token> tokens;
    tokens.reserve(text.size());

    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        utf8_decode_next(text, offset, cp);
        cp = normalize_punctuation(cp);

        zh_syllable_base syl{};
        if (lookup_zh_syllable(cp, syl)) {
            zh_token t{};
            t.t = zh_token::type::syllable;
            t.cp = cp;
            t.syl = syl;
            tokens.push_back(t);
        } else {
            zh_token t{};
            t.t = zh_token::type::boundary;
            t.cp = cp;
            tokens.push_back(t);
        }
    }

    // Apply minimal tone sandhi (character-level, no word segmentation).
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        if (tokens[i].t != zh_token::type::syllable) {
            continue;
        }
        if (tokens[i + 1].t != zh_token::type::syllable) {
            continue;
        }

        // 不 + 4th tone -> 2nd tone
        if (tokens[i].cp == 0x4E0D && tokens[i + 1].syl.tone == 4) {
            tokens[i].syl.tone = 2;
        }

        // 一 sandhi: before 4th or neutral -> 2nd, else -> 4th
        if (tokens[i].cp == 0x4E00) {
            const uint8_t next_tone = tokens[i + 1].syl.tone;
            tokens[i].syl.tone = (next_tone == 4 || next_tone == 5) ? 2 : 4;
        }
    }

    // 3rd tone sandhi: for any contiguous run of 3-3-...-3, change all but last to 2.
    for (size_t i = 0; i < tokens.size();) {
        if (tokens[i].t != zh_token::type::syllable || tokens[i].syl.tone != 3) {
            ++i;
            continue;
        }

        size_t j = i;
        while (j < tokens.size() && tokens[j].t == zh_token::type::syllable && tokens[j].syl.tone == 3) {
            ++j;
        }
        const size_t run_len = j - i;
        if (run_len >= 2) {
            for (size_t k = i; k + 1 < j; ++k) {
                tokens[k].syl.tone = 2;
            }
        }
        i = j;
    }

    // Emit Kokoro zh phoneme string.
    std::string out;
    out.reserve(text.size() * 2);

    for (const auto & tok : tokens) {
        if (tok.t == zh_token::type::syllable) {
            if (!tok.syl.initial.empty()) {
                out.append(zh_map(tok.syl.initial));
            }
            out.append(zh_map(tok.syl.final));
            out.push_back(static_cast<char>('0' + tok.syl.tone));
            continue;
        }

        // Keep supported punctuation / whitespace; drop unknown non-CJK symbols.
        if (tok.cp <= 0x7F) {
            const char c = static_cast<char>(tok.cp);
            if (c == '\0') {
                continue;
            }
            out.push_back(c);
            continue;
        }

        if (is_boundary_cp(tok.cp)) {
            utf8_append(out, tok.cp);
        }
    }

    return out;
}

} // namespace kokoro_zh
