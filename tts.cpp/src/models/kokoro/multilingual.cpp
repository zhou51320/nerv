#include "multilingual.h"

#include "common.h"
#include "phonemizer.h"
#include "ja_frontend.h"
#include "zh_frontend.h"

#include <cctype>
#include <cstdint>
#include <string>
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

static bool is_fullwidth_digit(uint32_t cp) {
    return cp >= 0xFF10 && cp <= 0xFF19;
}

static bool is_digit_cp(uint32_t cp) {
    if (cp <= 0x7F) {
        return is_ascii_digit(static_cast<uint8_t>(cp));
    }
    return is_fullwidth_digit(cp);
}

static bool is_number_separator_cp(uint32_t cp) {
    // '.'/',' 以及全角 '.'/','。
    return cp == '.' || cp == ',' || cp == 0xFF0E || cp == 0xFF0C;
}

static bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n';
}

static std::string_view zh_unit_to_words_lower(std::string_view unit_lower) {
    // 说明：只覆盖“最常见、歧义小”的单位；并且仅在 language=ZH 的多语言前端预处理阶段启用。
    // 若需要扩展更多单位，可继续在这里补充映射（尽量保持键为小写、值为中文词）。
    if (unit_lower == "km")  return "公里";
    if (unit_lower == "m")   return "米";
    if (unit_lower == "cm")  return "厘米";
    if (unit_lower == "mm")  return "毫米";
    if (unit_lower == "kg")  return "千克";
    if (unit_lower == "g")   return "克";
    if (unit_lower == "mg")  return "毫克";
    if (unit_lower == "l")   return "升";
    if (unit_lower == "ml")  return "毫升";
    if (unit_lower == "w")   return "瓦";
    if (unit_lower == "kw")  return "千瓦";
    if (unit_lower == "mw")  return "兆瓦";
    if (unit_lower == "gw")  return "吉瓦";
    if (unit_lower == "wh")  return "瓦时";
    if (unit_lower == "kwh") return "千瓦时";
    if (unit_lower == "mwh") return "兆瓦时";
    if (unit_lower == "gwh") return "吉瓦时";
    if (unit_lower == "v")   return "伏";
    if (unit_lower == "mv")  return "毫伏";
    if (unit_lower == "a")   return "安";
    if (unit_lower == "ma")  return "毫安";
    if (unit_lower == "mah") return "毫安时";
    if (unit_lower == "hz")  return "赫兹";
    if (unit_lower == "khz") return "千赫兹";
    if (unit_lower == "mhz") return "兆赫兹";
    if (unit_lower == "ghz") return "吉赫兹";
    return {};
}

static void ascii_to_lower_inplace(std::string & s) {
    for (char & c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
}

static std::string kokoro_normalize_zh_units_internal(const std::string & text) {
    // 说明：
    // - 目标：把 “23.5°C / 23.5℃” 等常见“数字+单位”写法改写为中文词（例如“摄氏度”），
    //   避免被多语言分段逻辑拆成英文片段（C 被英文 phonemizer 读成 'c'）。
    // - 该函数仅做“轻量规则”，不试图做完整的中文 TN（避免引入复杂依赖）。
    std::string out;
    out.reserve(text.size() * 2);

    bool   after_number = false; // 最近是否刚读过数字（允许中间夹空格/小数点）
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t cp_start = offset;
        uint32_t     cp = 0;
        if (!utf8_decode_next(text, offset, cp)) {
            break;
        }

        // 1) 温度单位：℃ / ℉
        if (cp == 0x2103) { // ℃
            out.append("摄氏度");
            after_number = false;
            continue;
        }
        if (cp == 0x2109) { // ℉
            out.append("华氏度");
            after_number = false;
            continue;
        }

        // 2) 温度单位：°C / °F（允许 ° 与字母之间有空格）
        if (cp == 0x00B0) { // °
            size_t look = offset;
            uint32_t next = 0;
            while (look < text.size()) {
                const size_t saved = look;
                if (!utf8_decode_next(text, look, next)) {
                    look = saved;
                    break;
                }
                if (is_space_cp(next)) {
                    continue;
                }
                break;
            }

            if (next == 'C' || next == 'c') {
                out.append("摄氏度");
                offset = look;
                after_number = false;
                continue;
            }
            if (next == 'F' || next == 'f') {
                out.append("华氏度");
                offset = look;
                after_number = false;
                continue;
            }

            // 单独的 “°”：按中文口语通常读作“度”。
            out.append("度");
            after_number = false;
            continue;
        }

        // 3) 数字/分隔符：用于识别 “数字 + 单位” 的上下文
        if (is_digit_cp(cp) || is_number_separator_cp(cp)) {
            out.append(text.substr(cp_start, offset - cp_start));
            after_number = true;
            continue;
        }
        if (after_number && is_space_cp(cp)) {
            out.append(text.substr(cp_start, offset - cp_start));
            continue;
        }

        // 4) 单位缩写：仅在“紧跟数字后”时尝试，把整段连续字母当作候选单位
        if (after_number && cp <= 0x7F && is_ascii_alpha(static_cast<uint8_t>(cp))) {
            size_t run_end = cp_start;
            while (run_end < text.size()) {
                const uint8_t b = static_cast<uint8_t>(text[run_end]);
                if (b < 0x80 && is_ascii_alpha(b)) {
                    run_end += 1;
                } else {
                    break;
                }
            }
            std::string unit = text.substr(cp_start, run_end - cp_start);
            ascii_to_lower_inplace(unit);
            const std::string_view repl = zh_unit_to_words_lower(unit);
            if (!repl.empty()) {
                out.append(repl);
            } else {
                out.append(text.substr(cp_start, run_end - cp_start));
            }
            offset = run_end;
            after_number = false;
            continue;
        }

        // 其它字符：原样输出，并重置 after_number（避免跨词误触发单位替换）
        out.append(text.substr(cp_start, offset - cp_start));
        after_number = false;
    }

    return out;
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

std::string kokoro_normalize_zh_units(const std::string & text) {
    // 说明：对外暴露一个可复用的“中文单位归一化”函数，主要用于 CLI 打印与快速调试。
    return kokoro_normalize_zh_units_internal(text);
}

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
    const bool digits_as_english = language == tts_language::EN;
    const bool force_zh = language == tts_language::ZH;
    const bool force_ja = language == tts_language::JA;

    // 说明：在中文偏好下，先做一轮“单位/符号”轻量归一化，避免单位被拆到英文片段里按字母读。
    const std::string normalized = force_zh ? kokoro_normalize_zh_units_internal(text) : text;

    std::string out;
    out.reserve(normalized.size() * 2);

    bool first = true;
    size_t i = 0;
    while (i < normalized.size()) {
        const uint8_t c = static_cast<uint8_t>(normalized[i]);

        if (c < 0x80 && is_ascii_word(c, digits_as_english)) {
            // 英文片段：一直读到非英文片段字符为止（language=EN 时允许数字进入）。
            const size_t start = i;
            bool has_word = false;
            while (i < normalized.size()) {
                const uint8_t cc = static_cast<uint8_t>(normalized[i]);
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

            std::string segment = normalized.substr(start, i - start);
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
        while (i < normalized.size()) {
            const uint8_t cc = static_cast<uint8_t>(normalized[i]);
            if (cc < 0x80 && is_ascii_word(cc, digits_as_english)) {
                break;
            }
            ++i;
        }

        std::string segment = normalized.substr(start, i - start);
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
