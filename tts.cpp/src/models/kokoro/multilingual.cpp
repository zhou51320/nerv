#include "multilingual.h"

#include "common.h"
#include "phonemizer.h"
#include "ja_frontend.h"
#include "zh_frontend.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

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

// 说明：将 Unicode 码点追加为 UTF-8 字节序列，便于后续重建文本。
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

static bool is_ascii_wordish_cp(uint32_t cp) {
    if (cp > 0x7F) {
        return false;
    }
    const uint8_t c = static_cast<uint8_t>(cp);
    return is_ascii_alpha(c) || is_ascii_digit(c) || c == '_';
}

static bool should_replace_dot_symbol(uint32_t prev, uint32_t next) {
    // 说明：仅在 ASCII “词内点号”场景读作“点/dot”；纯数字小数点保持原样。
    if (!is_ascii_wordish_cp(prev) || !is_ascii_wordish_cp(next)) {
        return false;
    }
    if (is_ascii_digit(static_cast<uint8_t>(prev)) && is_ascii_digit(static_cast<uint8_t>(next))) {
        return false;
    }
    return true;
}

static bool is_ascii_emailish_cp(uint32_t cp) {
    // 说明：用于识别 email/账号类符号附近的 ASCII 片段。
    if (cp > 0x7F) {
        return false;
    }
    const uint8_t c = static_cast<uint8_t>(cp);
    if (is_ascii_alpha(c) || is_ascii_digit(c)) {
        return true;
    }
    return c == '_' || c == '-' || c == '.' || c == '+';
}

static bool should_replace_at_symbol(uint32_t prev, uint32_t next) {
    // 说明：仅在 ASCII 账号/邮箱场景把 '@' 读作“艾特/at”。
    return is_ascii_emailish_cp(prev) && is_ascii_emailish_cp(next);
}

static bool is_number_separator_cp(uint32_t cp) {
    // '.'/',' 以及全角 '.'/','。
    return cp == '.' || cp == ',' || cp == 0xFF0E || cp == 0xFF0C;
}

static bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n';
}

static bool is_math_operator_cp(uint32_t cp) {
    return cp == '+' || cp == '-' || cp == '*' || cp == '/' || cp == '=' ||
           cp == 0x00D7 || cp == 0x00F7 || cp == 0xFF0B || cp == 0xFF0A ||
           cp == 0xFF0F || cp == 0xFF1D || cp == 'x' || cp == 'X';
}

static bool lookahead_is_digit(const std::string & text, size_t offset) {
    // 说明：跳过空白，查看后续是否出现数字（ASCII/全角）。
    size_t lookahead = offset;
    while (lookahead < text.size()) {
        const size_t saved = lookahead;
        uint32_t cp = 0;
        if (!utf8_decode_next(text, lookahead, cp)) {
            lookahead = saved;
            break;
        }
        if (is_space_cp(cp)) {
            continue;
        }
        return is_digit_cp(cp);
    }
    return false;
}

static bool lookahead_is_math_operator(const std::string & text, size_t offset) {
    // 说明：跳过空白，查看后续是否出现算术运算符。
    size_t lookahead = offset;
    while (lookahead < text.size()) {
        const size_t saved = lookahead;
        uint32_t cp = 0;
        if (!utf8_decode_next(text, lookahead, cp)) {
            lookahead = saved;
            break;
        }
        if (is_space_cp(cp)) {
            continue;
        }
        return is_math_operator_cp(cp);
    }
    return false;
}

static std::string_view zh_unit_to_words_lower(std::string_view unit_lower) {
    // 说明：只覆盖“最常见、歧义小”的单位；并且仅在 language=ZH 的多语言前端预处理阶段启用。
    // 若需要扩展更多单位，可继续在这里补充映射（尽量保持键为小写、值为中文词）。
    if (unit_lower == "km")  return "公里";
    if (unit_lower == "m")   return "米";
    if (unit_lower == "cm")  return "厘米";
    if (unit_lower == "mm")  return "毫米";
    if (unit_lower == "s")   return "秒";
    if (unit_lower == "ms")  return "毫秒";
    if (unit_lower == "us")  return "微秒";
    if (unit_lower == "ns")  return "纳秒";
    if (unit_lower == "min") return "分钟";
    if (unit_lower == "h")   return "小时";
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

static size_t skip_ascii_spaces(const std::string & text, size_t offset) {
    // 说明：单位归一化仅需要处理常见 ASCII 空白即可。
    while (offset < text.size()) {
        const char c = text[offset];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++offset;
            continue;
        }
        break;
    }
    return offset;
}

static bool try_parse_ascii_unit_run(const std::string & text, size_t start, std::string & out_unit, size_t & out_end) {
    // 说明：解析连续 ASCII 字母组成的 unit token（例如 "km"/"m"/"ms"/"kwh"）。
    if (start >= text.size()) {
        return false;
    }
    const uint8_t b0 = static_cast<uint8_t>(text[start]);
    if (!is_ascii_alpha(b0)) {
        return false;
    }

    size_t end = start;
    while (end < text.size()) {
        const uint8_t b = static_cast<uint8_t>(text[end]);
        if (is_ascii_alpha(b)) {
            ++end;
            continue;
        }
        break;
    }

    out_unit = text.substr(start, end - start);
    out_end = end;
    return true;
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

        if (after_number && (cp == 'x' || cp == 'X')) {
            // 说明：数字之间的 x/X 视作“乘”，避免当作英文读音。
            if (lookahead_is_digit(text, offset)) {
                out.append("乘");
                after_number = false;
                continue;
            }
        }

        // 4) 斜杠单位：例如 "m/s" / "km/h" / "mg/L"
        // 说明：
        // - "/" 在单位场景通常读作“每”；但是在 URL/路径等场景则不应强行替换。
        // - 这里采用“分子/分母都能识别为单位缩写”的保守条件来触发替换，尽量避免误伤。
        if (cp <= 0x7F && is_ascii_alpha(static_cast<uint8_t>(cp))) {
            std::string unit1;
            size_t      unit1_end = cp_start;
            if (try_parse_ascii_unit_run(text, cp_start, unit1, unit1_end)) {
                std::string unit1_lower = unit1;
                ascii_to_lower_inplace(unit1_lower);
                const std::string_view w1 = zh_unit_to_words_lower(unit1_lower);
                if (!w1.empty()) {
                    size_t look = skip_ascii_spaces(text, unit1_end);
                    if (look < text.size() && text[look] == '/') {
                        look = skip_ascii_spaces(text, look + 1);
                        std::string unit2;
                        size_t      unit2_end = look;
                        if (try_parse_ascii_unit_run(text, look, unit2, unit2_end)) {
                            std::string unit2_lower = unit2;
                            ascii_to_lower_inplace(unit2_lower);
                            const std::string_view w2 = zh_unit_to_words_lower(unit2_lower);
                            if (!w2.empty()) {
                                out.append(w1);
                                out.append("每");
                                out.append(w2);
                                offset = unit2_end;
                                after_number = false;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // 5) 单位缩写：仅在“紧跟数字后”时尝试，把整段连续字母当作候选单位
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
            if ((run_end - cp_start) == 1 && lookahead_is_math_operator(text, run_end)) {
                // 说明：数字后紧跟单字母且后面是运算符时，视作变量而非单位。
                out.append(text.substr(cp_start, run_end - cp_start));
                offset = run_end;
                after_number = false;
                continue;
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

static std::string normalize_dot_symbol(const std::string & text, tts_language language) {
    // 说明：
    // - 识别 tts.cpp / foo.bar 这类 ASCII“词内点号”，读作“点/ dot”，避免被当成句号跳过。
    // - 识别 email/账号里的 '@'，中文读作“艾特”，英文读作“at”。
    // - 正常句号（例如 "Hello." 或中文 "。"）不做替换。
    if (language != tts_language::ZH && language != tts_language::EN) {
        return text;
    }
    if (text.find('.') == std::string::npos &&
        text.find('@') == std::string::npos &&
        text.find("\xEF\xBC\x8E") == std::string::npos) {
        return text;
    }

    std::vector<uint32_t> cps;
    cps.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        utf8_decode_next(text, offset, cp);
        cps.push_back(cp);
    }

    std::string out;
    out.reserve(text.size() + 8);
    const bool use_zh = (language == tts_language::ZH);
    for (size_t i = 0; i < cps.size(); ++i) {
        const uint32_t cp = cps[i];
        if (cp == '@') {
            // 说明：默认把 '@' 读作“艾特/at”，覆盖邮箱/提及/独立符号等常见场景。
            if (use_zh) {
                out.append("艾特");
            } else {
                out.append(" at ");
            }
            continue;
        }
        if ((cp == '.' || cp == 0xFF0E) && i > 0 && i + 1 < cps.size()) {
            const uint32_t prev = cps[i - 1];
            const uint32_t next = cps[i + 1];
            if (should_replace_dot_symbol(prev, next)) {
                if (use_zh) {
                    out.append("点");
                } else {
                    out.append(" dot ");
                }
                continue;
            }
        }
        utf8_append(out, cp);
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
    const std::string dotted = normalize_dot_symbol(text, language);
    const std::string normalized = force_zh ? kokoro_normalize_zh_units_internal(dotted) : dotted;

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
