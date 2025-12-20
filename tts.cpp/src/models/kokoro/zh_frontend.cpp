#include "zh_frontend.h"

#include "zh_dict_builtin.h"
#include "zh_pinyin_data.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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

static bool is_ascii_digit(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

static bool is_fullwidth_digit(uint32_t cp) {
    return cp >= 0xFF10 && cp <= 0xFF19;
}

static bool is_digit_cp(uint32_t cp) {
    return is_ascii_digit(cp) || is_fullwidth_digit(cp);
}

static int digit_value(uint32_t cp) {
    if (is_ascii_digit(cp)) {
        return static_cast<int>(cp - '0');
    }
    if (is_fullwidth_digit(cp)) {
        return static_cast<int>(cp - 0xFF10);
    }
    return -1;
}

static bool is_number_separator(uint32_t cp) {
    return cp == '.' || cp == ',' || cp == 0xFF0E || cp == 0xFF0C;
}

static bool is_minus_sign(uint32_t cp) {
    return cp == '-' || cp == 0xFF0D || cp == 0x2212;
}

static char normalize_number_separator(uint32_t cp) {
    if (cp == 0xFF0E) {
        return '.';
    }
    if (cp == 0xFF0C) {
        return ',';
    }
    return static_cast<char>(cp);
}

static bool collect_number_token(const std::string & text, size_t start_offset, std::string & out_token, size_t & out_next_offset) {
    // 说明：解析连续数字（含英文/全角），并在数字之间保留 '.'/',' 作为分隔符。
    out_token.clear();
    size_t offset = start_offset;
    bool has_digit = false;
    while (offset < text.size()) {
        const size_t cp_start = offset;
        uint32_t cp = 0;
        if (!utf8_decode_next(text, offset, cp)) {
            break;
        }
        if (is_digit_cp(cp)) {
            const int val = digit_value(cp);
            if (val >= 0) {
                out_token.push_back(static_cast<char>('0' + val));
                has_digit = true;
                continue;
            }
        }
        if (is_number_separator(cp) && has_digit) {
            size_t lookahead = offset;
            uint32_t next_cp = 0;
            if (utf8_decode_next(text, lookahead, next_cp) && is_digit_cp(next_cp)) {
                out_token.push_back(normalize_number_separator(cp));
                offset = lookahead;
                continue;
            }
        }
        offset = cp_start;
        break;
    }
    out_next_offset = offset;
    return has_digit;
}

static constexpr std::string_view kZhDigits[] = {
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九"
};

static constexpr std::string_view kZhSmallUnits[] = {
    "", "十", "百", "千"
};

static constexpr std::string_view kZhGroupUnits[] = {
    "", "万", "亿", "兆"
};
static constexpr size_t kZhSeriesThreshold = 5;

static bool all_zero_digits(const std::string & digits) {
    for (char c : digits) {
        if (c != '0') {
            return false;
        }
    }
    return true;
}

static std::string digits_to_zh_series(const std::string & digits) {
    // 说明：逐位读数字（常用于前导 0、编号等场景）。
    std::string out;
    out.reserve(digits.size() * 3);
    for (char c : digits) {
        if (c >= '0' && c <= '9') {
            out.append(kZhDigits[c - '0']);
        }
    }
    return out;
}

static std::string convert_four_digits(const std::string & digits) {
    // 说明：将 1-4 位数字转换为中文读法（千/百/十/个），内部处理“零”衔接。
    std::string out;
    bool zero_pending = false;
    const int len = static_cast<int>(digits.size());
    for (int i = 0; i < len; ++i) {
        const int d = digits[i] - '0';
        const int pos = len - 1 - i; // 0: 个, 1: 十, 2: 百, 3: 千
        if (d == 0) {
            if (!out.empty()) {
                zero_pending = true;
            }
            continue;
        }
        if (zero_pending) {
            out.append(kZhDigits[0]);
            zero_pending = false;
        }
        if (pos == 1 && d == 1 && out.empty()) {
            out.append("十");
        } else {
            out.append(kZhDigits[d]);
            out.append(kZhSmallUnits[pos]);
        }
    }
    return out;
}

static std::string convert_integer_to_zh(const std::string & digits, bool allow_leading_zero_series) {
    // 说明：整数部分按“万/亿/兆”分组转换；超长/超出分组范围或带前导 0 时退化为逐位读取。
    if (digits.empty()) {
        return "";
    }
    if (digits.size() > kZhSeriesThreshold) {
        return digits_to_zh_series(digits);
    }
    if (all_zero_digits(digits)) {
        if (allow_leading_zero_series && digits.size() > 1) {
            return digits_to_zh_series(digits);
        }
        return std::string(kZhDigits[0]);
    }
    if (allow_leading_zero_series && digits.size() > 1 && digits[0] == '0') {
        return digits_to_zh_series(digits);
    }

    const size_t group_count = (digits.size() + 3) / 4;
    const size_t max_groups = sizeof(kZhGroupUnits) / sizeof(kZhGroupUnits[0]);
    if (group_count > max_groups) {
        return digits_to_zh_series(digits);
    }

    std::vector<std::string> groups;
    groups.reserve(group_count);
    for (size_t i = digits.size(); i > 0; i -= 4) {
        const size_t start = (i >= 4) ? (i - 4) : 0;
        groups.push_back(digits.substr(start, i - start));
        if (start == 0) {
            break;
        }
    }

    std::string out;
    bool zero_between = false;
    for (int idx = static_cast<int>(groups.size()) - 1; idx >= 0; --idx) {
        const std::string & group = groups[idx];
        int group_value = 0;
        for (char c : group) {
            group_value = group_value * 10 + (c - '0');
        }
        if (group_value == 0) {
            zero_between = true;
            continue;
        }
        if (!out.empty()) {
            if (zero_between || group_value < 1000) {
                out.append(kZhDigits[0]);
            }
        }
        out.append(convert_four_digits(group));
        out.append(kZhGroupUnits[idx]);
        zero_between = false;
    }
    return out;
}

static std::string number_token_to_zh(const std::string & token) {
    // 说明：支持整数/小数；',' 主要作为千分位分隔，若仅一处且不满足 3 位分组则视为小数点。
    if (token.empty()) {
        return "";
    }
    const bool has_dot = token.find('.') != std::string::npos;
    const bool has_comma = token.find(',') != std::string::npos;
    size_t decimal_pos = std::string::npos;
    if (has_dot) {
        decimal_pos = token.rfind('.');
    } else if (has_comma) {
        const size_t first = token.find(',');
        const size_t last = token.rfind(',');
        if (first == last) {
            size_t digits_after = 0;
            for (size_t i = last + 1; i < token.size(); ++i) {
                if (token[i] >= '0' && token[i] <= '9') {
                    digits_after += 1;
                }
            }
            if (digits_after != 3) {
                decimal_pos = last;
            }
        }
    }

    std::string int_digits;
    std::string frac_digits;
    for (size_t i = 0; i < token.size(); ++i) {
        const char c = token[i];
        if (c < '0' || c > '9') {
            continue;
        }
        if (decimal_pos != std::string::npos && i > decimal_pos) {
            frac_digits.push_back(c);
        } else {
            int_digits.push_back(c);
        }
    }

    if (int_digits.empty() && frac_digits.empty()) {
        return "";
    }

    const bool has_decimal = decimal_pos != std::string::npos && !frac_digits.empty();
    std::string int_read = convert_integer_to_zh(int_digits, !has_decimal);
    if (int_read.empty()) {
        int_read = std::string(kZhDigits[0]);
    }

    if (!has_decimal) {
        return int_read;
    }

    std::string out = int_read;
    out.append("点");
    out.append(digits_to_zh_series(frac_digits));
    return out;
}

static std::string normalize_zh_numbers(const std::string & text) {
    // 说明：将 ASCII/全角数字转换为中文读法，避免直接进入声调数字通道。
    std::string out;
    out.reserve(text.size() * 2);
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t start = offset;
        uint32_t cp = 0;
        if (!utf8_decode_next(text, offset, cp)) {
            break;
        }

        if (is_minus_sign(cp)) {
            size_t lookahead = offset;
            uint32_t next_cp = 0;
            if (utf8_decode_next(text, lookahead, next_cp) && is_digit_cp(next_cp)) {
                std::string token;
                size_t next_offset = offset;
                if (collect_number_token(text, offset, token, next_offset)) {
                    const std::string zh = number_token_to_zh(token);
                    if (!zh.empty()) {
                        out.append("负");
                        out.append(zh);
                        offset = next_offset;
                        continue;
                    }
                }
            }
            out.append(text.substr(start, offset - start));
            continue;
        }

        if (is_digit_cp(cp)) {
            std::string token;
            size_t next_offset = start;
            if (collect_number_token(text, start, token, next_offset)) {
                const std::string zh = number_token_to_zh(token);
                if (!zh.empty()) {
                    out.append(zh);
                    offset = next_offset;
                    continue;
                }
            }
        }

        out.append(text.substr(start, offset - start));
    }
    return out;
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

// -----------------------------
// 参考项目 kokoro.cpp-main 的“词典 + DP + 变调”思路（本仓库的轻量集成版）
// -----------------------------
//
// 说明：
// - Kokoro 的 tokenizer 是“UTF-8 单字符 token”，因此中文前端的输出必须是“每个 Unicode 字符都是一个 token”；
// - 本文件已有的 `zh_map()` 负责把拼音 token（initial/final/R/tone）映射到 Kokoro 支持的字符集合；
// - 这里引入短语拼音词典（pinyin_phrase.txt）做多音字消歧，并在“词段”上应用更接近口语的变调/儿化/轻声。

struct zh_syllable {
    // 说明：initial/final 是“token 名称”，最终会通过 zh_map() 映射到输出字符。
    std::string initial;
    std::string final;
    uint8_t     tone  = 5;     // 1~5（5=轻声）
    bool        erhua = false; // 儿化：输出时在声调数字前插入 'R'
};

struct u16string_hash {
    size_t operator()(const std::u16string & s) const noexcept {
        // 简单 FNV-1a：用 UTF-16 code unit 做哈希，足以支撑短语词典查找。
        uint64_t h = 14695981039346656037ull;
        for (char16_t c : s) {
            h ^= static_cast<uint16_t>(c);
            h *= 1099511628211ull;
        }
        return static_cast<size_t>(h);
    }
};

static std::string_view trim_ascii(std::string_view v) {
    while (!v.empty()) {
        const char c = v.front();
        if (c == ' ' || c == '\t') {
            v.remove_prefix(1);
            continue;
        }
        break;
    }
    while (!v.empty()) {
        const char c = v.back();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            v.remove_suffix(1);
            continue;
        }
        break;
    }
    return v;
}

static void split_pinyin_sequence(const std::string & s, std::vector<std::string> & out) {
    // 说明：把 "zhao2 huo3 le5" 拆成 ["zhao2","huo3","le5"]。
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
            ++i;
        }
        if (i >= s.size()) {
            break;
        }
        const size_t start = i;
        while (i < s.size() && s[i] != ' ' && s[i] != '\t') {
            ++i;
        }
        if (i > start) {
            out.emplace_back(s.substr(start, i - start));
        }
    }
}

static void ascii_to_lower_inplace(std::string & s) {
    for (char & c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
}

static void replace_all_inplace(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static bool parse_pinyin_syllable(std::string_view in, zh_syllable & out) {
    // 说明：把“标准拼音（通常带声调数字）”解析成 zh_frontend 的 (initial, final, tone) token。
    // 目标：尽量与 `zh_pinyin_data` 的 token 命名一致，复用 `zh_map()` 输出到 Kokoro token 串。
    std::string s{trim_ascii(in)};
    if (s.empty()) {
        return false;
    }

    ascii_to_lower_inplace(s);

    // 兼容 ü / u: 写法（词典通常使用 v，但这里做个兜底）。
    replace_all_inplace(s, "u:", "v");
    // 说明：避免在部分编译器/标准下 `u8"..."` 变成 `char8_t[]` 造成 `std::string` 构造失败，
    // 这里直接用 UTF-8 字节序列表示 ü/Ü。
    replace_all_inplace(s, "\xC3\xBC", "v"); // ü
    replace_all_inplace(s, "\xC3\x9C", "v"); // Ü

    uint8_t tone = 5;
    if (!s.empty() && std::isdigit(static_cast<unsigned char>(s.back()))) {
        const int t = s.back() - '0';
        if (t >= 1 && t <= 5) {
            tone = static_cast<uint8_t>(t);
        }
        s.pop_back();
    }
    if (s.empty()) {
        return false;
    }

    // y/w 归一化：把拼音正字法写法转换为显式韵母形式。
    if (s.rfind("yi", 0) == 0) {
        // yi -> i
        s.erase(0, 1);
    } else if (!s.empty() && s[0] == 'y') {
        if (s.size() >= 2 && s[1] == 'u') {
            // yu* -> v*
            s = "v" + s.substr(2);
        } else {
            // ya/yo/you/... -> i + a/o/ou/...
            s = "i" + s.substr(1);
        }
    } else if (s.rfind("wu", 0) == 0) {
        // wu -> u
        s.erase(0, 1);
    } else if (!s.empty() && s[0] == 'w') {
        // wa/wo/wei/... -> u + a/o/ei/...
        s = "u" + s.substr(1);
    }

    std::string initial;
    if (s.rfind("zh", 0) == 0 || s.rfind("ch", 0) == 0 || s.rfind("sh", 0) == 0) {
        initial = s.substr(0, 2);
    } else {
        const char c0 = s[0];
        switch (c0) {
            case 'b':
            case 'c':
            case 'd':
            case 'f':
            case 'g':
            case 'h':
            case 'j':
            case 'k':
            case 'l':
            case 'm':
            case 'n':
            case 'p':
            case 'q':
            case 'r':
            case 's':
            case 't':
            case 'x':
            case 'z':
                initial.assign(1, c0);
                break;
            default:
                break;
        }
    }

    std::string final = s.substr(initial.size());

    // i/u/v 的常见简写归一化：
    // - ui -> uei（shui -> shuei）
    // - iu -> iou（liu -> liou）
    // - un -> uen（lun -> luen），但 j/q/x + un 实际是 ün -> vn（jun -> jvn）
    if (final == "ui") {
        final = "uei";
    } else if (final == "iu") {
        final = "iou";
    } else if (final == "un") {
        if (initial == "j" || initial == "q" || initial == "x") {
            final = "vn";
        } else {
            final = "uen";
        }
    } else if (final == "ue") {
        final = "ve";
    } else if (final == "uan") {
        if (initial == "j" || initial == "q" || initial == "x") {
            final = "van";
        }
    } else if (final == "u") {
        if (initial == "j" || initial == "q" || initial == "x") {
            final = "v";
        }
    }

    // 特例：z/c/s 与 zh/ch/sh/r 的舌尖元音 i（misaki[zh] 用 ii/iii 区分）。
    if (final == "i") {
        if (initial == "z" || initial == "c" || initial == "s") {
            final = "ii";
        } else if (initial == "zh" || initial == "ch" || initial == "sh" || initial == "r") {
            final = "iii";
        }
    }

    out.initial = initial;
    out.final = final;
    out.tone = tone;
    out.erhua = false;
    return true;
}

static void utf16_append(std::u16string & out, uint32_t cp) {
    if (cp <= 0xFFFF) {
        out.push_back(static_cast<char16_t>(cp));
        return;
    }
    // surrogate pair
    cp -= 0x10000;
    out.push_back(static_cast<char16_t>(0xD800 + ((cp >> 10) & 0x3FF)));
    out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
}

static void utf8_to_u16(const std::string & s, std::u16string & out) {
    out.clear();
    out.reserve(s.size());
    size_t offset = 0;
    while (offset < s.size()) {
        uint32_t cp = 0;
        if (!utf8_decode_next(s, offset, cp)) {
            break;
        }
        utf16_append(out, cp);
    }
}

struct zh_pinyin_dict {
    // 说明：
    // - phrase：短语 -> “空格分隔的拼音序列”（每个拼音通常带声调数字）
    // - single：单字 -> 拼音（作为 `zh_pinyin_data` 缺失时的兜底）
    // loaded_dir 作为缓存 key（通常是目录名；也可能是 ":builtin" 或空字符串等特殊值）。
    std::string loaded_dir;
    bool loaded = false;
    bool ok = false;
    bool from_builtin = false;

    std::unordered_map<std::u16string, std::string, u16string_hash> phrase;
    std::unordered_map<char16_t, std::string>                       single;

    // 与参考实现一致：只保留 <=8 字短语（PinyinFinder::kMaxChars=8）。
    static constexpr size_t kMaxPhraseChars = 8;

    static std::string join_path(const std::string & dir, const char * filename) {
        if (dir.empty()) {
            return std::string{filename};
        }
        const char last = dir.back();
        if (last == '/' || last == '\\') {
            return dir + filename;
        }
        return dir + "/" + filename;
    }

    template <typename Fn>
    static void for_each_line(std::string_view data, Fn && fn) {
        // 说明：按 \n 分行，不复制大块数据；每行会去掉末尾 \r（兼容 CRLF）。
        size_t pos = 0;
        while (pos < data.size()) {
            size_t end = data.find('\n', pos);
            if (end == std::string_view::npos) {
                end = data.size();
            }
            std::string_view line = data.substr(pos, end - pos);
            if (!line.empty() && line.back() == '\r') {
                line.remove_suffix(1);
            }
            fn(line);
            pos = (end == data.size()) ? end : (end + 1);
        }
    }

    void parse_phrase_line(std::string_view line) {
        if (line.empty() || line[0] == '#') {
            return;
        }
        const size_t comment = line.find('#');
        if (comment != std::string_view::npos) {
            line = line.substr(0, comment);
        }
        const size_t colon = line.find(':');
        if (colon == std::string_view::npos) {
            return;
        }
        const std::string_view wv = trim_ascii(line.substr(0, colon));
        const std::string_view pv = trim_ascii(line.substr(colon + 1));
        if (wv.empty() || pv.empty()) {
            return;
        }

        std::u16string w16;
        utf8_to_u16(std::string(wv), w16);
        if (w16.empty() || w16.size() > kMaxPhraseChars) {
            return;
        }
        phrase.emplace(std::move(w16), std::string(pv));
    }

    void parse_single_line(std::string_view line) {
        if (line.empty() || line[0] == '#') {
            return;
        }
        if (line.rfind("U+", 0) != 0) {
            return;
        }
        const size_t colon = line.find(':');
        if (colon == std::string_view::npos || colon <= 2) {
            return;
        }

        const std::string hex = std::string(line.substr(2, colon - 2));
        uint32_t          cp = 0;
        try {
            cp = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
        } catch (...) {
            return;
        }

        const size_t comment = line.find('#', colon + 1);
        const size_t end = (comment == std::string_view::npos) ? line.size() : comment;
        std::string_view pv = trim_ascii(line.substr(colon + 1, end - (colon + 1)));
        if (pv.empty()) {
            return;
        }
        const size_t comma = pv.find(',');
        if (comma != std::string_view::npos) {
            pv = trim_ascii(pv.substr(0, comma));
        }
        if (pv.empty()) {
            return;
        }
        if (cp <= 0xFFFF) {
            single.emplace(static_cast<char16_t>(cp), std::string(pv));
        }
    }

    bool load_from_memory(std::string_view phrase_data, std::string_view single_data) {
        ok = false;
        phrase.clear();
        single.clear();

        // 说明：reserve 只是优化，避免大量 rehash；容量取值与文件加载路径一致。
        phrase.reserve(420000);
        single.reserve(28000);

        for_each_line(phrase_data, [this](std::string_view line) { parse_phrase_line(line); });
        for_each_line(single_data, [this](std::string_view line) { parse_single_line(line); });

        ok = !phrase.empty();
        return ok;
    }

    bool load_from_builtin(const std::string & key_for_cache) {
        // 说明：把内置词典也“挂载”为一个缓存 key（便于避免反复解析）。
        loaded_dir = key_for_cache;
        loaded = true;
        from_builtin = true;

#if defined(TTS_ZH_DICT_BUILTIN)
        return load_from_memory(kokoro_zh::zh_builtin_pinyin_phrase_utf8(), kokoro_zh::zh_builtin_pinyin_single_utf8());
#else
        // 说明：未启用内置词典时，保持失败。
        return false;
#endif
    }

    bool load_from_dir(const std::string & dir) {
        loaded_dir = dir;
        loaded = true;
        from_builtin = false;
        ok = false;
        phrase.clear();
        single.clear();

        // 1) 短语词典（多音字消歧）
        {
            const std::string path = join_path(dir, "pinyin_phrase.txt");
            std::ifstream     fin(path);
            if (fin.good()) {
                phrase.reserve(420000);
                std::string line;
                while (std::getline(fin, line)) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    parse_phrase_line(line);
                }
            }
        }

        // 2) 单字兜底（pinyin.txt）
        {
            const std::string path = join_path(dir, "pinyin.txt");
            std::ifstream     fin(path);
            if (fin.good()) {
                single.reserve(28000);
                std::string line;
                while (std::getline(fin, line)) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    parse_single_line(line);
                }
            }
        }

        ok = !phrase.empty();
        return ok;
    }
};

static std::mutex     g_zh_dict_mutex;
static zh_pinyin_dict g_zh_dict;

static const zh_pinyin_dict * zh_try_get_dict(const std::string & dict_dir) {
    // 说明：
    // - dict_dir 为缓存 key（同一个 key 只解析一次，避免每次生成都反复扫 10MB+ 词典）。
    // - 若启用内置词典（TTS_ZH_DICT_BUILTIN），则加载顺序为：外部目录 -> 内置回退。
    // - 约定：dict_dir == ":builtin" 可强制使用内置词典（无视外部目录）。
    std::lock_guard<std::mutex> lock(g_zh_dict_mutex);
    if (g_zh_dict.loaded && g_zh_dict.loaded_dir == dict_dir) {
        return g_zh_dict.ok ? &g_zh_dict : nullptr;
    }

    if (dict_dir == ":builtin") {
        if (!g_zh_dict.load_from_builtin(dict_dir)) {
            return nullptr;
        }
        return &g_zh_dict;
    }

    if (g_zh_dict.load_from_dir(dict_dir)) {
        return &g_zh_dict;
    }

    // 外部目录加载失败：如果编译时启用了内置词典，则回退到内置词典。
    if (g_zh_dict.load_from_builtin(dict_dir)) {
        return &g_zh_dict;
    }

    return nullptr;
}

static bool is_numeric_zh_char(char16_t c) {
    // 说明：用于“一”变调的“纯数字串”判断，避免“编号/逐位读”时错误变调。
    if (c >= u'0' && c <= u'9') {
        return true;
    }
    switch (c) {
        case u'零':
        case u'一':
        case u'二':
        case u'三':
        case u'四':
        case u'五':
        case u'六':
        case u'七':
        case u'八':
        case u'九':
        case u'十':
        case u'百':
        case u'千':
        case u'万':
        case u'亿':
        case u'兆':
        case u'点':
        case u'负':
            return true;
        default:
            return false;
    }
}

static const std::unordered_set<std::u16string, u16string_hash> & zh_must_neutral_words() {
    // 参考：参考项目 kokoro.cpp-main/ToneSandhi.cpp 的 must_neural_tone_words
    static const std::unordered_set<std::u16string, u16string_hash> k = {
        u"麻烦", u"麻利", u"鸳鸯", u"高粱", u"骨头", u"骆驼", u"马虎", u"首饰", u"馒头", u"馄饨", u"风筝",
        u"难为", u"队伍", u"阔气", u"闺女", u"门道", u"锄头", u"铺盖", u"铃铛", u"铁匠", u"钥匙", u"里脊",
        u"里头", u"部分", u"那么", u"道士", u"造化", u"迷糊", u"连累", u"这么", u"这个", u"运气", u"过去",
        u"软和", u"转悠", u"踏实", u"跳蚤", u"跟头", u"趔趄", u"财主", u"豆腐", u"讲究", u"记性", u"记号",
        u"认识", u"规矩", u"见识", u"裁缝", u"补丁", u"衣裳", u"衣服", u"衙门", u"街坊", u"行李", u"行当",
        u"蛤蟆", u"蘑菇", u"薄荷", u"葫芦", u"葡萄", u"萝卜", u"荸荠", u"苗条", u"苗头", u"苍蝇", u"芝麻",
        u"舒服", u"舒坦", u"舌头", u"自在", u"膏药", u"脾气", u"脑袋", u"脊梁", u"能耐", u"胳膊", u"胭脂",
        u"胡萝", u"胡琴", u"胡同", u"聪明", u"耽误", u"耽搁", u"耷拉", u"耳朵", u"老爷", u"老实", u"老婆",
        u"戏弄", u"将军", u"翻腾", u"罗嗦", u"罐头", u"编辑", u"结实", u"红火", u"累赘", u"糨糊", u"糊涂",
        u"精神", u"粮食", u"簸箕", u"篱笆", u"算计", u"算盘", u"答应", u"笤帚", u"笑语", u"笑话", u"窟窿",
        u"窝囊", u"窗户", u"稳当", u"稀罕", u"称呼", u"秧歌", u"秀气", u"秀才", u"福气", u"祖宗", u"砚台",
        u"码头", u"石榴", u"石头", u"石匠", u"知识", u"眼睛", u"眯缝", u"眨巴", u"眉毛", u"相声", u"盘算",
        u"白净", u"痢疾", u"痛快", u"疟疾", u"疙瘩", u"疏忽", u"畜生", u"生意", u"甘蔗", u"琵琶", u"琢磨",
        u"琉璃", u"玻璃", u"玫瑰", u"玄乎", u"狐狸", u"状元", u"特务", u"牲口", u"牙碜", u"牌楼", u"爽快",
        u"爱人", u"热闹", u"烧饼", u"烟筒", u"烂糊", u"点心", u"炊帚", u"灯笼", u"火候", u"漂亮", u"滑溜",
        u"溜达", u"温和", u"清楚", u"消息", u"浪头", u"活泼", u"比方", u"正经", u"欺负", u"模糊", u"槟榔",
        u"棺材", u"棒槌", u"棉花", u"核桃", u"栅栏", u"柴火", u"架势", u"枕头", u"枇杷", u"机灵", u"本事",
        u"木头", u"木匠", u"朋友", u"月饼", u"月亮", u"暖和", u"明白", u"时候", u"新鲜", u"故事", u"收拾",
        u"收成", u"提防", u"挖苦", u"挑剔", u"指甲", u"指头", u"拾掇", u"拳头", u"拨弄", u"招牌", u"招呼",
        u"抬举", u"护士", u"折腾", u"扫帚", u"打量", u"打算", u"打扮", u"打听", u"打发", u"扎实", u"扁担",
        u"戒指", u"懒得", u"意识", u"意思", u"悟性", u"怪物", u"思量", u"怎么", u"念头", u"念叨", u"别人",
        u"快活", u"忙活", u"志气", u"心思", u"得罪", u"张罗", u"弟兄", u"开通", u"应酬", u"庄稼", u"干事",
        u"帮手", u"帐篷", u"希罕", u"师父", u"师傅", u"巴结", u"巴掌", u"差事", u"工夫", u"岁数", u"屁股",
        u"尾巴", u"少爷", u"小气", u"小伙", u"将就", u"对头", u"对付", u"寡妇", u"家伙", u"客气", u"实在",
        u"官司", u"学问", u"字号", u"嫁妆", u"媳妇", u"媒人", u"婆家", u"娘家", u"委屈", u"姑娘", u"姐夫",
        u"妯娌", u"妥当", u"妖精", u"奴才", u"女婿", u"头发", u"太阳", u"大爷", u"大方", u"大意", u"大夫",
        u"多少", u"多么", u"外甥", u"壮实", u"地道", u"地方", u"在乎", u"困难", u"嘴巴", u"嘱咐", u"嘟囔",
        u"嘀咕", u"喜欢", u"喇嘛", u"喇叭", u"商量", u"唾沫", u"哑巴", u"哈欠", u"哆嗦", u"咳嗽", u"和尚",
        u"告诉", u"告示", u"含糊", u"吓唬", u"后头", u"名字", u"名堂", u"合同", u"吆喝", u"叫唤", u"口袋",
        u"厚道", u"厉害", u"千斤", u"包袱", u"包涵", u"匀称", u"勤快", u"动静", u"动弹", u"功夫", u"力气",
        u"前头", u"刺猬", u"刺激", u"别扭", u"利落", u"利索", u"利害", u"分析", u"出息", u"凑合", u"凉快",
        u"冷战", u"冤枉", u"冒失", u"养活", u"关系", u"先生", u"兄弟", u"便宜", u"使唤", u"佩服", u"作坊",
        u"体面", u"位置", u"似的", u"伙计", u"休息", u"什么", u"人家", u"亲戚", u"亲家", u"交情", u"云彩",
        u"事情", u"买卖", u"主意", u"丫头", u"丧气", u"两口", u"东西", u"东家", u"世故", u"不由", u"下水",
        u"下巴", u"上头", u"上司", u"丈夫", u"丈人", u"一辈", u"那个", u"菩萨", u"父亲", u"母亲", u"咕噜",
        u"邋遢", u"费用", u"冤家", u"甜头", u"介绍", u"荒唐", u"大人", u"泥鳅", u"幸福", u"熟悉", u"计划",
        u"扑腾", u"蜡烛", u"姥爷", u"照顾", u"喉咙", u"吉他", u"弄堂", u"蚂蚱", u"凤凰", u"拖沓", u"寒碜",
        u"糟蹋", u"倒腾", u"报复", u"逻辑", u"盘缠", u"喽啰", u"牢骚", u"咖喱", u"扫把", u"惦记"
    };
    return k;
}

static const std::unordered_set<std::u16string, u16string_hash> & zh_must_not_neutral_words() {
    // 参考：参考项目 kokoro.cpp-main/ToneSandhi.cpp 的 must_not_neural_tone_words
    static const std::unordered_set<std::u16string, u16string_hash> k = {
        u"男子", u"女子", u"分子", u"原子", u"量子", u"莲子", u"石子", u"瓜子", u"电子", u"人人", u"虎虎",
        u"幺幺", u"干嘛", u"学子", u"哈哈", u"数数", u"袅袅", u"局地", u"以下", u"娃哈哈", u"花花草草", u"留得",
        u"耕地", u"想想", u"熙熙", u"攘攘", u"卵子", u"死死", u"冉冉", u"恳恳", u"佼佼", u"吵吵", u"打打",
        u"考考", u"整整", u"莘莘", u"落地", u"算子", u"家家户户", u"青青"
    };
    return k;
}

static const std::unordered_set<std::u16string, u16string_hash> & zh_must_erhua_words() {
    // 参考：参考项目 kokoro.cpp-main/ZHFrontend.cpp 的 must_erhua
    static const std::unordered_set<std::u16string, u16string_hash> k = {
        u"小院儿", u"胡同儿", u"范儿", u"老汉儿", u"撒欢儿", u"寻老礼儿", u"妥妥儿", u"媳妇儿"
    };
    return k;
}

static const std::unordered_set<std::u16string, u16string_hash> & zh_not_erhua_words() {
    // 参考：参考项目 kokoro.cpp-main/ZHFrontend.cpp 的 not_erhua
    static const std::unordered_set<std::u16string, u16string_hash> k = {
        u"虐儿", u"为儿", u"护儿", u"瞒儿", u"救儿", u"替儿", u"有儿", u"一儿", u"我儿", u"俺儿", u"妻儿",
        u"拐儿", u"聋儿", u"乞儿", u"患儿", u"幼儿", u"孤儿", u"婴儿", u"婴幼儿", u"连体儿", u"脑瘫儿",
        u"流浪儿", u"体弱儿", u"混血儿", u"蜜雪儿", u"舫儿", u"祖儿", u"美儿", u"应采儿", u"可儿", u"侄儿",
        u"孙儿", u"侄孙儿", u"女儿", u"男儿", u"红孩儿", u"花儿", u"虫儿", u"马儿", u"鸟儿", u"猪儿", u"猫儿",
        u"狗儿", u"少儿"
    };
    return k;
}

static void apply_bu_sandhi(const std::u16string & word, std::vector<zh_syllable> & syllables) {
    for (size_t i = 0; i + 1 < word.size() && i + 1 < syllables.size(); ++i) {
        if (word[i] == u'不' && syllables[i + 1].tone == 4) {
            syllables[i].tone = 2;
        }
    }
}

static void apply_yi_sandhi(const std::u16string & word, std::vector<zh_syllable> & syllables) {
    bool has_yi = false;
    for (char16_t c : word) {
        if (c == u'一') {
            has_yi = true;
            break;
        }
    }
    if (!has_yi) {
        return;
    }

    bool all_numeric = true;
    for (char16_t c : word) {
        if (!is_numeric_zh_char(c)) {
            all_numeric = false;
            break;
        }
    }
    if (all_numeric) {
        return;
    }

    // A一A -> A(一=轻声)A
    if (word.size() == 3 && word[1] == u'一' && word[0] == word[2]) {
        if (syllables.size() > 1) {
            syllables[1].tone = 5;
        }
        return;
    }

    // “第一”保持一声（序数）
    if (word.size() >= 2 && word[0] == u'第' && word[1] == u'一') {
        return;
    }

    for (size_t i = 0; i + 1 < word.size() && i + 1 < syllables.size(); ++i) {
        if (word[i] != u'一') {
            continue;
        }
        const uint8_t next_tone = syllables[i + 1].tone;
        syllables[i].tone = (next_tone == 4 || next_tone == 5) ? 2 : 4;
    }
}

static void apply_three_sandhi(std::vector<zh_syllable> & syllables) {
    // 三声连读：连续 3-3-...-3 里，除最后一个之外都改为 2。
    for (size_t i = 0; i < syllables.size();) {
        if (syllables[i].tone != 3) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < syllables.size() && syllables[j].tone == 3) {
            ++j;
        }
        if (j - i >= 2) {
            for (size_t k = i; k + 1 < j; ++k) {
                syllables[k].tone = 2;
            }
        }
        i = j;
    }
}

static void apply_neutral_tone(const std::u16string & word, std::vector<zh_syllable> & syllables) {
    if (syllables.empty()) {
        return;
    }

    if (zh_must_not_neutral_words().count(word)) {
        return;
    }

    // 叠词：相邻重复字 -> 后一个轻声（不依赖 POS 的近似规则）
    for (size_t i = 1; i < word.size() && i < syllables.size(); ++i) {
        if (word[i] == word[i - 1]) {
            syllables[i].tone = 5;
        }
    }

    // 语气/结构助词
    {
        const char16_t last = word.back();
        const std::u16string particles = u"吧呢啊呐噻嘛吖嗨哦哒滴哩哟喽啰耶喔诶";
        if (particles.find(last) != std::u16string::npos) {
            syllables.back().tone = 5;
        } else if (last == u'的' || last == u'地' || last == u'得') {
            syllables.back().tone = 5;
        }
    }

    // 常见后缀：们/子、方向补语（上来/下去等）
    if (word.size() > 1) {
        const char16_t last = word.back();
        if (last == u'们' || last == u'子') {
            syllables.back().tone = 5;
        }
        if ((last == u'来' || last == u'去') && word.size() >= 2) {
            const char16_t prev = word[word.size() - 2];
            const std::u16string dirs = u"上下进出回过起开";
            if (dirs.find(prev) != std::u16string::npos) {
                syllables.back().tone = 5;
            }
        }
    }

    if (zh_must_neutral_words().count(word)) {
        syllables.back().tone = 5;
    }
}

static void apply_erhua(const std::u16string & word, std::vector<zh_syllable> & syllables) {
    // 儿化：当词以“儿”结尾且最后音节是 er(2/5) 时，把儿合并到前一个音节。
    if (word.size() < 2 || syllables.size() < 2) {
        return;
    }
    if (word.back() != u'儿') {
        return;
    }

    const bool must = zh_must_erhua_words().count(word) > 0;
    if (!must && zh_not_erhua_words().count(word) > 0) {
        return;
    }

    zh_syllable & last = syllables.back();
    if (last.final != "er") {
        return;
    }
    if (last.tone == 1) {
        // er1 -> er2（参考实现保留）
        last.tone = 2;
    }
    if (last.tone != 2 && last.tone != 5) {
        return;
    }

    syllables.pop_back();
    syllables.back().erhua = true;
}

static bool build_syllable_from_char(char16_t c, const zh_pinyin_dict & dict, zh_syllable & out) {
    // 优先内置逐字映射；缺失时使用 pinyin.txt 兜底。
    kokoro_zh::zh_syllable_base base{};
    if (kokoro_zh::lookup_zh_syllable(static_cast<uint32_t>(c), base)) {
        out.initial = std::string(base.initial);
        out.final = std::string(base.final);
        out.tone = base.tone;
        out.erhua = false;
        return true;
    }
    const auto it = dict.single.find(c);
    if (it != dict.single.end()) {
        return parse_pinyin_syllable(it->second, out);
    }
    return false;
}

static void build_word_syllables(const std::u16string & word,
                                 const std::vector<std::string> & pinyins,
                                 const zh_pinyin_dict & dict,
                                 std::vector<zh_syllable> & out) {
    out.clear();
    out.reserve(word.size());

    const bool use_pinyin = !pinyins.empty() && pinyins.size() == word.size();
    for (size_t i = 0; i < word.size(); ++i) {
        zh_syllable syl{};
        bool ok = false;
        if (use_pinyin) {
            ok = parse_pinyin_syllable(pinyins[i], syl);
        }
        if (!ok) {
            ok = build_syllable_from_char(word[i], dict, syl);
        }
        if (ok) {
            out.push_back(std::move(syl));
        }
    }

    // 规则顺序：不/一 → 轻声 → 三声连读 → 儿化（参考 ToneSandhi + ZHFrontend）。
    apply_bu_sandhi(word, out);
    apply_yi_sandhi(word, out);
    apply_neutral_tone(word, out);
    apply_three_sandhi(out);
    apply_erhua(word, out);
}

static void append_hanzi_segment_with_dict(const std::u16string & segment, const zh_pinyin_dict & dict, std::string & out) {
    // 连续汉字段：用短语词典做 DP 分段，优先匹配更长短语；对每个词段生成音节并追加到 out。
    static constexpr size_t kMaxPhraseChars = 8;
    const size_t n = segment.size();
    if (n == 0) {
        return;
    }

    const int INF = std::numeric_limits<int>::max() / 4;
    std::vector<int> dp(n + 1, INF);
    std::vector<uint8_t> next_len(n, 1);
    dp[n] = 0;

    for (int ii = static_cast<int>(n) - 1; ii >= 0; --ii) {
        const size_t i = static_cast<size_t>(ii);
        int best_cost = INF;
        uint8_t best_len = 1;

        const size_t max_k = std::min(kMaxPhraseChars, n - i);
        for (size_t k = 1; k <= max_k; ++k) {
            bool matched = false;
            if (k == 1) {
                matched = true;
            } else {
                const std::u16string key = segment.substr(i, k);
                matched = dict.phrase.find(key) != dict.phrase.end();
            }
            if (!matched) {
                continue;
            }

            const int cost = dp[i + k] + ((k == 1) ? 1 : 0);
            if (cost < best_cost) {
                best_cost = cost;
                best_len = static_cast<uint8_t>(k);
            }
        }

        if (best_cost == INF) {
            best_cost = dp[i + 1] + 1;
            best_len = 1;
        }

        dp[i] = best_cost;
        next_len[i] = best_len;
    }

    std::vector<std::string> pinyins;
    std::vector<zh_syllable> syllables;

    size_t i = 0;
    while (i < n) {
        const size_t len = static_cast<size_t>(next_len[i]);
        std::u16string word = segment.substr(i, len);

        pinyins.clear();
        if (len > 1) {
            const auto it = dict.phrase.find(word);
            if (it != dict.phrase.end()) {
                split_pinyin_sequence(it->second, pinyins);
            }
        }

        build_word_syllables(word, pinyins, dict, syllables);
        for (const auto & syl : syllables) {
            if (!syl.initial.empty()) {
                out.append(zh_map(syl.initial));
            }
            out.append(zh_map(syl.final));
            if (syl.erhua) {
                out.push_back('R');
            }
            out.push_back(static_cast<char>('0' + syl.tone));
        }

        i += len;
    }
}

static std::string text_to_zh_phonemes_with_dict(const std::string & text, const zh_pinyin_dict & dict) {
    // 说明：词典增强版：对连续汉字段做 DP + 变调/儿化/轻声；其它字符保留标点/空白。
    const std::string normalized = normalize_zh_numbers(text);

    std::string out;
    out.reserve(normalized.size() * 2);

    std::u16string hanzi_segment;
    hanzi_segment.reserve(normalized.size());

    auto flush_hanzi = [&] {
        if (hanzi_segment.empty()) {
            return;
        }
        append_hanzi_segment_with_dict(hanzi_segment, dict, out);
        hanzi_segment.clear();
    };

    size_t offset = 0;
    while (offset < normalized.size()) {
        uint32_t cp = 0;
        if (!utf8_decode_next(normalized, offset, cp)) {
            break;
        }
        cp = normalize_punctuation(cp);

        if (cp >= 0x4E00 && cp <= 0x9FFF) {
            hanzi_segment.push_back(static_cast<char16_t>(cp));
            continue;
        }

        flush_hanzi();

        if (cp <= 0x7F) {
            const char c = static_cast<char>(cp);
            if (c != '\0') {
                out.push_back(c);
            }
            continue;
        }

        if (is_boundary_cp(cp)) {
            utf8_append(out, cp);
        }
    }
    flush_hanzi();

    return out;
}

} // namespace

namespace kokoro_zh {

std::string text_to_zh_phonemes(const std::string & text, const std::string & dict_dir) {
    // 说明：如果存在短语拼音词典，则优先走“词典 + DP + 变调/儿化/轻声”路径；
    // 否则回退到旧的“逐字映射 + 极简变调”实现（完全零依赖）。
    //
    // 约定：
    // - dict_dir == "-"      ：显式禁用词典路径（便于对比/排查）。
    // - dict_dir == ":builtin"：强制使用内置词典（若编译时启用 TTS_ZH_DICT_BUILTIN）。
    // - dict_dir 为空        ：自动模式，优先尝试 ./dict，失败则（若启用内置词典）回退到内置。
    if (dict_dir != "-") {
        const std::string effective_dir = dict_dir.empty() ? "dict" : dict_dir;
        if (const zh_pinyin_dict * dict = zh_try_get_dict(effective_dir)) {
            return text_to_zh_phonemes_with_dict(text, *dict);
        }
    }

    // 说明：先做中文数字归一化，避免 ASCII 数字直接进入声调数字通道。
    const std::string normalized = normalize_zh_numbers(text);

    // Tokenize into syllables + boundary codepoints.
    std::vector<zh_token> tokens;
    tokens.reserve(normalized.size());

    size_t offset = 0;
    while (offset < normalized.size()) {
        uint32_t cp = 0;
        utf8_decode_next(normalized, offset, cp);
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
    out.reserve(normalized.size() * 2);

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
