#include "ja_frontend.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace {

// 说明：Kokoro 的 tokenizer 是“按 token 字符串匹配”的方式工作。
// 对日文来说，我们采用“一字符=一音素”的思路：
// - 元音使用 ASCII: a/i/u/e/o
// - 辅音使用 ASCII 或扩展 IPA/修饰字母（例如 ɕ/ʨ/ɴ/ː 等）
// - 一个假名（mora）通常映射为 1~2 个音素字符（辅音+元音），少数为 1（如 ン/ッ/ー）
//
// 备注：下面的假名到音素映射表等价于常见的日语假名音素拆分（含拗音/外来音），
// 用于把“假名串”转换为模型训练时使用的音素字符集。

struct ja_phoneme_seq {
    char32_t p0 = 0;
    char32_t p1 = 0;
};

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
    // 说明：将常见全角/日文标点归一化为 ASCII，便于 Kokoro 统一处理句界。
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

static bool is_hiragana(uint32_t cp) {
    // ひらがな范围（含小假名、ゔ 等）；迭代符号在 0x309D..0x309F。
    return (cp >= 0x3041 && cp <= 0x3096) || (cp >= 0x309D && cp <= 0x309F);
}

static bool is_kanji(uint32_t cp) {
    // 说明：这里只做最常见 CJK Unified Ideographs 范围检测（不覆盖扩展区）。
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

static bool is_ascii_boundary(uint32_t cp) {
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
            return true;
        default:
            return false;
    }
}

static const std::unordered_map<uint32_t, ja_phoneme_seq> & ja_m2p_table() {
    // key 说明：
    // - 单字符 key：0x0000XXXX（Katakana 码点）
    // - 双字符 key：0xAAAAFFFF（高 16 位是第一个 char16_t，低 16 位是第二个 char16_t）
    static const std::unordered_map<uint32_t, ja_phoneme_seq> k = {
        // 说明：映射表由假名->音素规则生成；这里固定顺序只是为了便于 diff，不影响运行时行为。
        { 0x30A1, { U'a', 0 } },
        { 0x30A2, { U'a', 0 } },
        { 0x30A3, { U'i', 0 } },
        { 0x30A4, { U'i', 0 } },
        { 0x30A5, { U'u', 0 } },
        { 0x30A6, { U'u', 0 } },
        { 0x30A7, { U'e', 0 } },
        { 0x30A8, { U'e', 0 } },
        { 0x30A9, { U'o', 0 } },
        { 0x30AA, { U'o', 0 } },
        { 0x30AB, { U'k', U'a' } },
        { 0x30AC, { U'g', U'a' } },
        { 0x30AD, { U'k', U'i' } },
        { 0x30AE, { U'g', U'i' } },
        { 0x30AF, { U'k', U'u' } },
        { 0x30B0, { U'g', U'u' } },
        { 0x30B1, { U'k', U'e' } },
        { 0x30B2, { U'g', U'e' } },
        { 0x30B3, { U'k', U'o' } },
        { 0x30B4, { U'g', U'o' } },
        { 0x30B5, { U's', U'a' } },
        { 0x30B6, { U'z', U'a' } },
        { 0x30B7, { U'\u0255', U'i' } },
        { 0x30B8, { U'\u02A5', U'i' } },
        { 0x30B9, { U's', U'u' } },
        { 0x30BA, { U'z', U'u' } },
        { 0x30BB, { U's', U'e' } },
        { 0x30BC, { U'z', U'e' } },
        { 0x30BD, { U's', U'o' } },
        { 0x30BE, { U'z', U'o' } },
        { 0x30BF, { U't', U'a' } },
        { 0x30C0, { U'd', U'a' } },
        { 0x30C1, { U'\u02A8', U'i' } },
        { 0x30C2, { U'\u02A5', U'i' } },
        { 0x30C3, { U'\u0294', 0 } },
        { 0x30C4, { U'\u02A6', U'u' } },
        { 0x30C5, { U'z', U'u' } },
        { 0x30C6, { U't', U'e' } },
        { 0x30C7, { U'd', U'e' } },
        { 0x30C8, { U't', U'o' } },
        { 0x30C9, { U'd', U'o' } },
        { 0x30CA, { U'n', U'a' } },
        { 0x30CB, { U'n', U'i' } },
        { 0x30CC, { U'n', U'u' } },
        { 0x30CD, { U'n', U'e' } },
        { 0x30CE, { U'n', U'o' } },
        { 0x30CF, { U'h', U'a' } },
        { 0x30D0, { U'b', U'a' } },
        { 0x30D1, { U'p', U'a' } },
        { 0x30D2, { U'h', U'i' } },
        { 0x30D3, { U'b', U'i' } },
        { 0x30D4, { U'p', U'i' } },
        { 0x30D5, { U'f', U'u' } },
        { 0x30D6, { U'b', U'u' } },
        { 0x30D7, { U'p', U'u' } },
        { 0x30D8, { U'h', U'e' } },
        { 0x30D9, { U'b', U'e' } },
        { 0x30DA, { U'p', U'e' } },
        { 0x30DB, { U'h', U'o' } },
        { 0x30DC, { U'b', U'o' } },
        { 0x30DD, { U'p', U'o' } },
        { 0x30DE, { U'm', U'a' } },
        { 0x30DF, { U'm', U'i' } },
        { 0x30E0, { U'm', U'u' } },
        { 0x30E1, { U'm', U'e' } },
        { 0x30E2, { U'm', U'o' } },
        { 0x30E3, { U'j', U'a' } },
        { 0x30E4, { U'j', U'a' } },
        { 0x30E5, { U'j', U'u' } },
        { 0x30E6, { U'j', U'u' } },
        { 0x30E7, { U'j', U'o' } },
        { 0x30E8, { U'j', U'o' } },
        { 0x30E9, { U'r', U'a' } },
        { 0x30EA, { U'r', U'i' } },
        { 0x30EB, { U'r', U'u' } },
        { 0x30EC, { U'r', U'e' } },
        { 0x30ED, { U'r', U'o' } },
        { 0x30EE, { U'w', U'a' } },
        { 0x30EF, { U'w', U'a' } },
        { 0x30F0, { U'i', 0 } },
        { 0x30F1, { U'e', 0 } },
        { 0x30F2, { U'o', 0 } },
        { 0x30F3, { U'\u0274', 0 } },
        { 0x30F4, { U'v', U'u' } },
        { 0x30F5, { U'k', U'a' } },
        { 0x30F6, { U'k', U'e' } },
        { 0x30F7, { U'v', U'a' } },
        { 0x30F8, { U'v', U'i' } },
        { 0x30F9, { U'v', U'e' } },
        { 0x30FA, { U'v', U'o' } },
        { 0x30FC, { U'\u02D0', 0 } },
        { 0x30A430A7, { U'j', U'e' } },
        { 0x30A630A3, { U'w', U'i' } },
        { 0x30A630A5, { U'w', U'u' } },
        { 0x30A630A7, { U'w', U'e' } },
        { 0x30A630A9, { U'w', U'o' } },
        { 0x30AD30A3, { U'\u1D84', U'i' } },
        { 0x30AD30A7, { U'\u1D84', U'e' } },
        { 0x30AD30E3, { U'\u1D84', U'a' } },
        { 0x30AD30E5, { U'\u1D84', U'u' } },
        { 0x30AD30E7, { U'\u1D84', U'o' } },
        { 0x30AE30A3, { U'\u1D83', U'i' } },
        { 0x30AE30A7, { U'\u1D83', U'e' } },
        { 0x30AE30E3, { U'\u1D83', U'a' } },
        { 0x30AE30E5, { U'\u1D83', U'u' } },
        { 0x30AE30E7, { U'\u1D83', U'o' } },
        { 0x30AF30A1, { U'K', U'a' } },
        { 0x30AF30A3, { U'K', U'i' } },
        { 0x30AF30A5, { U'K', U'u' } },
        { 0x30AF30A7, { U'K', U'e' } },
        { 0x30AF30A9, { U'K', U'o' } },
        { 0x30AF30EE, { U'K', U'a' } },
        { 0x30B030A1, { U'G', U'a' } },
        { 0x30B030A3, { U'G', U'i' } },
        { 0x30B030A5, { U'G', U'u' } },
        { 0x30B030A7, { U'G', U'e' } },
        { 0x30B030A9, { U'G', U'o' } },
        { 0x30B030EE, { U'G', U'a' } },
        { 0x30B730A7, { U'\u0255', U'e' } },
        { 0x30B730E3, { U'\u0255', U'a' } },
        { 0x30B730E5, { U'\u0255', U'u' } },
        { 0x30B730E7, { U'\u0255', U'o' } },
        { 0x30B830A7, { U'\u02A5', U'e' } },
        { 0x30B830E3, { U'\u02A5', U'a' } },
        { 0x30B830E5, { U'\u02A5', U'u' } },
        { 0x30B830E7, { U'\u02A5', U'o' } },
        { 0x30B930A3, { U's', U'i' } },
        { 0x30BA30A3, { U'z', U'i' } },
        { 0x30C130A7, { U'\u02A8', U'e' } },
        { 0x30C130E3, { U'\u02A8', U'a' } },
        { 0x30C130E5, { U'\u02A8', U'u' } },
        { 0x30C130E7, { U'\u02A8', U'o' } },
        { 0x30C230A7, { U'\u02A5', U'e' } },
        { 0x30C230E3, { U'\u02A5', U'a' } },
        { 0x30C230E5, { U'\u02A5', U'u' } },
        { 0x30C230E7, { U'\u02A5', U'o' } },
        { 0x30C430A1, { U'\u02A6', U'a' } },
        { 0x30C430A3, { U'\u02A6', U'i' } },
        { 0x30C430A7, { U'\u02A6', U'e' } },
        { 0x30C430A9, { U'\u02A6', U'o' } },
        { 0x30C630A3, { U't', U'i' } },
        { 0x30C630A7, { U'\u01AB', U'e' } },
        { 0x30C630E3, { U'\u01AB', U'a' } },
        { 0x30C630E5, { U'\u01AB', U'u' } },
        { 0x30C630E7, { U'\u01AB', U'o' } },
        { 0x30C730A3, { U'd', U'i' } },
        { 0x30C730A7, { U'\u1D81', U'e' } },
        { 0x30C730E3, { U'\u1D81', U'a' } },
        { 0x30C730E5, { U'\u1D81', U'u' } },
        { 0x30C730E7, { U'\u1D81', U'o' } },
        { 0x30C830A5, { U't', U'u' } },
        { 0x30C930A5, { U'd', U'u' } },
        { 0x30CB30A3, { U'\u0272', U'i' } },
        { 0x30CB30A7, { U'\u0272', U'e' } },
        { 0x30CB30E3, { U'\u0272', U'a' } },
        { 0x30CB30E5, { U'\u0272', U'u' } },
        { 0x30CB30E7, { U'\u0272', U'o' } },
        { 0x30D230A3, { U'\u00E7', U'i' } },
        { 0x30D230A7, { U'\u00E7', U'e' } },
        { 0x30D230E3, { U'\u00E7', U'a' } },
        { 0x30D230E5, { U'\u00E7', U'u' } },
        { 0x30D230E7, { U'\u00E7', U'o' } },
        { 0x30D330A3, { U'\u1D80', U'i' } },
        { 0x30D330A7, { U'\u1D80', U'e' } },
        { 0x30D330E3, { U'\u1D80', U'a' } },
        { 0x30D330E5, { U'\u1D80', U'u' } },
        { 0x30D330E7, { U'\u1D80', U'o' } },
        { 0x30D430A3, { U'\u1D88', U'i' } },
        { 0x30D430A7, { U'\u1D88', U'e' } },
        { 0x30D430E3, { U'\u1D88', U'a' } },
        { 0x30D430E5, { U'\u1D88', U'u' } },
        { 0x30D430E7, { U'\u1D88', U'o' } },
        { 0x30D530A1, { U'f', U'a' } },
        { 0x30D530A3, { U'f', U'i' } },
        { 0x30D530A7, { U'f', U'e' } },
        { 0x30D530A9, { U'f', U'o' } },
        { 0x30DF30A3, { U'\u1D86', U'i' } },
        { 0x30DF30A7, { U'\u1D86', U'e' } },
        { 0x30DF30E3, { U'\u1D86', U'a' } },
        { 0x30DF30E5, { U'\u1D86', U'u' } },
        { 0x30DF30E7, { U'\u1D86', U'o' } },
        { 0x30EA30A3, { U'\u1D89', U'i' } },
        { 0x30EA30A7, { U'\u1D89', U'e' } },
        { 0x30EA30E3, { U'\u1D89', U'a' } },
        { 0x30EA30E5, { U'\u1D89', U'u' } },
        { 0x30EA30E7, { U'\u1D89', U'o' } },
        { 0x30F430A1, { U'v', U'a' } },
        { 0x30F430A3, { U'v', U'i' } },
        { 0x30F430A7, { U'v', U'e' } },
        { 0x30F430A9, { U'v', U'o' } },
        { 0x30F430E3, { U'\u1D80', U'a' } },
        { 0x30F430E5, { U'\u1D80', U'u' } },
        { 0x30F430E7, { U'\u1D80', U'o' } },
    };
    return k;
}

static void append_seq(std::string & out, const ja_phoneme_seq & seq) {
    if (seq.p0 != 0) {
        utf8_append(out, static_cast<uint32_t>(seq.p0));
    }
    if (seq.p1 != 0) {
        utf8_append(out, static_cast<uint32_t>(seq.p1));
    }
}

static void append_unknown(std::string & out) {
    // 说明：用一个可见占位符提示“这里有无法自动转换的内容”（多见于汉字）。
    utf8_append(out, 0x2753); // ❓
}

static void append_katakana_range_as_phonemes(const std::u16string & s,
                                              size_t begin,
                                              size_t end,
                                              std::string & out) {
    const auto & table = ja_m2p_table();
    size_t i = begin;
    while (i < end) {
        const char16_t c = s[i];
        // ASCII 边界/标点：直接输出（保持句界信息）。
        if (c <= 0x7F && is_ascii_boundary(static_cast<uint32_t>(c))) {
            out.push_back(static_cast<char>(c));
            ++i;
            continue;
        }

        // 优先尝试双字符（拗音/外来音）。
        if (i + 1 < end) {
            const uint32_t key2 = (static_cast<uint32_t>(c) << 16) | static_cast<uint32_t>(s[i + 1]);
            const auto     it2 = table.find(key2);
            if (it2 != table.end()) {
                append_seq(out, it2->second);
                i += 2;
                continue;
            }
        }

        // 单字符匹配。
        const uint32_t key1 = static_cast<uint32_t>(c);
        const auto     it1 = table.find(key1);
        if (it1 != table.end()) {
            append_seq(out, it1->second);
            i += 1;
            continue;
        }

        // 其它字符：尽量保留 ASCII，否则输出占位符。
        if (c <= 0x7F) {
            out.push_back(static_cast<char>(c));
        } else {
            append_unknown(out);
        }
        i += 1;
    }
}

} // namespace

namespace kokoro_ja {

std::string text_to_ja_phonemes(const std::string & text) {
    // 说明：
    // 1) UTF-8 解码 + 标点归一化 + 平假名→片假名（便于统一查表）
    // 2) 对“汉字(かな)”形式做一个轻量的读音注入：遇到汉字串后紧跟括号且括号内有假名，则用括号内假名替代汉字串。
    //    - 示例：東京(とうきょう) -> トウキョウ -> t oː k j oː（具体音素按表映射）
    // 3) 对剩余假名串按表映射为音素字符序列；标点/空白原样保留。
    std::u16string normalized;
    normalized.reserve(text.size());

    size_t offset = 0;
    while (offset < text.size()) {
        uint32_t cp = 0;
        if (!utf8_decode_next(text, offset, cp)) {
            break;
        }
        cp = normalize_punctuation(cp);
        if (is_hiragana(cp)) {
            cp += 0x60; // Hiragana -> Katakana
        }
        if (cp <= 0xFFFF) {
            normalized.push_back(static_cast<char16_t>(cp));
        }
    }

    std::string out;
    out.reserve(text.size() * 2);

    const size_t n = normalized.size();
    size_t i = 0;
    while (i < n) {
        const uint32_t cp = static_cast<uint32_t>(normalized[i]);

        // 1) ASCII 边界/标点：原样输出。
        if (cp <= 0x7F && is_ascii_boundary(cp)) {
            out.push_back(static_cast<char>(cp));
            ++i;
            continue;
        }

        // 2) 汉字串 + (假名) 的轻量替换。
        if (is_kanji(cp)) {
            size_t j = i;
            while (j < n && is_kanji(static_cast<uint32_t>(normalized[j]))) {
                ++j;
            }
            if (j < n && normalized[j] == u'(') {
                size_t k = j + 1;
                while (k < n && normalized[k] != u')') {
                    ++k;
                }
                if (k < n) {
                    // 括号内包含至少一个“可映射假名”时才使用（避免误把纯 ASCII 备注当读音）。
                    bool has_mappable_kana = false;
                    const auto & table = ja_m2p_table();
                    for (size_t t = j + 1; t < k; ++t) {
                        const uint32_t kc = static_cast<uint32_t>(normalized[t]);
                        if (table.find(kc) != table.end()) {
                            has_mappable_kana = true;
                            break;
                        }
                    }
                    if (has_mappable_kana) {
                        append_katakana_range_as_phonemes(normalized, j + 1, k, out);
                        i = k + 1;
                        continue;
                    }
                }
            }

            // 没有读音标注：输出占位符并跳过该汉字串。
            append_unknown(out);
            i = j;
            continue;
        }

        // 3) 其它内容（主要是假名/符号）：连续处理一段，便于双字符拗音/外来音匹配。
        const size_t start = i;
        size_t       end = i;
        while (end < n) {
            const uint32_t cp2 = static_cast<uint32_t>(normalized[end]);
            if (cp2 <= 0x7F && is_ascii_boundary(cp2)) {
                break;
            }
            if (is_kanji(cp2)) {
                break;
            }
            ++end;
        }
        append_katakana_range_as_phonemes(normalized, start, end, out);
        i = end;
    }

    return out;
}

} // namespace kokoro_ja
