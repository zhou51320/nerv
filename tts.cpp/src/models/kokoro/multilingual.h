#pragma once

#include <cstdint>
#include <string>

struct phonemizer;
enum class tts_language : uint8_t;

// Kokoro 多语言音素化：
// - ASCII 字母片段 -> 英文 phonemizer（IPA）
// - 其他片段（含数字）-> 中文/日文前端（按 language 或文本脚本自动选择）
// 说明：
// - language 用于控制数字/非 ASCII 片段的处理偏好：
//   - EN：数字归入英文片段；非 ASCII 片段按脚本自动选择（含假名->日文，否则->中文）。
//   - ZH：数字交给中文前端做中文读法；非 ASCII 片段走中文前端。
//   - JA：数字交给日文前端（尽量按假名读法）；非 ASCII 片段走日文前端。
std::string kokoro_phonemize_multilingual(const std::string & text,
                                          phonemizer * fallback_en_phonemizer,
                                          tts_language language,
                                          const std::string & zh_dict_dir = "");

bool kokoro_contains_cjk(const std::string & text);
