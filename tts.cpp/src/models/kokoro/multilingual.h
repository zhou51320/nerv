#pragma once

#include <string>

struct phonemizer;

// Multilingual phonemization for Kokoro:
// - ASCII English segments -> existing Kokoro phonemizer (IPA)
// - Non-ASCII segments     -> built-in zh frontend (Bopomofo + tone digits)
std::string kokoro_phonemize_multilingual(const std::string & text, phonemizer * fallback_en_phonemizer);

bool kokoro_contains_cjk(const std::string & text);

