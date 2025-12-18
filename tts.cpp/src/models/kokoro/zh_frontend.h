#pragma once

#include <string>

// Minimal Mandarin (zh) G2P for Kokoro based on the phoneme mapping used by misaki[zh] v1.1.
// Produces a UTF-8 string where each UTF-8 *character* is a Kokoro token (Bopomofo + a small set of CJK chars + tone digits).

namespace kokoro_zh {

std::string text_to_zh_phonemes(const std::string & text);

} // namespace kokoro_zh

