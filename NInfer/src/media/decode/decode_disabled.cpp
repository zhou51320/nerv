#include "media/decode/decode.h"

namespace ninfer::media::decode {

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    throw std::invalid_argument("image decoding is not supported in this NINFER_TEXT_ONLY build");
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    throw std::invalid_argument("video decoding is not supported in this NINFER_TEXT_ONLY build");
}

} // namespace ninfer::media::decode
