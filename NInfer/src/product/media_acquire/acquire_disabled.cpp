#include "product/media_acquire/acquire.h"

namespace ninfer::product::media_acquire {

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::invalid_argument("media acquisition is not supported in this NINFER_TEXT_ONLY build");
}

} // namespace ninfer::product::media_acquire
