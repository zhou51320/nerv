#ifndef TTS_NUMBERS_COMPAT_H
#define TTS_NUMBERS_COMPAT_H

#if defined(__has_include)
#   if __has_include(<numbers>)
#       include <numbers>
#       define TTS_HAS_STD_NUMBERS 1
#   endif
#endif

#ifndef TTS_HAS_STD_NUMBERS
namespace std {
namespace numbers {

template<typename T>
inline constexpr T pi_v = static_cast<T>(3.141592653589793238462643383279502884L);

inline constexpr double pi = pi_v<double>;

} // namespace numbers
} // namespace std
#endif

#undef TTS_HAS_STD_NUMBERS

#endif // TTS_NUMBERS_COMPAT_H
