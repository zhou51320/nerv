#pragma once

#include <type_traits>

namespace ninfer::ops {

// C++17 generic lambdas receive these tags instead of a C++20 explicit template parameter list.
template <auto Value>
using DispatchValue = std::integral_constant<decltype(Value), Value>;

} // namespace ninfer::ops
