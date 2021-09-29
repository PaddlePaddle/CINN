#pragma once

#include <algorithm>
#include <functional>
#include <vector>

namespace cinn {
namespace utils {

template <typename InT, typename OutValT>
std::vector<OutValT> Map(const InT &in, std::function<OutValT(const typename InT::value_type &)> fn) {
  std::vector<OutValT> res;
  std::transform(
      in.begin(), in.end(), std::back_inserter(res), [&](const typename InT::value_type &x) { return fn(x); });
  return res;
}

template <typename T>
auto Min(T && t) {
  return t;
}

template <typename T, typename... Ts>
auto Min(T &&t, Ts &&... ts) {
  return std::min(t, Min(ts...));
}

template <typename T>
auto Max(T && t) {
  return t;
}

template <typename T, typename... Ts>
auto Max(T &&t, Ts &&... ts) {
  return std::max(t, Max(ts...));
}
}  // namespace utils
}  // namespace cinn
