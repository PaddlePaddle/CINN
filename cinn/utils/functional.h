#pragma once

#include <algorithm>
#include <functional>
#include <vector>

namespace cinn {
namespace utils {

template <typename InT, typename OutValT>
std::vector<OutValT> Map(const InT& in, std::function<OutValT(const typename InT::value_type&)> fn) {
  std::vector<OutValT> res;
  std::transform(
      in.begin(), in.end(), std::back_inserter(res), [&](const typename InT::value_type& x) { return fn(x); });
  return res;
}

template <typename T>
inline T Max(T a, T b, T c) {
  return std::max(a, std::max(b, c));
}
template <typename T>
inline T Max(T a, T b, T c, T d) {
  return std::max(a, Max(b, c, d));
}
template <typename T>
inline T Min(T a, T b, T c) {
  return std::min(a, std::min(b, c));
}
template <typename T>
inline T Min(T a, T b, T c, T d) {
  return std::min(a, Min(b, c, d));
}

}  // namespace utils
}  // namespace cinn
