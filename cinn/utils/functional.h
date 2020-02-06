#pragma once

#include <functional>

namespace cinn {
namespace utils {

template <typename InT, typename OutT>
OutT Map(const InT& in, std::function<typename OutT::value_type(const typename InT::value_type&)> fn) {
  OutT res;
  std::transform(
      in.begin(), in.end(), std::back_inserter(res), [&](const typename InT::value_type& x) { return fn(x); });
  return res;
}

}  // namespace utils
}  // namespace cinn
