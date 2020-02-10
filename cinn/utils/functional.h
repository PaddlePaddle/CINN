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

}  // namespace utils
}  // namespace cinn
