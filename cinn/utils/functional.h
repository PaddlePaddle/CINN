// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
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
auto Min(T &&t) {
  return t;
}

template <typename T, typename... Ts>
auto Min(T &&t, Ts &&... ts) {
  return std::min(t, Min(ts...));
}

template <typename T>
auto Max(T &&t) {
  return t;
}

template <typename T, typename... Ts>
auto Max(T &&t, Ts &&... ts) {
  return std::max(t, Max(ts...));
}

template <typename T>
struct is_vector : std::false_type {};

template <typename... Ts>
struct is_vector<std::vector<Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_f(const T &) {
  return is_vector<T>::value;
}

}  // namespace utils
}  // namespace cinn
