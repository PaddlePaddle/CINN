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

#include <absl/meta/type_traits.h>
#include <absl/types/optional.h>

#include <algorithm>
#include <functional>
#include <type_traits>
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

template <typename T, typename = absl::void_t<>>
struct HasRange : std::false_type {};

template <typename T>
struct HasRange<T, absl::void_t<decltype(std::declval<T &>().begin()), decltype(std::declval<T &>().end())>>
    : std::true_type {};

template <typename T>
std::vector<T> InnerFlatten(const absl::optional<T> &e, std::false_type) {
  if (e) {
    return {*e};
  } else {
    return std::vector<T>{};
  }
}

template <typename T>
auto InnerFlatten(const absl::optional<T> &c, std::true_type) {
  using E               = typename T::value_type;
  absl::optional<E> val = absl::nullopt;
  if (c && !c->empty()) {
    val = static_cast<E>(*c->begin());
  }

  auto res = InnerFlatten(val, HasRange<E>{});

  if (val) {
    auto it = ++c->begin();
    while (it != c->end()) {
      val      = static_cast<E>(*it);
      auto tmp = InnerFlatten(val, HasRange<E>{});
      res.insert(res.end(), tmp.begin(), tmp.end());
      ++it;
    }
  }
  return res;
}

template <typename T>
auto Flatten(const T &v) {
  return InnerFlatten(absl::make_optional(v), HasRange<T>{});
}

}  // namespace utils
}  // namespace cinn
