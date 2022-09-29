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
#include <ostream>
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
inline const bool is_vector_f(const T &) {
  return is_vector<T>::value;
}

template <typename T, typename = absl::void_t<>>
struct HasRange : std::false_type {};

template <typename T>
struct HasRange<T, absl::void_t<decltype(*std::declval<T &>().begin()), decltype(*std::declval<T &>().end())>>
    : std::true_type {};

template <typename T>
std::vector<T> InnerFlatten(const absl::optional<std::reference_wrapper<const T>> &e, std::false_type) {
  if (e) {
    return {e->get()};
  } else {
    return std::vector<T>{};
  }
}

template <typename T, typename E = std::decay_t<decltype(*std::declval<const T>().begin())>>
auto InnerFlatten(const absl::optional<std::reference_wrapper<const T>> &c, std::true_type) {
  absl::optional<std::reference_wrapper<const E>> val;
  if (c && !c->get().empty()) {
    val = *(c->get().begin());
  }

  auto res = InnerFlatten(val, HasRange<E>{});

  if (val) {
    auto it = ++(c->get().begin());
    while (it != c->get().end()) {
      val      = *it;
      auto tmp = InnerFlatten(val, HasRange<E>{});
      res.insert(res.end(), tmp.begin(), tmp.end());
      ++it;
    }
  }
  return res;
}

std::vector<bool> InnerFlatten(const absl::optional<std::reference_wrapper<const std::vector<bool>>> &c,
                               std::true_type);

std::vector<std::string> InnerFlatten(const absl::optional<std::reference_wrapper<const std::string>> &c,
                                      std::true_type);

template <typename T>
auto Flatten(const T &v) {
  absl::optional<std::reference_wrapper<const T>> w = v;
  return InnerFlatten(w, HasRange<T>{});
}

}  // namespace utils
}  // namespace cinn
