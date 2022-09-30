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

template <class T>
struct is_scalar : std::integral_constant<bool,
                                          std::is_fundamental<T>::value ||
                                              std::is_same<std::string, typename std::remove_cv<T>::type>::value> {};

template <typename T>
std::enable_if_t<is_scalar<T>::value, std::vector<T>> Flatten(
    const absl::optional<std::reference_wrapper<const T>> &c) {
  return c ? std::vector<T>{c->get()} : std::vector<T>{};
}

template <template <typename...> class C, typename E>
std::enable_if_t<is_scalar<E>::value, std::vector<E>> Flatten(
    const absl::optional<std::reference_wrapper<const C<E>>> &c) {
  return c ? std::vector<E>(c->get().begin(), c->get().end()) : std::vector<E>{};
}

template <typename T, typename E = std::decay_t<decltype(*std::declval<const T>().begin())>>
auto Flatten(const absl::optional<std::reference_wrapper<const T>> &c) {
  absl::optional<std::reference_wrapper<const E>> val;
  if (c && !c->get().empty()) {
    val = *(c->get().begin());
  }

  auto res = Flatten(val);

  if (val) {
    auto it = ++(c->get().begin());
    while (it != c->get().end()) {
      val      = *it;
      auto tmp = Flatten(val);
      res.insert(res.end(), tmp.begin(), tmp.end());
      ++it;
    }
  }
  return res;
}

template <typename T>
auto Flatten(const T &v) {
  absl::optional<std::reference_wrapper<const T>> w = v;
  return Flatten(w);
}

}  // namespace utils
}  // namespace cinn
