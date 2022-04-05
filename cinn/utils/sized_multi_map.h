// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <functional>
#include <map>
#include <memory>
#include <utility>

namespace cinn {
namespace utils {

/**
 * A data structure stores limited size ordered duplicatable keys and mapped
 * values.
 *
 * The default implementation would pop maximal element when size reaches
 * capacity. Users could change pop_max_when_full parameter of constructor
 * to false to pop minimal element.
 *
 * The underneath implementation uses std::multimap
 */
template <class Key, class T, class Compare = std::less<Key>, class Alloc = std::allocator<std::pair<const Key, T>>>
class SizedMultiMap {
 public:
  SizedMultiMap(size_t capacity, bool pop_max_when_full = true)
      : capacity_(capacity), pop_max_when_full_(pop_max_when_full) {}

  void Push(const Key& key, const T& data) {
    multi_map_.insert({key, data});
    if (multi_map_.size() > capacity_) {
      Pop();
    }
  }

  void Pop() {
    CHECK_GE(multi_map_.size(), 1UL) << "Call Pop on empty SizedMultiMap";
    if (pop_max_when_full_) {
      multi_map_.erase(--multi_map_.end());
    } else {
      multi_map_.erase(multi_map_.begin());
    }
  }

  std::pair<Key, T> MaxKeyValuePair() const { return *(multi_map_.rbegin()); }

  std::pair<Key, T> MinKeyValuePair() const { return *(multi_map_.begin()); }

  size_t Size() const { return multi_map_.size(); }

  template <class ContainerType>
  ContainerType ReturnAsContainer() const {
    return ContainerType(multi_map_.begin(), multi_map_.end());
  }

 private:
  size_t capacity_;
  bool pop_max_when_full_;
  std::multimap<Key, T, Compare, Alloc> multi_map_;
};

}  // namespace utils
}  // namespace cinn
