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

#include "cinn/utils/functional.h"

#include <absl/types/optional.h>

#include <type_traits>
#include <vector>

namespace cinn {
namespace utils {

std::vector<bool> InnerFlatten(const absl::optional<std::reference_wrapper<const std::vector<bool>>> &c,
                               std::true_type) {
  absl::optional<std::reference_wrapper<const bool>> val;
  bool bool_val{false};
  if (c && !c->get().empty()) {
    // The return type of *(const std::vector<bool>::begin()) is std::_Bit_const_iterator::const_reference,
    // which can be cast to bool, but cannot be cast to absl::optional<std::reference_wrapper<const bool>>.
    bool_val = static_cast<bool>(*(c->get().begin()));
    val      = bool_val;
  }

  auto res = InnerFlatten(val, HasRange<bool>{});

  if (val) {
    auto it = ++(c->get().begin());
    while (it != c->get().end()) {
      bool_val = static_cast<bool>(*it);
      val      = bool_val;
      auto tmp = InnerFlatten(val, HasRange<bool>{});
      res.insert(res.end(), tmp.begin(), tmp.end());
      ++it;
    }
  }
  return res;
}

std::vector<std::string> InnerFlatten(const absl::optional<std::reference_wrapper<const std::string>> &c,
                                      std::true_type) {
  return {c->get()};
}

}  // namespace utils
}  // namespace cinn
