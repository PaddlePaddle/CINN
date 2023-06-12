// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <unordered_set>

#include "cinn/api/tensor_interface.h"
#include "cinn/utils/small_vector.h"

namespace cinn {
namespace api {

class TensorInterfaceList : public cinn::utils::SmallVector<TensorInterfacePtr, 16> {
 public:
  using cinn::utils::SmallVector<TensorInterfacePtr, 16>::SmallVector;

  TensorInterfaceList& operator+=(const TensorInterfaceList& other) {
    std::unordered_set<TensorInterfacePtr> tensor_set(this->begin(), this->end());
    for (const auto& tensor_if : other) {
      if (tensor_set.find(tensor_if) == tensor_set.end()) {
        this->push_back(tensor_if);
        tensor_set.insert(tensor_if);
      }
    }
    return *this;
  }
};

}  // namespace api
}  // namespace cinn
