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
#include <unordered_map>
#include <unordered_set>

#include "cinn/api/tensor_interface.h"
#include "cinn/api/tensor_interface_list.h"
#include "cinn/hlir/framework/op.h"

namespace cinn {
namespace api {

class OpGroupInterface {
 public:
  virtual hlir::framework::OpPatternKind kind() const = 0;

  //  virtual const TensorInterfaceList& input_tensors() const = 0;

  //  virtual const TensorInterfaceList& output_tensors() const = 0;

  //  virtual const std::unordered_set<std::shared_ptr<OpGroupInterface>> producers() const = 0;

  //  virtual const std::unordered_set<std::shared_ptr<OpGroupInterface>> consumers() const = 0;

  virtual const std::unordered_map<std::shared_ptr<OpGroupInterface>, TensorInterfaceList>& producer_groups() const = 0;

  virtual const std::unordered_map<std::shared_ptr<OpGroupInterface>, TensorInterfaceList>& consumer_groups() const = 0;

  const std::unordered_map<std::shared_ptr<OpGroupInterface>, TensorInterfaceList>& producer2inputs() const {
    return producer_groups();
  }

  const std::unordered_map<std::shared_ptr<OpGroupInterface>, TensorInterfaceList>& consumer2outputs() const {
    return consumer_groups();
  }

 protected:
  OpGroupInterface() = default;
};

}  // namespace api
}  // namespace cinn
