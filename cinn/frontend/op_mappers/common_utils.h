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

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <vector>

#include "cinn/frontend/paddle/cpp/op_desc.h"

namespace cinn {
namespace frontend {
namespace utils {

template <typename T>
inline T GetAttrOrDefault(const paddle::cpp::OpDesc& op_desc, const std::string& name, const T& default_value = T{}) {
  if (op_desc.HasAttr(name)) {
    return op_desc.GetAttr<T>(name);
  }
  return default_value;
}

}  // namespace utils
}  // namespace frontend
}  // namespace cinn
