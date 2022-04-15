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
#include "cinn/utils/type_defs.h"

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

#define EXPAND_SINGLE_NUM_TO_VECTOR(DATA_TYPE, ATTR_TYPE)                                                         \
  template <>                                                                                                     \
  inline std::vector<DATA_TYPE> GetAttrOrDefault(                                                                 \
      const paddle::cpp::OpDesc& op_desc, const std::string& name, const std::vector<DATA_TYPE>& default_value) { \
    if (op_desc.HasAttr(name)) {                                                                                  \
      auto attr_type = op_desc.GetAttrType(name);                                                                 \
      if (attr_type == paddle::cpp::OpDescAPI::AttrType::ATTR_TYPE##S) {                                          \
        return op_desc.GetAttr<std::vector<DATA_TYPE>>(name);                                                     \
      } else if (attr_type == paddle::cpp::OpDescAPI::AttrType::ATTR_TYPE) {                                      \
        return std::vector<DATA_TYPE>{op_desc.GetAttr<DATA_TYPE>(name)};                                          \
      } else {                                                                                                    \
        LOG(FATAL) << "Op " << op_desc.Type() << "'s attribute " << name << " should be " << #ATTR_TYPE           \
                   << "S. Please Check!";                                                                         \
      }                                                                                                           \
    }                                                                                                             \
    return default_value;                                                                                         \
  }

EXPAND_SINGLE_NUM_TO_VECTOR(int, INT)
EXPAND_SINGLE_NUM_TO_VECTOR(float, FLOAT)
EXPAND_SINGLE_NUM_TO_VECTOR(std::string, STRING)
EXPAND_SINGLE_NUM_TO_VECTOR(bool, BOOLEAN)
EXPAND_SINGLE_NUM_TO_VECTOR(int64_t, LONG)

#undef EXPAND_SINGLE_NUM_TO_VECTOR

template <>
inline bool GetAttrOrDefault(const paddle::cpp::OpDesc& op_desc, const std::string& name, const bool& default_value) {
  if (op_desc.HasAttr(name)) {
    auto attr_type = op_desc.GetAttrType(name);
    if (attr_type == paddle::cpp::OpDescAPI::AttrType::BOOLEAN) {
      return op_desc.GetAttr<bool>(name);
    } else if (attr_type == paddle::cpp::OpDescAPI::AttrType::INT) {
      return static_cast<bool>(op_desc.GetAttr<int>(name));
    } else if (attr_type == paddle::cpp::OpDescAPI::AttrType::LONG) {
      return static_cast<bool>(op_desc.GetAttr<int64_t>(name));
    } else {
      LOG(FATAL) << "Op " << op_desc.Type() << "'s attribute " << name << " should be BOOLEAN. Please Check!";
    }
  }
  return default_value;
}

template <typename T>
inline utils::ShapeType ToShapeType(const std::vector<T>& shape) {
  return utils::ShapeType(shape.begin(), shape.end());
}

template <typename T>
inline utils::DimType ToDimType(const T& val) {
  return static_cast<utils::DimType>(val);
}

}  // namespace utils
}  // namespace frontend
}  // namespace cinn
