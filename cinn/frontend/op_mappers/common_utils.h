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
