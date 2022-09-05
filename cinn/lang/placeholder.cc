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

#include "cinn/lang/placeholder.h"

#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace lang {

ir::Tensor CreatePlaceHolder(const std::vector<int> &shape, Type type, const std::string &name) {
  std::vector<Expr> expr_shape;
  for (int s : shape) {
    expr_shape.push_back(Expr(s));
  }
  return CreatePlaceHolder(expr_shape, type, name);
}

ir::Tensor CreatePlaceHolder(const std::vector<Expr> &shape, Type type, const std::string &name) {
  if (type == Float(32)) {
    return Placeholder<float>(name, shape);
  } else if (type == Float(64)) {
    return Placeholder<double>(name, shape);
  } else if (type == Int(32)) {
    return Placeholder<int32_t>(name, shape);
  } else if (type == Int(64)) {
    return Placeholder<int64_t>(name, shape);
  } else if (type == Bool()) {
    return Placeholder<bool>(name, shape);
  }
  CINN_NOT_IMPLEMENTED
}

}  // namespace lang
}  // namespace cinn
