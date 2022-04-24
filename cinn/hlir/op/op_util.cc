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

#include "cinn/hlir/op/op_util.h"

namespace cinn {
namespace hlir {

cinn::utils::ShapeType ToShapeType(const std::vector<Expr>& args) {
  cinn::utils::ShapeType input_shape;
  std::for_each(args.begin(), args.end(), [&](const Expr& expr) { input_shape.emplace_back(expr.as_int32()); });
  return input_shape;
}

}  // namespace hlir
}  // namespace cinn
