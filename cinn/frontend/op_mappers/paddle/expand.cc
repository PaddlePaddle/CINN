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

#include <absl/types/optional.h>

#include <string>

#include "cinn/common/context.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ExpandV2OpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("shape"));
  auto expand_shape = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "shape");

  auto x = ctx.GetVar(x_name);

  CHECK_GT(expand_shape.size(), x->shape.size())
      << "The number of elements of 'shape' for "
         "expand_v2 op must be greater than or equal to the rank of the input.";

  auto vec_in_dims = x->shape;
  auto diff        = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> out_shape(vec_in_dims.size());

  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    CHECK_NE(expand_shape[i], 0) << "The expanded size cannot be zero.";
    if (i < diff) {
      CHECK_GT(expand_shape[i], 0) << "The expanded size for non-existing dimensions must be positive.";
      out_shape[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        CHECK_EQ(expand_shape[i], expand_shape[i]);
        out_shape[i] = expand_shape[i];
      } else {
        out_shape[i] = expand_shape[i];
      }
    } else {
      CHECK_EQ(expand_shape[i], -1) << "When the value in shape is negative for expand_v2 op, only -1 is supported";
      out_shape[i] = vec_in_dims[i];
    }
  }

  auto out = ctx.Builder()->BroadcastTo(x, out_shape);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_expand) {
  CINN_REGISTER_OP_MAPPER(expand_v2, cinn::frontend::paddle_mappers::ExpandV2OpMapper)
  return true;
}
