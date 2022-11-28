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

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void UnSqueeze2OpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x      = ctx.GetVar(x_name);

  const auto& origin_axes = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axes");

  int axis_pos = 0, x_pos = 0;
  int out_rank = origin_axes.size() + x->shape.size();

  const auto& axes = utils::GetPositiveAxes(origin_axes, out_rank);

  std::vector<int> new_shape(out_rank);
  for (int i = 0; i < out_rank; ++i) {
    if (axis_pos < axes.size() && axes[axis_pos] == i) {
      new_shape[i] = 1;
      axis_pos++;
    } else if (x_pos < x->shape.size()) {
      new_shape[i] = x->shape[x_pos];
      x_pos++;
    }
  }

  VLOG(4) << "unsqueeze x[" << cinn::utils::Join(x->shape, ",") << "] at axes[" << cinn::utils::Join(axes, ",")
          << "] to output[" << cinn::utils::Join(new_shape, ",") << "]";

  auto out = ctx.Builder()->Reshape(x, new_shape);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);

  // squeeze2 adds an intermediate output(XShape) based on squeeze,
  // the XShape is used to carry the shape and lod of X which will be used in
  // squeeze_grad, in this way, the framework can reuse the memory of X
  // immediately the squeeze2_op is finished.
  // Considering compatibility issues, we could not fix squeeze2_op
  CHECK_EQ(op_desc.Output("XShape").size(), 1UL);
  auto xshape_name = op_desc.Output("XShape").front();

  auto xshape = ctx.Builder()->Identity(x);

  ctx.AddVar(xshape_name, xshape);
  ctx.AddVarModelToProgram(xshape_name, xshape->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_unsqueeze) {
  CINN_REGISTER_OP_MAPPER(unsqueeze2, cinn::frontend::paddle_mappers::UnSqueeze2OpMapper)
  return true;
}
