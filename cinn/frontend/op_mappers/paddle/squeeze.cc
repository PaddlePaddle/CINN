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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void Squeeze2OpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x      = ctx.GetVar(x_name);

  auto axes = utils::GetPositiveAxes(utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axes"), x->shape.size());

  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "squeeze axes: " << cinn::utils::Join(axes, ",");

  std::vector<int> new_shape;
  if (axes.empty()) {
    for (auto dim : x->shape) {
      if (dim != 1) {
        new_shape.emplace_back(dim);
      }
    }
  } else {
    int axis_pos = 0;
    for (int i = 0; i < x->shape.size(); ++i) {
      if (axis_pos < axes.size() && axes[axis_pos] == i) {
        axis_pos++;
        if (x->shape[i] == 1) {
          continue;
        }
      }
      new_shape.emplace_back(x->shape[i]);
    }
  }

  VLOG(4) << "squeeze x[" << cinn::utils::Join(x->shape, ",") << "] at axes[" << cinn::utils::Join(axes, ",")
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

CINN_REGISTER_HELPER(paddle_squeeze) {
  CINN_REGISTER_OP_MAPPER(squeeze2, cinn::frontend::paddle_mappers::Squeeze2OpMapper)

  return true;
}
