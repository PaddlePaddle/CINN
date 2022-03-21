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

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace science_mappers {

void FillConstantOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto y_name = op_desc.Output("Y").front();

  auto shape = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "shape");
  // TODO(jiangcheng): value support different datatype, not just float
  auto value = utils::GetAttrOrDefault<float>(op_desc, "value");

  VLOG(4) << "fill constant (" << value << ") with shape (" << cinn::utils::Join(shape, ",") << ").";

  const auto& cinn_name = cinn::utils::TransValidVarName(y_name);
  CheckVarNameValid(cinn_name);

  auto out = ctx.Builder()->FillConstant<float>(shape, value, cinn_name);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

void BroadcastOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Y").size(), 1UL);
  auto y_name = op_desc.Output("Y").front();

  CHECK(op_desc.HasAttr("shape")) << "The broadcast_p operator should has 'shape' attribute, but " << x_name
                                  << "'s broadcast hasn't.";

  auto y_shape = op_desc.GetAttr<std::vector<int>>("shape");
  auto x       = ctx.GetVar(x_name);

  VLOG(4) << "Broadcast " << x_name << " from shape (" << cinn::utils::Join(x->shape, ",") << ") to shape ("
          << cinn::utils::Join(y_shape, ",") << ").";

  auto out = ctx.Builder()->BroadcastTo(x, y_shape);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_broadcast) {
  CINN_REGISTER_OP_MAPPER(fill_constant_p, cinn::frontend::science_mappers::FillConstantOpMapper)
  CINN_REGISTER_OP_MAPPER(broadcast_p, cinn::frontend::science_mappers::BroadcastOpMapper)

  return true;
}
