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

#include <variant>

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void ScaleOpMapper(const paddle::cpp::OpDesc& op_desc, const cinn::frontend::OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x      = ctx.GetVar(x_name);

  auto scale            = utils::GetAttrOrDefault<float>(op_desc, "scale", 1.0f);
  auto bias             = utils::GetAttrOrDefault<float>(op_desc, "bias", 0.0f);
  auto bias_after_scale = utils::GetAttrOrDefault<bool>(op_desc, "bias_after_scale", true);

  auto out = ctx.Builder()->scale(x, scale, bias, bias_after_scale);
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(scale) {
  CINN_REGISTER_OP_MAPPER(scale, cinn::frontend::op_mappers::ScaleOpMapper)
  return true;
}
