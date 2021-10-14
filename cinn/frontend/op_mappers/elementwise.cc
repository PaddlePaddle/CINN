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
namespace op_mappers {

void AddOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.builder_->add(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

void ElementwiseAddOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.builder_->elementwise_add(x, y, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

void ElementwiseMulOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.builder_->elementwise_mul(x, y, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise) {
  CINN_REGISTER_OP_MAPPER(add, cinn::frontend::op_mappers::AddOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_add, cinn::frontend::op_mappers::ElementwiseAddOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_mul, cinn::frontend::op_mappers::ElementwiseMulOpMapper)
  return true;
}
