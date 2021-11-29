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
  auto out = ctx.Builder()->add(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
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
  auto out = ctx.Builder()->elementwise_add(x, y, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ElementwiseAddGradOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Input(paddle::GradVarName("Out")).size(), 1UL);
  auto dout_name = op_desc.Input(paddle::GradVarName("Out")).front();

  CHECK_EQ(op_desc.Output(paddle::GradVarName("X")).size(), 1UL);
  auto dx_name = op_desc.Output(paddle::GradVarName("X")).front();
  CHECK_EQ(op_desc.Output(paddle::GradVarName("Y")).size(), 1UL);
  auto dy_name = op_desc.Output(paddle::GradVarName("Y")).front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  auto x    = ctx.GetVar(x_name);
  auto y    = ctx.GetVar(y_name);
  auto dout = ctx.GetVar(dout_name);
  auto outs = ctx.Builder()->elementwise_add_grad(dout, x, y, axis);
  CHECK_EQ(outs.size(), 2) << "elementwise_add_grad should return 2 variables";

  auto dx = outs.front();
  ctx.AddVar(dx_name, dx);
  ctx.AddVarModelToProgram(dx_name, dx->id);
  auto dy = outs.back();
  ctx.AddVar(dy_name, dy);
  ctx.AddVarModelToProgram(dy_name, dy->id);
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
  auto out = ctx.Builder()->elementwise_mul(x, y, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SumOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  auto out = ctx.Builder()->sum(xs);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(elementwise) {
  CINN_REGISTER_OP_MAPPER(add, cinn::frontend::op_mappers::AddOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_add, cinn::frontend::op_mappers::ElementwiseAddOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_add_grad, cinn::frontend::op_mappers::ElementwiseAddGradOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_mul, cinn::frontend::op_mappers::ElementwiseMulOpMapper)
  CINN_REGISTER_OP_MAPPER(sum, cinn::frontend::op_mappers::SumOpMapper)
  return true;
}
