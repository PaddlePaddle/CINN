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
namespace paddle_mappers {

enum class EltwiseType { kUnk = 0, kAdd, kDiv, kMul, kSub };

template <EltwiseType Type>
struct OpBuilder {};

#define ELTWISE_SPEC(enum_t, function)                                                               \
  template <>                                                                                        \
  struct OpBuilder<enum_t> {                                                                         \
    constexpr static Variable (NetBuilder::*func)(const Variable&, const Variable&, int){&function}; \
  }
ELTWISE_SPEC(EltwiseType::kAdd, NetBuilder::Add);
ELTWISE_SPEC(EltwiseType::kDiv, NetBuilder::Divide);
ELTWISE_SPEC(EltwiseType::kMul, NetBuilder::Multiply);
ELTWISE_SPEC(EltwiseType::kSub, NetBuilder::Subtract);
#undef ELTWISE_SPEC

void AddOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Add(x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

template <EltwiseType Type>
void ElementwiseOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  VLOG(5) << "Elementwise operator mapping type: " << static_cast<int>(Type);
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);

  auto x   = ctx.GetVar(x_name);
  auto y   = ctx.GetVar(y_name);
  auto out = (ctx.Builder()->*OpBuilder<Type>::func)(x, y, axis);

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
  auto outs = ctx.Builder()->ElementwiseAddGrad(dout, x, y, axis);
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
  auto out = ctx.Builder()->Multiply(x, y, axis);

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

  auto out = ctx.Builder()->Sum(xs);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_elementwise) {
  using namespace cinn::frontend::paddle_mappers;
  CINN_REGISTER_OP_MAPPER(add, AddOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_add, ElementwiseOpMapper<EltwiseType::kAdd>)
  CINN_REGISTER_OP_MAPPER(elementwise_add_grad, ElementwiseAddGradOpMapper)
  CINN_REGISTER_OP_MAPPER(elementwise_mul, ElementwiseOpMapper<EltwiseType::kMul>)
  CINN_REGISTER_OP_MAPPER(elementwise_div, ElementwiseOpMapper<EltwiseType::kDiv>)
  CINN_REGISTER_OP_MAPPER(elementwise_sub, ElementwiseOpMapper<EltwiseType::kSub>)
  CINN_REGISTER_OP_MAPPER(sum, SumOpMapper)
  return true;
}
