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

void ReluOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  auto get_output_name = [&op_desc](const std::string& op_name) {
    CHECK_EQ(op_desc.Output(op_name).size(), 1UL);
    return op_desc.Output(op_name).front();
  };

  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);

  bool compute_mask = utils::GetAttrOrDefault<float>(op_desc, "compute_mask", false);
  auto x            = ctx.GetVar(x_name);
  auto outs         = ctx.Builder()->relu(x, compute_mask);

  std::vector<std::string> output_names;
  if (compute_mask) {
    output_names = {"Out", "Mask"};
  } else {
    output_names = {"Out"};
  }

  for (int i = 0; i < outs.size(); i++) {
    auto out_name = get_output_name(output_names[i]);
    ctx.AddVar(out_name, outs[i]);
    ctx.AddVarModelToProgram(out_name, outs[i]->id);
  }
}

void Relu6OpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto threshold = utils::GetAttrOrDefault<float>(op_desc, "threshold", 6.0f);
  auto x         = ctx.GetVar(x_name);
  auto out       = ctx.Builder()->relu6(x, threshold);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ReluGradOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input(paddle::GradVarName("Out")).size(), 1UL);
  auto dout_name = op_desc.Input(paddle::GradVarName("Out")).front();
  CHECK_EQ(op_desc.Input("Mask").size(), 1UL);
  auto mask_name = op_desc.Input("Mask").front();
  CHECK_EQ(op_desc.Output(paddle::GradVarName("X")).size(), 1UL);
  auto dx_name = op_desc.Output(paddle::GradVarName("X")).front();

  auto dout = ctx.GetVar(dout_name);
  auto mask = ctx.GetVar(mask_name);
  auto dx   = ctx.Builder()->relu_grad(dout, mask);

  ctx.AddVar(dx_name, dx);
  ctx.AddVarModelToProgram(dx_name, dx->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(relu) {
  CINN_REGISTER_OP_MAPPER(relu, cinn::frontend::op_mappers::ReluOpMapper)
  CINN_REGISTER_OP_MAPPER(relu_grad, cinn::frontend::op_mappers::ReluGradOpMapper)
  CINN_REGISTER_OP_MAPPER(relu6, cinn::frontend::op_mappers::Relu6OpMapper)
  return true;
}
