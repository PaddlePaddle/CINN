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

#include "cinn/backends/cuda_util.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

void Conv2dOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
  auto y_name = op_desc.Input("Filter").front();

  CHECK_EQ(op_desc.Output("Output").size(), 1UL);
  auto out_name = op_desc.Output("Output").front();

  auto strides   = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings  = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups    = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");
  auto x                 = ctx.GetVar(x_name);
  Variable y             = ctx.GetVar(y_name);
  auto out = ctx.Builder()->conv2d(x, y, strides, paddings, dilations, groups, data_format, padding_algorithm);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void DepthwiseConv2dOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
  auto y_name = op_desc.Input("Filter").front();

  CHECK_EQ(op_desc.Output("Output").size(), 1UL);
  auto out_name = op_desc.Output("Output").front();

  auto strides   = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings  = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups    = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");
  auto x                 = ctx.GetVar(x_name);
  auto y                 = ctx.GetVar(y_name);

  Variable out;
  if (ctx.Target().arch == Target::Arch::X86) {
    out = ctx.Builder()->conv2d(x, y, strides, paddings, dilations, groups, data_format, padding_algorithm);
  } else {
    out = ctx.Builder()->depthwise_conv2d(x, y, strides, paddings, dilations, groups, data_format, padding_algorithm);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void Conv2dGradOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  // get dy
  CHECK_EQ(op_desc.Input(paddle::GradVarName("Output")).size(), 1UL);
  auto dy_name = op_desc.Input(paddle::GradVarName("Output")).front();

  // get intput input,filter
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
  auto w_name = op_desc.Input("Filter").front();

  // get dx,dfilter
  CHECK_EQ(op_desc.Output(paddle::GradVarName("Input")).size(), 1UL);
  auto dx_name = op_desc.Output(paddle::GradVarName("Input")).front();

  CHECK_EQ(op_desc.Output(paddle::GradVarName("Filter")).size(), 1UL);
  auto dw_name = op_desc.Output(paddle::GradVarName("Filter")).front();

  auto strides   = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings  = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups    = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");

  auto dy     = ctx.GetVar(dy_name);
  auto x      = ctx.GetVar(x_name);
  auto weight = ctx.GetVar(w_name);

  auto out =
      ctx.Builder()->conv2d_grad(dy, x, weight, strides, paddings, dilations, groups, data_format, padding_algorithm);
  ctx.AddVar(dx_name, out[0]);
  ctx.AddVarModelToProgram(dx_name, out[0]->id);
  ctx.AddVar(dw_name, out[1]);
  ctx.AddVarModelToProgram(dw_name, out[1]->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(conv2d) {
  CINN_REGISTER_OP_MAPPER(conv2d, cinn::frontend::op_mappers::Conv2dOpMapper)
  CINN_REGISTER_OP_MAPPER(depthwise_conv2d, cinn::frontend::op_mappers::DepthwiseConv2dOpMapper)

#ifdef CINN_WITH_CUDNN
  CINN_REGISTER_OP_MAPPER(conv2d_grad, cinn::frontend::op_mappers::Conv2dGradOpMapper)
#endif
  return true;
}
