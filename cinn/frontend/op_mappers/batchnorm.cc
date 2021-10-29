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

void BatchnormOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Scale").size(), 1UL);
  auto scale_name = op_desc.Input("Scale").front();
  CHECK_EQ(op_desc.Input("Bias").size(), 1UL);
  auto bias_name = op_desc.Input("Bias").front();
  CHECK_EQ(op_desc.Input("Mean").size(), 1UL);
  auto mean_name = op_desc.Input("Mean").front();
  CHECK_EQ(op_desc.Input("Variance").size(), 1UL);
  auto variance_name = op_desc.Input("Variance").front();
  CHECK(!op_desc.Output("Y").empty());
  auto out_name = op_desc.Output("Y").front();

  auto epsilon     = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);
  auto momentum    = utils::GetAttrOrDefault<float>(op_desc, "momentum", 0.9f);
  auto data_layout = utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto x           = ctx.GetVar(x_name);
  auto scale       = ctx.GetVar(scale_name);
  auto bias        = ctx.GetVar(bias_name);
  auto mean        = ctx.GetVar(mean_name);
  auto variance    = ctx.GetVar(variance_name);
  auto out         = ctx.Builder()->batchnorm(x, scale, bias, mean, variance, epsilon, momentum, data_layout);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void BatchNormTrainOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  auto get_input_var = [&op_desc, &ctx](const std::string& op_name) {
    CHECK_EQ(op_desc.Input(op_name).size(), 1UL);
    auto var_name = op_desc.Input(op_name).front();
    return ctx.GetVar(var_name);
  };

  auto get_output_name = [&op_desc](const std::string& op_name) {
    CHECK_EQ(op_desc.Output(op_name).size(), 1UL);
    return op_desc.Output(op_name).front();
  };

  auto x        = get_input_var("X");
  auto scale    = get_input_var("Scale");
  auto bias     = get_input_var("Bias");
  auto mean     = get_input_var("Mean");
  auto variance = get_input_var("Variance");

  auto data_layout = utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto momentum    = utils::GetAttrOrDefault<float>(op_desc, "momentum", 0.9f);
  auto epsilon     = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);

  // batch norm training, output{y, moving_mean, moving_variance, save_mean, save_variance}
  auto out_vec = ctx.Builder()->batch_norm_train(x, scale, bias, mean, variance, epsilon, momentum, data_layout);
  CHECK_EQ(out_vec.size(), 5ul) << "batch_norm_train API's should return 5 Variable!";

  auto y_name = get_output_name("Y");
  ctx.AddVar(y_name, out_vec[0]);
  ctx.AddVarModelToProgram(y_name, out_vec[0]->id);

  auto mean_out_name = get_output_name("MeanOut");
  ctx.AddVar(mean_out_name, out_vec[1]);
  ctx.AddVarModelToProgram(mean_out_name, out_vec[1]->id);

  auto variance_out_name = get_output_name("VarianceOut");
  ctx.AddVar(variance_out_name, out_vec[2]);
  ctx.AddVarModelToProgram(variance_out_name, out_vec[2]->id);

  auto save_mean_name = get_output_name("SavedMean");
  ctx.AddVar(save_mean_name, out_vec[3]);
  ctx.AddVarModelToProgram(save_mean_name, out_vec[3]->id);

  auto saved_variance_name = get_output_name("SavedVariance");
  ctx.AddVar(saved_variance_name, out_vec[4]);
  ctx.AddVarModelToProgram(saved_variance_name, out_vec[4]->id);
}

void BatchNormGradOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  auto get_input_var = [&op_desc, &ctx](const std::string& op_name) {
    CHECK_EQ(op_desc.Input(op_name).size(), 1UL);
    auto var_name = op_desc.Input(op_name).front();
    return ctx.GetVar(var_name);
  };

  auto get_output_name = [&op_desc](const std::string& op_name) {
    CHECK_EQ(op_desc.Output(op_name).size(), 1UL);
    return op_desc.Output(op_name).front();
  };

  auto x              = get_input_var("X");
  auto dy             = get_input_var(paddle::GradVarName("Y"));
  auto scale          = get_input_var("Scale");
  auto saved_mean     = get_input_var("SavedMean");
  auto saved_variance = get_input_var("SavedVariance");

  auto data_layout = utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto epsilon     = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);

  // batch norm grad, output(grad_x, grad_scale, grad_bias)
  auto out_vec = ctx.Builder()->batch_norm_grad(dy, x, scale, saved_mean, saved_variance, epsilon, data_layout);
  CHECK_EQ(out_vec.size(), 3ul) << "batch_norm_grad API's should return 3 Variable!";

  auto dx_name = get_output_name(paddle::GradVarName("X"));
  ctx.AddVar(dx_name, out_vec[0]);
  ctx.AddVarModelToProgram(dx_name, out_vec[0]->id);

  auto dscale_name = get_output_name(paddle::GradVarName("Scale"));
  ctx.AddVar(dscale_name, out_vec[1]);
  ctx.AddVarModelToProgram(dscale_name, out_vec[1]->id);

  auto dbias_name = get_output_name(paddle::GradVarName("Bias"));
  ctx.AddVar(dbias_name, out_vec[2]);
  ctx.AddVarModelToProgram(dbias_name, out_vec[2]->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(batchnorm) {
  CINN_REGISTER_OP_MAPPER(batchnorm, cinn::frontend::op_mappers::BatchnormOpMapper)
  CINN_REGISTER_OP_MAPPER(batchnorm_train, cinn::frontend::op_mappers::BatchNormTrainOpMapper)
  CINN_REGISTER_OP_MAPPER(batchnorm_grad, cinn::frontend::op_mappers::BatchNormGradOpMapper)
  return true;
}
