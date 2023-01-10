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

void BatchNormOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  auto get_output_name = [&op_desc](const std::string& op_name) {
    CHECK_EQ(op_desc.Output(op_name).size(), 1UL);
    return op_desc.Output(op_name).front();
  };

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

  auto epsilon     = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);
  auto momentum    = utils::GetAttrOrDefault<float>(op_desc, "momentum", 0.9f);
  auto data_layout = utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto x           = ctx.GetVar(x_name);
  auto scale       = ctx.GetVar(scale_name);
  auto bias        = ctx.GetVar(bias_name);
  auto mean        = ctx.GetVar(mean_name);
  auto variance    = ctx.GetVar(variance_name);

  auto is_test = utils::GetAttrOrDefault<bool>(op_desc, "is_test", false);

  std::vector<std::string> output_names;
  if (is_test) {
    output_names = {"Y"};
    VLOG(4) << "Invoke batch_norm OpMapper with test mode";
  } else {
    output_names = {"Y", "SavedMean", "SavedVariance", "MeanOut", "VarianceOut"};
    VLOG(4) << "Invoke batch_norm OpMapper with train mode";
  }

  auto outs = ctx.Builder()->BatchNorm(x, scale, bias, mean, variance, epsilon, momentum, data_layout, is_test);
  CHECK_EQ(outs.size(), output_names.size()) << "batch_norm API's should return" << output_names.size() << "Variables!";

  for (int i = 0; i < outs.size(); i++) {
    auto out_name = get_output_name(output_names[i]);

    bool replace_old_var = false;
    if (output_names[i] == "MeanOut" || output_names[i] == "VarianceOut") {
      // For batch_norm train, the MeanOut and VarianceOut share memory with Mean,
      // so that its name is the same as the input Mean and Variance,
      // Which means we need agree the out var replace the input var.
      replace_old_var = true;
    }

    ctx.AddVar(out_name, outs[i], replace_old_var);
    ctx.AddVarModelToProgram(out_name, outs[i]->id);
  }
}

void BatchNormGradOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  std::unordered_map<std::string, std::string> input_names_map;
  auto get_input_var = [&op_desc, &ctx, &input_names_map](const std::string& op_name) {
    CHECK_EQ(op_desc.Input(op_name).size(), 1UL);
    auto var_name = op_desc.Input(op_name).front();
    input_names_map.emplace(op_name, var_name);
    return ctx.GetVar(var_name);
  };

  std::unordered_map<std::string, std::string> output_names_map;
  auto get_output_name = [&op_desc, &output_names_map](const std::string& op_name) -> std::string {
    if (op_desc.Output(op_name).empty()) {
      CHECK_NE(op_name, paddle::GradVarName("X")) << "The input X should not empty.";
      return "";
    }

    CHECK_EQ(op_desc.Output(op_name).size(), 1UL);
    auto var_name = op_desc.Output(op_name).front();
    output_names_map.emplace(op_name, var_name);
    return var_name;
  };

  std::vector<std::string> output_names = {get_output_name(paddle::GradVarName("X")),
                                           get_output_name(paddle::GradVarName("Scale")),
                                           get_output_name(paddle::GradVarName("Bias"))};

  auto x              = get_input_var("X");
  auto dy             = get_input_var(paddle::GradVarName("Y"));
  auto scale          = get_input_var("Scale");
  auto saved_mean     = get_input_var("SavedMean");
  auto saved_variance = get_input_var("SavedVariance");

  auto data_layout = utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto epsilon     = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);

  auto get_arg_debug_info = [](const std::unordered_map<std::string, std::string>& names_map) {
    std::string res;
    for (const auto& pair : names_map) {
      res.append(pair.first + ":" + pair.second + ", ");
    }
    return res;
  };

  VLOG(4) << "{" << get_arg_debug_info(output_names_map) << "} = batch_norm_grad("
          << get_arg_debug_info(input_names_map) << ", data_layout=" << data_layout << ", epsilon=" << epsilon << ")";

  // batch norm grad, output(grad_x, grad_scale, grad_bias)
  auto outs = ctx.Builder()->BatchNormGrad(dy, x, scale, saved_mean, saved_variance, epsilon, data_layout);
  CHECK_EQ(outs.size(), 3ul) << "batch_norm_grad API's should return 3 Variable!";

  for (int i = 0; i < outs.size(); i++) {
    if (output_names[i].empty()) {
      continue;
    }

    ctx.AddVar(output_names[i], outs[i]);
    ctx.AddVarModelToProgram(output_names[i], outs[i]->id);
  }
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_batchnorm) {
  CINN_REGISTER_OP_MAPPER(batch_norm, cinn::frontend::paddle_mappers::BatchNormOpMapper)
  CINN_REGISTER_OP_MAPPER(batch_norm_grad, cinn::frontend::paddle_mappers::BatchNormGradOpMapper)
  return true;
}
