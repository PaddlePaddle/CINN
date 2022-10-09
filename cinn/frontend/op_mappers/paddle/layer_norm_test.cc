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

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cinn/common/target.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/optimize.h"
#include "cinn/frontend/paddle/cpp/op_desc.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/utils/data_util.h"
#include "cinn/utils/string.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace frontend {

namespace {
std::vector<float> x_data{
    2.6568303,   -0.91297066, 5.598573,   3.4791756,   0.5435751,   -1.6057454, -4.0341325, 0.69835335,  -3.461902,
    -1.116006,   -7.140052,   5.5378814,  1.0436267,   4.28496,     1.0147153,  1.7059532,  -4.1363974,  2.2155983,
    -1.9488351,  0.19683905,  0.7669514,  6.778486,    -4.103779,   -2.3099174, 7.5175047,  0.37112173,  0.19513914,
    0.32614738,  -0.89420015, 0.25999096, -7.1823254,  -1.5887612,  2.8081188,  -2.027434,  -0.30117422, 0.09851207,
    1.8846271,   1.3924855,   1.2555653,  -1.8026311,  -2.390344,   0.53275025, 3.2047665,  -0.2750236,  0.3840474,
    1.1406593,   -0.9980533,  0.4702035,  -0.89256626, 0.69087005,  -3.2366154, -3.7514133, -0.804769,   -2.4814897,
    -2.8071048,  5.2047596,   0.52646077, -1.4753606,  -0.32114556, -6.542912,  4.20583,    0.2072492,   -1.7105737,
    0.9126658,   0.14393447,  0.8458823,  1.1466345,   2.2417424,   -5.2724614, 2.0421154,  2.4660935,   4.2775483,
    -1.9298731,  -4.233445,   0.0754128,  2.3246357,   -0.9613564,  -5.2187233, -5.5004983, 0.07136486,  -3.1703022,
    3.6736274,   -0.4343733,  -8.923371,  4.8379946,   3.7862253,   1.5748019,  3.7755294,  -2.2443461,  -1.5790445,
    -3.085116,   -4.31788,    2.2910767,  2.7384732,   -5.559047,   3.0368736,  2.6795564,  -0.82532924, -1.1076185,
    -0.24732637, 2.2325885,   1.2674553,  -3.6190743,  0.9010671,   -3.140488,  -0.7214774, 5.1392756,   -0.18481685,
    0.7630325,   -2.9616263,  0.27014923, -6.8555927,  -6.589393,   -1.2736413, 2.7218244,  -5.998231,   -5.208979,
    -5.5303316,  3.4857583,   -1.1865834};

std::vector<float> scale_data{-5.766006,  -0.4156053, 0.25646606, 0.02883168, 3.8052704,   1.7682005, 3.6355832,
                              -0.3673084, 2.8386993,  -3.868579,  3.6602883,  -0.39298716, -3.695925, 4.4462757,
                              7.0570855,  -3.5902069, -3.105476,  6.8345404,  0.60832334,  2.9107456};

std::vector<float> bias_data{-1.3113174,  1.3627944,  3.090474,   0.68646824, -2.4245484, -4.676319,  3.34968,
                             1.1968151,   4.697088,   -2.30895,   -1.4362527, 1.9465477,  0.31132585, -4.1655216,
                             -0.52577823, -1.3339206, -3.4677775, 0.24170296, 4.293887,   -0.5902035};

// `output_ref` is got from paddle.static.nn.layer_norm
std::vector<float> output_ref{
    -5.5870366,  1.5081296,   3.5112798,  0.71509576,  -2.060959,  -5.6691036, -1.3903475, 1.1443406,  1.4925791,
    -0.7160236,  -9.683688,   1.3090309,  -0.6067683,  1.3444542,  1.1648834,  -2.9526386, 0.67818254, 4.3879538,
    3.8885348,   -0.62060076, -2.3895357, 0.49952024,  2.7456644,  0.66396695, 6.3637705,  -4.5657387, 3.3758774,
    1.1790383,   3.7452564,   -2.4157097, -9.900394,   2.164141,   -2.7517893, -7.240647,  -1.576194,  -1.2507149,
    -5.139817,   2.8639576,   4.501098,   -2.3975883,  2.6488564,  1.1572046,  3.494326,   0.69131714, -0.7708883,
    -3.3671665,  2.8986416,   1.0244026,  4.465946,    -4.469884,  -5.2022123, 2.432652,   0.48111048, -7.3830886,
    -6.5614576,  -9.889604,   -4.9960856, -1.9247487,  4.384855,   -7.474834,  -10.020681, 1.305637,   2.956863,
    0.6974167,   -1.9839413,  -4.0454125, 5.022311,    0.8897142,  -0.2535091, -5.278245,  1.9060199,  1.3432761,
    2.5150836,   -10.333455,  0.1253173,  -4.4378223,  -2.6488004, -11.551401, 3.185361,   -0.3257001, 3.5571413,
    0.9172361,   3.0704093,   0.61567813, 2.895646,    -2.724933,  5.1106343,  0.79255366, 3.0362904,  -0.7663127,
    -4.439463,   2.4046443,   -2.2201414, -0.56309193, -11.204735, -4.542739,  -5.932641,  -1.0411981, 4.1316147,
    -0.66547096, -7.252135,   1.0507741,  2.920014,    0.70505416, -4.4262137, -4.367493,  10.156422,  1.0755657,
    6.413514,    -0.47440732, 0.2543828,  2.5761626,   5.947683,   -4.1000805, 7.74532,    3.5264506,  0.02642507,
    -8.0845175,  5.141467,    -0.47396272};

// `mean_ref` is got from paddle.static.nn.layer_norm
std::vector<float> mean_ref{0.231002, 0.17222133, -0.69111407, -0.19329107, -0.1549799, -1.3244542};

// `var_ref` is got from paddle.static.nn.layer_norm
std::vector<float> var_ref{10.701677, 10.11525, 6.12105, 8.482212, 12.753654, 11.918896};

void SetData(hlir::framework::Tensor tensor, const common::Target& target, const std::vector<float>& values) {
  auto* data = tensor->mutable_data<float>(target);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    return;
  }
#endif
  CHECK(target == common::DefaultHostTarget());
  std::copy(values.begin(), values.end(), data);
}
}  // namespace

TEST(layer_norm, acc) {
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::string op_type = "layer_norm";
  auto* layer_norm    = OpMapperRegistry::Global()->Find(op_type);
  CHECK_NOTNULL(layer_norm);

  auto symbol_scope = hlir::framework::Scope::Create();
  std::unordered_map<std::string, Variable> var_map;
  std::unordered_map<std::string, std::string> var_model_to_program_map;
  std::unordered_set<std::string> fetch_var_names;
  NetBuilder builder("net_builder");
  frontend::OpMapperContext ctx(*symbol_scope, target, &builder, &var_map, &var_model_to_program_map, &fetch_var_names);

  auto create_input = [&](const auto& name, const auto& type, const auto& shape) {
    auto id    = utils::TransValidVarName(name);
    auto input = ctx.Builder()->CreateInput(type, shape, id);
    ctx.AddVar(name, input);
    ctx.AddVarModelToProgram(name, input.id().data());
  };
  std::string x_name           = "embedding_0.tmp_0";
  std::vector<int> x_shape     = {2, 3, 4, 5};
  std::string scale_name       = "pre_encoder_layer_norm_scale";
  std::vector<int> scale_shape = {20};
  std::string bias_name        = "pre_encoder_layer_norm_bias";
  std::vector<int> bias_shape  = {20};
  create_input(x_name, Float(32), x_shape);
  create_input(scale_name, Float(32), scale_shape);
  create_input(bias_name, Float(32), bias_shape);

  std::string y_name        = "layer_norm_0.tmp_2";
  std::string mean_name     = "layer_norm_0.tmp_0";
  std::string variance_name = "layer_norm_0.tmp_1";

  int begin_norm_axis = 2;
  float epsilon       = 9.999999960041972e-13f;

  paddle::cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  op_desc.SetInput("X", {x_name});
  op_desc.SetInput("Scale", {scale_name});
  op_desc.SetInput("Bias", {bias_name});
  op_desc.SetOutput("Y", {y_name});
  op_desc.SetOutput("Mean", {mean_name});
  op_desc.SetOutput("Variance", {variance_name});
  op_desc.SetAttr("begin_norm_axis", begin_norm_axis);
  op_desc.SetAttr("epsilon", epsilon);

  layer_norm->Run(op_desc, ctx);

  ctx.AddFetchVarName(var_model_to_program_map.at(y_name));
  ctx.AddFetchVarName(var_model_to_program_map.at(mean_name));
  ctx.AddFetchVarName(var_model_to_program_map.at(variance_name));

  auto program = builder.Build();
  // auto cinn_graph = Optimize(&program, fetch_var_names, target);
  auto cinn_graph = std::make_shared<hlir::framework::Graph>(program, target);
  VLOG(4) << "graph:\n" << cinn_graph->Visualize();

  auto exe_scope = BuildScope(target, cinn_graph);
  hlir::framework::GraphCompiler gc(target, exe_scope, cinn_graph);
  hlir::framework::GraphCompiler::CompileOptions options;
  options.attached_code              = "";
  options.with_instantiate_variables = true;
  auto compile_obj                   = gc.Build(options, std::move(fetch_var_names));

  exe_scope->Var<hlir::framework::Tensor>(var_model_to_program_map.at(x_name));
  exe_scope->Var<hlir::framework::Tensor>(var_model_to_program_map.at(scale_name));
  exe_scope->Var<hlir::framework::Tensor>(var_model_to_program_map.at(bias_name));

  auto x_ten     = exe_scope->GetTensor(var_model_to_program_map.at(x_name));
  auto scale_ten = exe_scope->GetTensor(var_model_to_program_map.at(scale_name));
  auto bias_ten  = exe_scope->GetTensor(var_model_to_program_map.at(bias_name));

  SetData(x_ten, target, x_data);
  SetData(scale_ten, target, scale_data);
  SetData(bias_ten, target, bias_data);

  compile_obj.runtime_program->Execute();

  auto compare = [&](auto var, const auto& ref) {
    auto var_tensor        = exe_scope->GetTensor(std::string(var->id));
    std::vector<float> res = GetTensorData<float>(var_tensor, target);
    for (int i = 0; i < res.size(); i++) {
      ASSERT_LT(std::abs(res[i] - ref[i]), 1e-5f + 1e-8f * std::abs(ref[i]));
    }
    VLOG(4) << "---- variable info:";
    VLOG(4) << "     variable name: " << var->id;
    VLOG(4) << "     variable shape: " << utils::Join(var->shape, ", ");
    VLOG(4) << "     variable value: " << utils::Join(res, ", ");
  };

  compare(ctx.GetVar(y_name), output_ref);
  compare(ctx.GetVar(mean_name), mean_ref);
  compare(ctx.GetVar(variance_name), var_ref);
}
}  // namespace frontend
}  // namespace cinn
