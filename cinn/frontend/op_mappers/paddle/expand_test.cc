// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

TEST(expand, acc) {
  // x [1, 3]
  std::vector<float> x_data{1.0f, 2.0f, 3.0f};
  // out [2, 3]
  std::vector<float> output_ref{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::string op_type = "expand";
  auto* expand        = OpMapperRegistry::Global()->Find(op_type);
  CHECK_NOTNULL(expand);

  auto symbol_scope = hlir::framework::Scope::Create();
  std::unordered_map<std::string, Variable> var_map;
  std::unordered_map<std::string, std::string> var_model_to_program_map;
  std::unordered_set<std::string> fetch_var_names;
  NetBuilder builder("net_builder");
  frontend::OpMapperContext ctx(*symbol_scope, target, &builder, &var_map, &var_model_to_program_map, &fetch_var_names);

  // input
  auto create_input = [&](const auto& name, const auto& type, const auto& shape) {
    auto id    = utils::TransValidVarName(name);
    auto input = ctx.Builder()->CreateInput(type, shape, id);
    ctx.AddVar(name, input);
    ctx.AddVarModelToProgram(name, input.id().data());
  };
  std::string x_name       = "before_0.tmp_0";
  std::vector<int> x_shape = {1, 3};
  create_input(x_name, Float(32), x_shape);

  // output
  std::string out_name = "expand_0.tmp_0";

  // attributes
  std::vector<int> expand_times = {2, 1};

  paddle::cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  op_desc.SetInput("X", {x_name});
  op_desc.SetOutput("Out", {out_name});
  op_desc.SetAttr("expand_times", expand_times);

  expand->Run(op_desc, ctx);

  ctx.AddFetchVarName(var_model_to_program_map.at(out_name));

  auto program    = builder.Build();
  auto cinn_graph = std::make_shared<hlir::framework::Graph>(program, target);
  VLOG(4) << "graph:\n" << cinn_graph->Visualize();

  auto exe_scope = BuildScope(target, cinn_graph);
  hlir::framework::GraphCompiler gc(target, exe_scope, cinn_graph);
  hlir::framework::GraphCompiler::CompileOptions options;
  options.attached_code              = "";
  options.with_instantiate_variables = true;
  auto compile_obj                   = gc.Build(options, std::move(fetch_var_names));

  exe_scope->Var<hlir::framework::Tensor>(var_model_to_program_map.at(x_name));

  auto x_ten = exe_scope->GetTensor(var_model_to_program_map.at(x_name));

  SetData(x_ten, target, x_data);

  compile_obj.runtime_program->Execute();

  auto compare = [&](auto var, const auto& ref) {
    auto var_tensor        = exe_scope->GetTensor(std::string(var->id));
    std::vector<float> res = GetTensorData<float>(var_tensor, target);
    for (int i = 0; i < res.size(); ++i) {
      ASSERT_LT(std::abs(res[i] - ref[i]), 1e-5f + 1e-8f * std::abs(ref[i]));
    }
    VLOG(4) << "---- variable info:";
    VLOG(4) << "     variable name: " << var->id;
    VLOG(4) << "     variable shape: " << utils::Join(var->shape, ", ");
    VLOG(4) << "     variable value: " << utils::Join(res, ", ");
  };

  compare(ctx.GetVar(out_name), output_ref);
}

TEST(expand_v2, acc) {
  // x [2, 1, 4, 5]
  std::vector<float> x_data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,

                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  // out [2, 2, 4, 5]
  std::vector<float> output_ref{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,

                                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,

                                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,

                                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::string op_type = "expand_v2";
  auto* expand_v2     = OpMapperRegistry::Global()->Find(op_type);
  CHECK_NOTNULL(expand_v2);

  auto symbol_scope = hlir::framework::Scope::Create();
  std::unordered_map<std::string, Variable> var_map;
  std::unordered_map<std::string, std::string> var_model_to_program_map;
  std::unordered_set<std::string> fetch_var_names;
  NetBuilder builder("net_builder");
  frontend::OpMapperContext ctx(*symbol_scope, target, &builder, &var_map, &var_model_to_program_map, &fetch_var_names);

  // input
  auto create_input = [&](const auto& name, const auto& type, const auto& shape) {
    auto id    = utils::TransValidVarName(name);
    auto input = ctx.Builder()->CreateInput(type, shape, id);
    ctx.AddVar(name, input);
    ctx.AddVarModelToProgram(name, input.id().data());
  };
  std::string x_name       = "before_0.tmp_0";
  std::vector<int> x_shape = {2, 1, 4, 5};
  create_input(x_name, Float(32), x_shape);

  // output
  std::string out_name = "expand_0.tmp_0";

  // attributes
  std::vector<int> shape = {-1, 2, -1, 5};

  paddle::cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  op_desc.SetInput("X", {x_name});
  op_desc.SetOutput("Out", {out_name});
  op_desc.SetAttr("shape", shape);

  expand_v2->Run(op_desc, ctx);

  ctx.AddFetchVarName(var_model_to_program_map.at(out_name));

  auto program    = builder.Build();
  auto cinn_graph = std::make_shared<hlir::framework::Graph>(program, target);
  VLOG(4) << "graph:\n" << cinn_graph->Visualize();

  auto exe_scope = BuildScope(target, cinn_graph);
  hlir::framework::GraphCompiler gc(target, exe_scope, cinn_graph);
  hlir::framework::GraphCompiler::CompileOptions options;
  options.attached_code              = "";
  options.with_instantiate_variables = true;
  auto compile_obj                   = gc.Build(options, std::move(fetch_var_names));

  exe_scope->Var<hlir::framework::Tensor>(var_model_to_program_map.at(x_name));

  auto x_ten = exe_scope->GetTensor(var_model_to_program_map.at(x_name));

  SetData(x_ten, target, x_data);

  compile_obj.runtime_program->Execute();

  auto compare = [&](auto var, const auto& ref) {
    auto var_tensor        = exe_scope->GetTensor(std::string(var->id));
    std::vector<float> res = GetTensorData<float>(var_tensor, target);
    for (int i = 0; i < res.size(); ++i) {
      ASSERT_LT(std::abs(res[i] - ref[i]), 1e-5f + 1e-8f * std::abs(ref[i]));
    }
    VLOG(4) << "---- variable info:";
    VLOG(4) << "     variable name: " << var->id;
    VLOG(4) << "     variable shape: " << utils::Join(var->shape, ", ");
    VLOG(4) << "     variable value: " << utils::Join(res, ", ");
  };

  compare(ctx.GetVar(out_name), output_ref);
}
}  // namespace frontend
}  // namespace cinn
