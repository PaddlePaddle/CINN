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
template <typename T>
void SetData(hlir::framework::Tensor tensor, const common::Target& target, const std::vector<T>& values) {
  auto* data = tensor->mutable_data<T>(target);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
    return;
  }
#endif
  CHECK(target == common::DefaultHostTarget());
  std::copy(values.begin(), values.end(), data);
}
template <>
void SetData(hlir::framework::Tensor tensor, const common::Target& target, const std::vector<bool>& values) {
  auto* data = tensor->mutable_data<bool>(target);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    bool* array = new bool[values.size()];
    for (unsigned int i = 0; i < values.size(); i++) {
      array[i] = static_cast<bool>(values[i]);
    }
    auto src_ptr = static_cast<const void*>(array);
    cudaMemcpy(data, src_ptr, values.size() * sizeof(bool), cudaMemcpyHostToDevice);
    delete[] array;
    return;
  }
#endif
  CHECK(target == common::DefaultHostTarget());
  std::copy(values.begin(), values.end(), data);
}
}  // namespace

TEST(where, where_ok) {
  std::vector<bool> c_data{false, false, true, true};
  std::vector<float> x_data{0.9383, 0.1983, 3.2, 1.2};
  std::vector<float> y_data{1.0, 1.0, 1.0, 1.0};

  std::vector<float> output_ref{1.0, 1.0, 3.2, 1.2};

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::string op_type = "where";
  auto* where         = OpMapperRegistry::Global()->Find(op_type);
  CHECK_NOTNULL(where);

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
  std::string c_name       = "before_0.tmp_0";
  std::string x_name       = "before_0.tmp_1";
  std::string y_name       = "before_0.tmp_2";
  std::vector<int> c_shape = {1, 4};
  std::vector<int> x_shape = {1, 4};
  std::vector<int> y_shape = {1, 4};
  create_input(c_name, Bool(), c_shape);
  create_input(x_name, Float(32), x_shape);
  create_input(y_name, Float(32), y_shape);

  // output
  std::string out_name = "where_0.tmp_0";

  paddle::cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  op_desc.SetInput("Condition", {c_name});
  op_desc.SetInput("X", {x_name});
  op_desc.SetInput("Y", {y_name});
  op_desc.SetOutput("Out", {out_name});

  where->Run(op_desc, ctx);

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

  auto c_ten = exe_scope->GetTensor(var_model_to_program_map.at(c_name));
  auto x_ten = exe_scope->GetTensor(var_model_to_program_map.at(x_name));
  auto y_ten = exe_scope->GetTensor(var_model_to_program_map.at(y_name));

  SetData<bool>(c_ten, target, c_data);
  SetData<float>(x_ten, target, x_data);
  SetData<float>(y_ten, target, y_data);

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
