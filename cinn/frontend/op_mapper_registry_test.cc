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

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <typeinfo>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/frontend/decomposer/test_helper.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/paddle_model_to_program.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/utils/registry.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace frontend {

using ::cinn::common::Target;
using ::cinn::frontend::paddle::cpp::OpDesc;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Scope;
using ::cinn::hlir::framework::Shape;
using ::cinn::hlir::framework::Tensor;
using ::cinn::utils::TransValidVarName;

TEST(OpMapperRegistryTest, basic) {
  auto kernel = OpMapperRegistry::Global()->Find("sigmoid");
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(typeid(*kernel), typeid(OpMapper));
  ASSERT_EQ(kernel->name, "sigmoid");
}

void SetRandData(hlir::framework::Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  // std::random_device seed;
  // std::default_random_engine engine(seed());
  // std::uniform_real_distribution<float> dist(0.f, 1.f);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = 1;  // dist(engine);  // All random data
  }

#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

// Test that the reverse of HW gets same result
// between OpMapper and PaddleModelToProgram
TEST(OpMapperRegistryTest, conv2d_reverse) {
  std::unique_ptr<OpDesc> op_desc = std::make_unique<OpDesc>();
  op_desc->SetType("conv2d");
  op_desc->SetInput("Input", {"input"});
  op_desc->SetInput("Filter", {"filter"});
  op_desc->SetOutput("Output", {"output"});

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  // Get runtime Program from OpMapper
  std::shared_ptr<Scope> net_scope = Scope::Create();

  net_scope->Var<Tensor>("input");
  net_scope->Var<Tensor>("filter");
  net_scope->Var<Tensor>("output");

  Tensor input_tensor  = net_scope->GetTensor("input");
  Tensor filter_tensor = net_scope->GetTensor("filter");
  Tensor output_tensor = net_scope->GetTensor("output");

  input_tensor->Resize(Shape({32, 16, 32, 32}));
  filter_tensor->Resize(Shape({32, 16, 3, 3}));

  SetRandData(input_tensor, target);
  SetRandData(filter_tensor, target);

  NetBuilder net_builder("net_builder_name");
  net_builder.CreateInput(Float(32), {32, 16, 32, 32}, "input");
  net_builder.CreateInput(Float(32), {32, 16, 3, 3}, "filter");
  std::unordered_map<std::string, Variable> var_map;
  std::unordered_map<std::string, std::string> var_model_to_program_map({{"input", "input"}, {"filter", "filter"}});

  OpMapperContext op_mapper_ctx(*net_scope, target, &net_builder, &var_map, &var_model_to_program_map);
  const OpMapper* op_mapper = OpMapperRegistry::Global()->Find(op_desc->Type());
  ASSERT_NE(op_mapper, nullptr);
  std::cout << op_mapper->name << std::endl;
  op_mapper->Run(*op_desc, op_mapper_ctx);

  Program net_program = net_builder.Build();

  for (size_t i = 0; i < net_program.size(); ++i) {
    // OpMapperContext doesn't add output into scope
    const auto& instr   = net_program[i];
    const auto& outputs = instr.GetOutputs();
    for (const auto& var : outputs) {
      net_scope->Var<Tensor>(var->id);
      net_scope->GetTensor(var->id);
    }
  }

  std::cout << net_program << std::endl;
  auto net_graph = std::make_shared<Graph>(net_program, target);
  GraphCompiler net_gc(target, net_scope, net_graph);
  auto net_runtime_prog = net_gc.Build();

  input_tensor  = net_scope->GetTensor("input");
  filter_tensor = net_scope->GetTensor("filter");

  input_tensor->Resize(Shape({16, 3, 224, 224}));
  filter_tensor->Resize(Shape({3, 3, 5, 5}));

  SetRandData(input_tensor, target);
  SetRandData(filter_tensor, target);

  net_runtime_prog->Execute();
  /*
  std::vector<float> net_data(output_tensor->shape().numel());
  CopyToVector(output_tensor, &net_data);
  for (auto x : net_data) {
    std::cout << x << std::endl;
  }

  // Get runtime Program from PaddleModelToProgram
  /*
  std::shared_ptr<Scope> cinn_scope = Scope::Create();
  cinn_scope->Var<Tensor>("input");
  cinn_scope->Var<Tensor>("filter");
  PaddleModelToProgram model_to_prog(cinn_scope.get(), target);
  model_to_prog.AddOp(*op_desc);
  std::unique_ptr<Program> cinn_program = model_to_prog.GetProgram();
  auto cinn_graph                       = std::make_shared<Graph>(*cinn_program, target);
  GraphCompiler cinn_gc(target, cinn_scope, cinn_graph);
  auto cinn_runtime_prog = cinn_gc.Build();
  */
  // Execute and expect same result
}

}  // namespace frontend
}  // namespace cinn
