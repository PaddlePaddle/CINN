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

#include <random>

#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/decomposer_registry.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn::frontend {

void SetRandData(hlir::framework::Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(1.f, 2.f);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = dist(engine);  // All random data
  }

#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

void RunProgram(NetBuilder& builder, const std::vector<std::string>& inputs) {
  auto prog     = builder.Build();
  Target target = GetTarget();
  LOG(INFO) << "===================== Before Decomposition =====================";
  for (int i = 0; i < prog.size(); i++) {
    LOG(INFO) << "instruction: " << prog[i];
  }
  ProgramPass::Apply(&prog, target, {"Decomposer"});
  LOG(INFO) << "===================== After Decomposition =====================";
  for (int i = 0; i < prog.size(); i++) {
    LOG(INFO) << "instruction: " << prog[i];
  }
  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);

  auto runtime_program = gc.Build();
  for (auto& in : inputs) {
    scope->Var<hlir::framework::Tensor>(in);
    auto tensor = scope->GetTensor(in);
    SetRandData(tensor, target);
  }
  runtime_program->Execute();
}

TEST(Decomposer, elementwise_add_bcast) {
  NetBuilder builder("elementwise_add");
  auto x   = builder.CreateInput(Float(32), {4, 10, 20, 10});
  auto y   = builder.CreateInput(Float(32), {10, 20});
  auto out = builder.elementwise_add(x, y, 1);

  std::vector<std::string> inputs = {"X", "Y"};
  RunProgram(builder, inputs);
}

TEST(Decomposer, elementwise_add_grad_bcast) {
  NetBuilder builder("elementwise_add_grad");
  auto dout = builder.CreateInput(Float(32), {4, 10, 20, 10});
  auto x    = builder.CreateInput(Float(32), {4, 10, 20, 10});
  auto y    = builder.CreateInput(Float(32), {10, 20});
  auto dx   = builder.elementwise_add_grad(dout, x, y, 1);

  std::vector<std::string> inputs = {"Dout", "X", "Y"};
  RunProgram(builder, inputs);
}

TEST(Decomposer, elementwise_add) {
  NetBuilder builder("elementwise_add");
  auto x   = builder.CreateInput(Float(32), {10, 20});
  auto y   = builder.CreateInput(Float(32), {10, 20});
  auto out = builder.elementwise_add(x, y);

  std::vector<std::string> inputs = {"X", "Y"};
  RunProgram(builder, inputs);
}

TEST(Decomposer, elementwise_add_grad) {
  NetBuilder builder("elementwise_add_grad");
  auto dout = builder.CreateInput(Float(32), {10, 20});
  auto x    = builder.CreateInput(Float(32), {10, 20});
  auto y    = builder.CreateInput(Float(32), {10, 20});
  auto dx   = builder.elementwise_add_grad(dout, x, y);

  std::vector<std::string> inputs = {"Dout", "X", "Y"};
  RunProgram(builder, inputs);
}

}  // namespace cinn::frontend
