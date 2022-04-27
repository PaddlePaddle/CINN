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
  std::uniform_real_distribution<float> dist(0.f, 1.f);
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

TEST(RemoveIdentity, can_remove) {
  NetBuilder builder("net_builder");
  auto x          = builder.CreateInput(Float(32), {32, 16});
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(x);
  auto relu_1     = builder.ReduceSum(x, {0, 1});
  auto relu_2     = builder.ReduceSum(x, {0, 1});
  auto program    = builder.Build();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  LOG(INFO) << program;
  size_t before_size = program.size();
  ProgramPass::Apply(&program, {}, target, {"RemoveIdentity"});
  size_t after_size = program.size();
  ASSERT_EQ(before_size, after_size + 2);
  LOG(INFO) << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = hlir::framework::BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("X");

  auto X = scope->GetTensor("X");
  SetRandData(X, target);

  runtime_program->Execute();
}

TEST(RemoveIdentity, cant_remove_by_fetchids) {
  NetBuilder builder("net_builder");
  auto x          = builder.CreateInput(Float(32), {32, 16});
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(x);
  auto relu_1     = builder.ReduceSum(x, {0, 1});
  auto relu_2     = builder.ReduceSum(x, {0, 1});
  auto program    = builder.Build();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  LOG(INFO) << program;
  size_t before_size = program.size();
  ProgramPass::Apply(&program, {identity_1->id}, target, {"RemoveIdentity"});
  size_t after_size = program.size();
  ASSERT_EQ(before_size, after_size + 1);
  LOG(INFO) << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = hlir::framework::BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("X");

  auto X = scope->GetTensor("X");
  SetRandData(X, target);

  runtime_program->Execute();
}

TEST(RemoveIdentity, cant_remove_by_pattern) {
  NetBuilder builder("net_builder");
  auto x          = builder.CreateInput(Float(32), {32, 16});
  auto identity_1 = builder.Identity(x);
  auto identity_2 = builder.Identity(x);
  auto mul        = builder.Mul(x, identity_1);
  auto program    = builder.Build();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  LOG(INFO) << program;
  size_t before_size = program.size();
  ProgramPass::Apply(&program, {}, target, {"RemoveIdentity"});
  size_t after_size = program.size();
  ASSERT_EQ(before_size, after_size + 1);
  LOG(INFO) << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = hlir::framework::BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("X");

  auto X = scope->GetTensor("X");
  SetRandData(X, target);

  runtime_program->Execute();
}

}  // namespace cinn::frontend
