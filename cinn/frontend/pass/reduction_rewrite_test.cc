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

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

void SetRandData(const hlir::framework::Tensor& tensor, Target target) {
#ifdef CINN_WITH_CUDA
  auto* data = tensor->mutable_data<float>(target);
  std::vector<float> host_memory(tensor->shape().numel(), 0);
  for (float& v : host_memory) {
    v = (rand() * 1.f) / RAND_MAX;  // All random data
  }
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data),
                       host_memory.data(),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyHostToDevice));
#else
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    data[j] = (rand() * 1.f) / RAND_MAX;  // All random data
  }
#endif
}

std::vector<float> GetTensorData(const hlir::framework::Tensor& tensor, Target target) {
  std::vector<float> data;
#ifdef CINN_WITH_CUDA
  data.resize(tensor->shape().numel());
  CUDA_CALL(cudaMemcpy(data.data(),
                       reinterpret_cast<void*>(tensor->mutable_data<float>(target)),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyDeviceToHost));
#else
  for (size_t i = 0; i < tensor->shape().numel(); ++i) {
    data.push_back(tensor->data<float>()[i]);
  }
#endif
  return data;
}

void RunWithGraph(const std::shared_ptr<hlir::framework::Graph>& graph,
                  const Target& target,
                  const std::shared_ptr<hlir::framework::Scope>& scope) {
  hlir::framework::ApplyPasses(graph.get(), {"InferShape", "OpFusion"});
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(ReductionRewrite, nochange) {
  Target target = GetTarget();
  Program program;
  Placeholder X(Float(32), {256, 256, 256}, "X");
  auto out = program.reduce_min(X, {});
  program.SetInputs({X});
  size_t before_size = program.size();
  ProgramPass::Apply(&program, target, {"ReductionRewrite"});
  size_t after_size = program.size();
  ASSERT_EQ(before_size, after_size);
}

TEST(ReductionRewrite, basic) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {256, 256, 256}, "X");
  auto out     = builder.Reduce(x, ReduceKind::kSum, {0, 2});
  auto program = builder.Build();
  auto target  = GetTarget();
  // before
  auto before_graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto before_scope = hlir::framework::BuildScope(target, before_graph);
  before_scope->Var<hlir::framework::Tensor>("X");
  SetRandData(before_scope->GetTensor("X"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Before " << program;
  RunWithGraph(before_graph, target, before_scope);
  auto origin_out = GetTensorData(before_scope->GetTensor(out->id), target);
  // after
  ProgramPass::Apply(&program, target, {"ReductionRewrite"});
  size_t rewrite_size = program.size();
  VLOG(1) << "After " << program;
  auto after_graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(after_graph.get(), "InferShape");
  auto after_scope = hlir::framework::BuildScope(target, after_graph);
  after_scope->Var<hlir::framework::Tensor>("X");
  after_scope->GetTensor("X")->set_buffer(before_scope->GetTensor("X")->get_buffer());
  RunWithGraph(after_graph, target, after_scope);
  auto rewrite_out = GetTensorData(after_scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, rewrite_size - 1);
  ASSERT_EQ(origin_out.size(), rewrite_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_LT(std::abs((rewrite_out[i] - origin_out[i]) / rewrite_out[i]), 0.0001);
  }
}

}  // namespace cinn::frontend
