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

#include <random>
#include <unordered_set>

#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "gtest/gtest.h"

namespace cinn::frontend {

namespace {

bool IsCompiledWithCUDA() {
#if !defined(CINN_WITH_CUDA)
  return false;
#else
  return true;
#endif
}

void PrintMatrix(const std::vector<float>& mat, int m, int n) {
  std::cout << "-----------------------------------------------------\n";
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << mat[i * n + j] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "-----------------------------------------------------\n\n";
}

void SetRandData(hlir::framework::Tensor tensor, Target target, int seed = -1) {
  if (seed == -1) {
    std::random_device rd;
    seed = rd();
  }
  std::default_random_engine engine(seed);
  std::uniform_int_distribution<int> dist(1, 10);
  size_t num_ele = tensor->shape().numel();
  std::vector<float> random_data(num_ele);
  for (size_t i = 0; i < num_ele; i++) {
    random_data[i] = static_cast<float>(dist(engine));  // All random data
  }
  PrintMatrix(random_data, tensor->shape().data()[0], tensor->shape().data()[1]);

  auto* data = tensor->mutable_data<float>(target);
#ifdef CINN_WITH_CUDA
  cudaMemcpy(data, random_data.data(), num_ele * sizeof(float), cudaMemcpyHostToDevice);
#else
  std::copy(random_data.begin(), random_data.end(), data);
#endif
}

std::vector<float> GetTensorData(const hlir::framework::Tensor& tensor, Target target) {
  auto size = tensor->shape().numel();
  std::vector<float> data(size);
#ifdef CINN_WITH_CUDA
  cudaMemcpy(
      data.data(), static_cast<const void*>(tensor->data<float>()), size * sizeof(float), cudaMemcpyDeviceToHost);
#else
  std::copy(tensor->data<float>(), tensor->data<float>() + size, data.begin());
#endif
  return data;
}

void RunGraph(std::shared_ptr<hlir::framework::Graph> graph,
              const Target& target,
              const std::shared_ptr<hlir::framework::Scope>& scope,
              std::unordered_set<std::string>&& fetch_ids) {
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  LOG(INFO) << "Graph Viz:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

std::vector<float> RunProgram(const Program& program,
                              const Target& target,
                              std::vector<cinn::frontend::Placeholder> inputs,
                              std::vector<std::string> output_ids,
                              int seed = -1) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = hlir::framework::BuildScope(target, graph);
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(std::string(input.id()));
    SetRandData(scope->GetTensor(std::string(input.id())), target, seed);
  }
  std::unordered_set<std::string> fetch_ids(output_ids.begin(), output_ids.end());
  RunGraph(graph, target, scope, std::move(fetch_ids));
  return GetTensorData(scope->GetTensor(output_ids.front()), target);
}

}  // namespace

TEST(GemmRwriter, Basic) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {6, 8}, "A");
  auto b       = builder.Transpose(a, {1, 0});
  auto c       = builder.CreateInput(Float(32), {7, 6}, "C");
  auto d       = builder.Transpose(c, {1, 0});
  auto e       = builder.Matmul(b, d);
  auto f       = builder.FillConstant<float>({8, 7}, 2.0f, "F");
  auto out     = builder.Add(e, f);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::unordered_set<std::string> fetch_ids{out->id};
  // apply common pass
  ProgramPass::Apply(&program, fetch_ids, target, {"Decomposer"});
  ApplyPass(&program, fetch_ids, "RemoveIdentity");

  // get origin output
  auto origin_out = RunProgram(program, target, {a, c}, {out->id}, 123);
  PrintMatrix(origin_out, 8, 7);

  // fuse transpose + add + dot, then run and get the fused output
  ApplyPass(&program, fetch_ids, "TransposeFolding");
  ProgramPass::Apply(&program, fetch_ids, target, {"GemmRewriter"});
  auto fused_out = RunProgram(program, target, {a, c}, {out->id}, 123);
  PrintMatrix(fused_out, 8, 7);
}

// TEST(GemmRwriter, Complex) {
//   if (!IsCompiledWithCUDA()) {
//     return;
//   }
//   NetBuilder builder("net_builder");
//   auto a       = builder.FillConstant<float>({2, 20}, 2.0f, "A");
//   auto b       = builder.Transpose(a, {1, 0});
//   auto c       = builder.CreateInput(Float(32), {121, 20}, "C");
//   auto d       = builder.Matmul(c, b);
//   auto x       = builder.FillConstant<float>({2, 20}, 1.0f, "X");
//   auto y       = builder.Transpose(x, {1, 0});
//   auto z       = builder.CreateInput(Float(32), {20, 121}, "Z");
//   auto l       = builder.Transpose(z, {1, 0});
//   auto q       = builder.Matmul(l, y);
//   auto p       = builder.Mul(c, a);
//   auto m       = builder.Sub(d, p);
//   auto n       = builder.Add(d, q);
//   auto out     = builder.Add(m, n);
//   auto program = builder.Build();

//   Target target = common::DefaultNVGPUTarget();
//   std::unordered_set<std::string> fetch_ids{out->id};
//   // apply common pass
//   ProgramPass::Apply(&program, fetch_ids, target, {"Decomposer"});
//   ApplyPass(&program, fetch_ids, "RemoveIdentity");

//   // get origin output
//   auto origin_out = RunProgram(program, target, {c, z}, {out->id});

//   // fuse transpose + add + dot, then run and get the fused output
//   ApplyPass(&program, fetch_ids, "TransposeFolding");
//   ProgramPass::Apply(&program, fetch_ids, target, {"GemmRewriter"});
//   auto fused_out = RunProgram(program, target, {c, z}, {out->id});
// }

}  // namespace cinn::frontend
