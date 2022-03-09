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

#include "cinn/cinn.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
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

void RunWithProgram(const Program& program,
                    const Target& target,
                    const std::shared_ptr<hlir::framework::Scope>& scope) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(), {"InferShape", "OpFusion"});
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(TransposeFolding, FoldIntoDotCase1) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 3}, "X");
  auto y           = builder.CreateInput(Float(32), {2, 3}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out         = builder.Dot(x, transpose_y);
  auto program     = builder.Build();
  auto target      = GetTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData(scope->GetTensor("X"), target);
  SetRandData(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData(scope->GetTensor(out->id), target);
  ApplyPass(&program, {}, "TransposeFolding");
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldIntoDotCase2) {
  NetBuilder builder("net_builder");
  auto a             = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b             = builder.Transpose(a, {1, 0});
  auto c             = builder.CreateInput(Float(32), {121, 20}, "C");
  auto d             = builder.Matmul(c, b);
  auto x             = builder.FillConstant<float>({2, 20}, 1.0f, "X");
  auto y             = builder.Transpose(x, {1, 0});
  auto z             = builder.CreateInput(Float(32), {121, 20}, "Z");
  auto q             = builder.Matmul(z, y);
  auto program       = builder.Build();
  auto target        = GetTarget();
  auto graph         = std::make_shared<hlir::framework::Graph>(program, target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, hlir::framework::BuildScope(target, graph));
  ApplyPass(&program, {}, "TransposeFolding");
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, hlir::framework::BuildScope(target, graph));
  ASSERT_EQ(origin_size, folded_size + 2);
}

}  // namespace cinn::frontend
