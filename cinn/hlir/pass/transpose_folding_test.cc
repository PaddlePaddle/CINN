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
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace frontend {

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

TEST(TransposeFolding, FoldIntoDot) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 3}, "X");
  auto y           = builder.CreateInput(Float(32), {2, 3}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out         = builder.Dot(x, transpose_y);
  auto program     = builder.Build();
  VLOG(1) << "Program:\n" << program;
  Target target = GetTarget();
  auto graph    = std::make_shared<hlir::framework::Graph>(program, target);
  VLOG(1) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "TransposeFolding");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");

  auto scope = hlir::framework::BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");

  auto X1 = scope->GetTensor("X");
  auto Y1 = scope->GetTensor("Y");
  SetRandData(X1, target);
  SetRandData(Y1, target);

  runtime_program->Execute();
}

TEST(TransposeFolding, FoldIntoConv) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 3, 1, 1}, "X");
  auto y           = builder.CreateInput(Float(32), {3, 2, 1, 1}, "Y");
  auto transpose_x = builder.Transpose(x, {1, 0, 2, 3});
  auto out         = builder.Conv(transpose_x, y);
  auto program     = builder.Build();

  Target target = GetTarget();
  VLOG(1) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  VLOG(1) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "TransposeFolding");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = hlir::framework::BuildScope(target, graph);

  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");

  auto X1 = scope->GetTensor("X");
  auto Y1 = scope->GetTensor("Y");
  SetRandData(X1, target);
  SetRandData(Y1, target);

  runtime_program->Execute();
}

}  // namespace frontend
}  // namespace cinn
