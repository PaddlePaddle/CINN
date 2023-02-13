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

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

TEST(Decomposer, norm_decomposer) {
  //   int n = 16, c = 128, h = 14, w = 14;
  int32_t axis  = -1;
  float epsilon = 1e-12f;
  NetBuilder net_builder("norm_decomposer");
  std::unordered_set<std::string> output_names;
  {
    auto x = net_builder.CreateInput(Float(32), {10, 5, 20}, "x");
    auto y = net_builder.Norm(x, axis, epsilon);
    output_names.insert(y->id);
  }
  auto program = net_builder.Build();

  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, output_names, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

  auto scope = BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto run_program = gc.Build();

  std::vector<float> x(10 * 5 * 20);
  InitRandomVector<float>(&x, 10 * 5 * 20, 0.0f, 1.0f, 1e-3);
  std::vector<std::pair<std::string, std::vector<float>>> inputs = {{"x", x}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data  = tensor->mutable_data<float>(target);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();
}

}  // namespace cinn::frontend
