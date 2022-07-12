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

#include "cinn/frontend/decomposer/test_helper.h"
#include "cinn/hlir/framework/graph_compiler.h"

namespace cinn {
namespace hlir {
namespace framework {

using namespace frontend;

TEST(FUSION_CHECKER_TEST, Test_01) {
  int h = 32, w = 32;
  NetBuilder net_builder("Test_01");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.ElementwiseAdd(A, B);

    auto D = net_builder.Reduce(C, ReduceKind::kSum, {0});
    auto E = net_builder.Reduce(C, ReduceKind::kSum, {0});
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto scope = std::make_shared<Scope>();
  GraphCompiler gc(target, scope, graph);

  gc.Build();
}
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
