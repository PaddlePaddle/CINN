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

#include "cinn/hlir/framework/graph.h"

#include <gtest/gtest.h>

#include "cinn/frontend/net_builder.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"

DECLARE_string(cinn_fusion_groups_graphviz_dir);

namespace cinn {
namespace hlir {
namespace framework {

TEST(Graph, visualize) {
  frontend::NetBuilder builder("test");
  auto x            = builder.CreateInput(Float(32), {32, 16}, "x");
  auto y            = builder.CreateInput(Float(32), {32, 16}, "y");
  auto add_1        = builder.ElementwiseAdd(x, y);
  auto relu_1       = builder.Relu(add_1);
  auto reduce_sum_1 = builder.ReduceSum(relu_1, {1});
  auto program      = builder.Build();

  auto target = common::DefaultHostTarget();
  auto graph  = std::make_shared<Graph>(program, target);
  ApplyPass(graph.get(), "OpFusion");

  FLAGS_cinn_fusion_groups_graphviz_dir = ".";
  graph->VisualizeGroupedGraph(graph->groups, {reduce_sum_1->id});
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
