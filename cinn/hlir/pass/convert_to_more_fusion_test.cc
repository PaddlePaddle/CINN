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

#include "cinn/cinn.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace frontend {

TEST(convert_to_more_fusion, simple_convert) {
  Placeholder A(Float(32), {3, 3}, "A");
  Placeholder B(Float(32), {3, 3}, "B");

  Program program;
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  auto ewadd = program.elementwise_add(A, B);
  auto relu = program.relu(ewadd);
  auto reduce_sum = program.reduce_sum(ewadd, {0, 1});
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto get_ewadd_nodes = [](const common::GraphNode* graph_node) -> bool {
    auto op_node = graph_node->safe_as<hlir::framework::Node>();
    return op_node && "elementwise_add" == op_node->attrs.node_name;
  };
  // CollectNodes
  LOG(INFO) << "original graph:\n" << graph->Visualize();
  auto before_ewadd_nodes = graph->CollectNodes(get_ewadd_nodes);
  hlir::framework::ApplyPass(graph.get(), "ConvertToMoreFusion");
  auto after_ewadd_nodes = graph->CollectNodes(get_ewadd_nodes);
  ASSERT_EQ(before_ewadd_nodes.size(), after_ewadd_nodes.size() - 1);
  LOG(INFO) << "after pass graph:\n" << graph->Visualize();
  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");

  auto scope = hlir::framework::BuildScope(target, graph);
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  // check compiled program has 2 kernel
  ASSERT_EQ(runtime_program->size(), 2);
}

} // namespace frontend
} // namespace cinn
