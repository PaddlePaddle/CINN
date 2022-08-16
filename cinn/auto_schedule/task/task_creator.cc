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

#include "cinn/auto_schedule/task/task_creator.h"

#include <memory>
#include <tuple>
#include <vector>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::common::GraphEdge;
using ::cinn::common::GraphNode;
using ::cinn::hlir::framework::Node;

std::vector<TuneTask> TaskCreator::CreateTuneTaskOpLevel(hlir::framework::Graph* graph) {
  std::vector<TuneTask> ret_tasks;

  const std::vector<std::vector<Node*>>& groups = graph->groups;

  // The input graph has run Op Fusion
  if (!groups.empty()) {
    for (const std::vector<Node*>& sub_graph : groups) {
      ret_tasks.emplace_back(TuneTask());
      ret_tasks.back().task_graph.push_back(sub_graph);
      ret_tasks.back().target = graph->target_;
    }
    return ret_tasks;
  }

  // The input graph hasn't run Op Fusion
  std::tuple<std::vector<GraphNode*>, std::vector<GraphEdge*>> topo_result = graph->topological_order();
  const std::vector<GraphNode*>& nodes                                     = std::get<0>(topo_result);
  for (GraphNode* n : nodes) {
    Node* op_node = n->safe_as<Node>();
    if (op_node) {
      // n must be an op node
      ret_tasks.emplace_back(TuneTask());
      ret_tasks.back().task_graph.push_back(std::vector<Node*>(1, op_node));
      ret_tasks.back().target = graph->target_;
    }
  }

  return ret_tasks;
}

}  // namespace auto_schedule
}  // namespace cinn
