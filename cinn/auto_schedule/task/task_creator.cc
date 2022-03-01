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
#include <vector>

#include "cinn/common/graph.h"
#include "cinn/hlir/framework/graph.h"

namespace cinn {
namespace auto_schedule {

using common::GraphEdge;
using common::GraphNode;

std::vector<TuneTask> TaskCreator::CreateTuneTaskOpLevel(hlir::framework::Graph* graph) {
  std::vector<TuneTask> ret_tasks;
  std::tuple<std::vector<GraphNode*>, std::vector<GraphEdge*>> topo_result = graph->topological_order();

  const std::vector<GraphNode*>& nodes = topo_result.get<0>();

  for (const GraphNode* n : nodes) {
    const Node* op_node = node->safe_as<Node>();
    if (op_node) {
      // n must be an op node
      ret_task.emplace_back(TuneTask());
      ret_task.back().SubGraph().push_back(op_node);
    }
  }

  return ret_tasks;
}

}  // namespace auto_schedule
}  // namespace cinn
