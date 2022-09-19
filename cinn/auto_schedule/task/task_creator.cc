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

#include <glog/logging.h>

#include <memory>
#include <tuple>
#include <vector>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::common::GraphEdge;
using ::cinn::common::GraphNode;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::Node;
using ::cinn::hlir::framework::NodeData;

std::vector<TuneTask> TaskCreator::CreateTuneTaskOpLevel(Graph* graph) {
  std::vector<TuneTask> ret_tasks;

  const std::vector<std::shared_ptr<Graph::Group>>& groups = graph->fusion_groups;

  // The input graph has run Op Fusion
  if (!groups.empty()) {
    for (const auto& sub_graph : groups) {
      ret_tasks.emplace_back(TuneTask());
      ret_tasks.back().task_graph.push_back(sub_graph);
      ret_tasks.back().target = graph->target_;
    }
    return ret_tasks;
  }

  // The input graph hasn't run Op Fusion
  std::tuple<std::vector<GraphNode*>, std::vector<GraphEdge*>> topo_result = graph->topological_order();
  const std::vector<GraphNode*>& nodes_in_order                            = std::get<0>(topo_result);
  for (auto graph_node : nodes_in_order) {
    // n must be an op node
    auto node = graph_node->safe_as<Node>();
    if (node) {
      auto group = std::make_shared<Graph::Group>();
      // init group
      group->nodes.push_back(node);
      group->nodes_set.insert(node);
      group->output_nodes.insert(node);
      // input node
      for (auto& edge : node->inlinks()) {
        auto input_graph_node = edge->source();
        auto input_node_data  = input_graph_node->safe_as<NodeData>();
        CHECK(input_node_data);
        // input data has no source node
        if (input_node_data->source_node.get()) {
          group->input_nodes[input_node_data->source_node.get()] = 1;
        }
      }

      // group type
      group->op_pattern_kind = hlir::framework::kOpaque;
      // use current node as master node for schedule
      group->master_nodes.insert(node);
      group->group_id = node->id();

      VLOG(6) << "Huihuang debug, group name = " << group->GetFuncName();

      graph->fusion_groups.push_back(group);
      ret_tasks.emplace_back(TuneTask());
      ret_tasks.back().task_graph.push_back(group);
      ret_tasks.back().target = graph->target_;
    }
  }

  return ret_tasks;
}

}  // namespace auto_schedule
}  // namespace cinn
