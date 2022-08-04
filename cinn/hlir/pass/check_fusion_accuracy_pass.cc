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

#include <queue>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/pass/check_fusion_accuracy_pass_util.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::NodePtr;

using common::GraphEdge;
using common::GraphNode;

using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

namespace {
constexpr char check_node_suffix[] = "_acc_check";
}

std::vector<NodeData*> CreateNodeInputs(const std::unordered_map<Node*, int> group_inputs, Graph* graph, Node* node) {
  const auto& inlinks = node->inlinks_in_order();
  for (const auto& in_edge : inlinks) {
    auto in_node = in_edge->source();
  }
}

GroupPtr CreateCheckGroup(Graph* graph, Node* node) {
  CHECK_NOTNULL(node->op()) << "Node " << node->id() << " is not operator! Please check.";

  auto check_node =
      Node::Create(node->op(), node->attrs.node_name, utils::GenerateCheckFusionAccuracyNodeId(node->id()));
}

void CheckFusionAccuracyPassImpl(Graph* graph) {
  GroupList check_fusion_group;

  for (auto& group : graph->fusion_groups) {
    check_fusion_group.emplace_back(group);
    // fusion group only has one node, do not need check, skip
    if (group->nodes_set.size() <= 1) continue;

    std::unordered_set<Node*> check_group_outputs;
    for (auto* node : TopologicalOrder(group->nodes)) {
      if (node->is_variable()) continue;
    }
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(CheckFusionAccuracyPass) {
  CINN_REGISTER_PASS(CheckFusionAccuracyPass)
      .describe("Check Fusion Accuracy Pass.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::CheckFusionAccuracyPassImpl);

  return true;
}
