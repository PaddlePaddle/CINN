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

#include <algorithm>
#include <queue>
#include <unordered_set>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::OpPatternKind;
using framework::shape_t;

using common::GraphEdge;
using common::GraphNode;

using Group  = std::shared_ptr<Graph::Group>;
using Groups = std::vector<Group>;

using ShapeDict         = absl::flat_hash_map<std::string, shape_t>;
using ConditionFunction = std::function<bool(const Node*, const Node*)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class OpFusionPassHelper {
 public:
  OpFusionPassHelper(Graph* graph) {
    target_          = graph->target_;
    shape_dict_      = graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
    op_pattern_dict_ = &framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    fusion_groups_   = graph->fusion_groups;
  }

  Groups operator()() {
    // run fusion merge untill no update.
    while (DoFusionMerge()) {
    }
    return fusion_groups_;
  }

 private:
  void DoFusionMerge() {
    bool updated    = false;
    auto tmp_groups = fusion_groups_;
    for (int idx = 0; idx < tmp_groups.size(); ++idx) {
      auto group    = tmp_groups[idx];
      auto consumer = ;
    }

    return updated;
  }

  Group Fuse(Groups& groups) { return groups[0]; }

  float Cost(Group& group) { return 0.0f; }

  bool IsDepency(const Group& group, const Groups& groups) {
    std::queue<const Graph::Group*> candidates;
    candidates.push_back(group.get());

    std::unordered_set<const Graph::Group*> target_set;
    std::unordered_set<const Graph::Group*> visited_set;
    for (auto& element : groups) {
      target_set.insert(element.get());
    }

    while (!candidates.empty()) {
      auto group_ptr = candidates.front();
      candidates.pop();

      for (auto producer : group_ptr->producer_groups) {
        if (target_set.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
        }
      }
    }

    return false;
  }
  // target
  common::Target target_;
  // fusion groups
  Groups fusion_groups_;
  // shape dict
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;
};

void FusionMergePassInternal(Graph* graph) {}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(FusionMergePass) {
  CINN_REGISTER_PASS(FusionMergePass)
      .describe(
          "Fusion Merge Pass which performs Fusion-Ops fusion, Producer Fusion-Ops are fused into Consumer Fusion-Ops "
          "with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::FusionMergePassInternal);

  return true;
}
