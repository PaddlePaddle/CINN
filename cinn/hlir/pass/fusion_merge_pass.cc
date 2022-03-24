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

#include "cinn/hlir/pass/fusion_helper_base.h"

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

using Comparator = Graph::Group::SharedGroupComparator;
using Hasher     = Graph::Group::SharedGroupHasher;

using Group  = std::shared_ptr<Graph::Group>;
using Groups = std::vector<Group>;

using ShapeDict         = absl::flat_hash_map<std::string, shape_t>;
using ConditionFunction = std::function<bool(const Group&, const Group&)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class FusionMergePassHelper : public FusionHelperBase {
 public:
  OpFusionPassHelper(Graph* graph)
      : FusionHelperBase(graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape"), graph->target_) {
    fusion_groups_ = graph->fusion_groups;
  }

  Groups operator()() {
    // run fusion merge untill no update.
    while (DoFusionMerge()) {
    }
    return fusion_groups_;
  }

 private:
  void DoFusionMerge() {
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      // if producer is sub group.
      if (!producer->belong_groups.size()) {
        continue;
      }

      updated = DoFusionMergeHorizontal(producer->consumer_groups);
      updated = DoFusionMergeVertical(producer.get(), producer->consumer_groups);
    }
    return updated;
  }

  bool DoFusionMergeHorizontal(std::unordered_set<Group, Hasher, Comparator>& consumers) {
    std::vector<Group> candidate_consumers;
    // check consumers exist depency relation
    for (auto& consumer : consumers) {
      if (!IsDepency(*consumer, consumers)) {
        candidate_consumers.push_back(*consumer);
      }
    }

    std::vector<Groups> fusionable_consumers;
    // fuse consumer groups
    for (auto consumer : candidate_consumers) {
      // if fusionable consumers is not exist
      if (!fusionable_consumers.size()) {
        fusionable_consumers.emplace_back(consumer);
        continue;
      }

      // relation
      auto& relation = fusion_relation_map_[group->op_pattern_kind];
      // check horizontal relation exist
      if (!relation.horizontal_relation.size()) {
        fusionable_consumers.emplace_back(consumer);
        continue;
      }

      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!relation.horizontal_relation.count(last->op_pattern_kind)) {
          continue;
        }

        if (!relation.horizontal_relation[last->op_pattern_kind](group, last)) {
          continue;
        }

        groups.push_back(group);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.emplace_back(group);
      }
    }

    bool updated = false;
    for (auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        updated = true;
        DoHorizontalFuse(groups);
      }
    }

    return updated;
  }

  bool DoFusionMergeVertical(Group& producer, std::unordered_set<Group, Hasher, Comparator>& consumers) {
    auto& relation = fusion_relation_map_[producer->op_pattern_kind];
    // if producer can't fuse others
    if (!relation.vertical_relation.size()) {
      return false;
    }

    Groups fusionable_consumers;
    for (auto& consumer : consumers) {
      // if can't fuse
      if (relation.vertical_relation.count(*consumer->op_pattern_kind)) {
        continue;
      }

      // if condition function is false
      if (!relation.vertical_relation[consumer->op_pattern_kind](producer, *consumer)) {
        continue;
      }

      fusionable_consumers.push_back(*consumer);
    }

    // if fusionable consumers exist
    if (fusionable_consumers.size()) {
      DoVerticalFuse(producer, fusionable_consumers);
      return true;
    }

    return false;
  }

  void DoHorizontalFuse(Groups& groups) {
    auto fusion_group = std::make_shared<Graph::Group>();
    for (auto group : groups) {
      if (group->fused_sub_groups.size()) {
      }
    }
  }

  void DoVerticalFuse(Group& producer, Groups& consumers) {}

  float Cost(const Group& group) { return 0.0f; }

  bool IsDepency(const Group& group, const std::unordered_set<Group, Hasher, Comparator>& target_set) {
    std::queue<const Group> candidates;
    candidates.push_back(group);

    std::unordered_set<const Group> visited_set;
    while (!candidates.empty()) {
      auto candidate = candidates.front();
      candidates.pop();

      for (auto& producer : candidate->producer_groups) {
        if (target_set.count(*producer)) {
          return true;
        }
        if (!visited_set.count(*producer)) {
          visited_set.insert(*producer);
        }
      }
    }

    return false;
  }

  void InitFusionRelation() {
    // fuse condition function
    auto always_fuse   = [this](const Group& first, const Group& second) -> bool { return true; };
    auto is_same_shape = [this](const Group& first, const Group& second) -> bool {
      auto output_var_0 = this->GetNodeDataShape(*first->output_nodes.begin());
      auto output_var_1 = this->GetNodeDataShape(*second->output_nodes.begin());
      return output_var_0 == output_var_1;
    };
    auto reduce_fuse_reduce = [this](const Group& first, const Group& second) -> bool {
      Node* reducer_0 = nullptr;
      for (auto& reducer : first->master_nodes) {
        if (GetOpKind(reducer) == OpPatternKind::kCommReduce) {
          reducer_0 = reducer;
          break;
        }
      }

      Node* reducer_1 = nullptr;
      for (auto& reducer : second->master_nodes) {
        if (GetOpKind(reducer) == OpPatternKind::kCommReduce) {
          reducer_1 = reducer;
          break;
        }
      }

      // check reduce has same input shape and output shape
      auto reducer_0_input_shape  = shape_dict_.at(reducer_0->inlinks_in_order()[0]->source()->id());
      auto reducer_0_output_shape = shape_dict_.at(reducer_0->outlinks_in_order()[0]->sink()->id());

      auto reducer_1_input_shape  = shape_dict_.at(reducer_1->inlinks_in_order()[0]->source()->id());
      auto reducer_1_output_shape = shape_dict_.at(reducer_1->outlinks_in_order()[0]->sink()->id());

      auto reducer_0_reduce_dim = absl::get<std::vector<int>>(reducer_0->attrs.attr_store.at("dim"));
      auto reducer_1_reduce_dim = absl::get<std::vector<int>>(reducer_1->attrs.attr_store.at("dim"));

      for (auto& dim : reducer_0_reduce_dim) {
        // if dim = -1, set as shape.size() - 1
        if (dim == -1) {
          dim = producer_input_shape.size() - 1;
        }
      }

      for (auto& dim : reducer_1_reduce_dim) {
        // if dim = -1,  set as shape.size() - 1
        if (dim == -1) {
          dim = reducer_input_shape.size() - 1;
        }
      }
      // check shape is same
      if (reducer_0_input_shape != reducer_1_input_shape || reducer_0_output_shape != reducer_1_output_shape ||
          reducer_0_reduce_dim != reducer_1_reduce_dim ||
          absl::get<std::vector<int>>(reducer_0->attrs.attr_store.at("keep_dim")) !=
              absl::get<std::vector<int>>(reducer_1->attrs.attr_store.at("keep_dim"))) {
        return false;
      }
      return true;
    };

    auto elementwise_fuse_broadcast = [this](const Group& first, const Group& second) -> bool { return true; };

    auto broadcast_fuse_reduce = [this](const Group& first, const Group& second) -> bool { return true; };

    auto injective_fuse_reduce = [this](const Group& first, const Group& second) -> bool { return true; };

    auto reduce_fuse_elementwise = [this](const Group& first, const Group& second) -> bool { return true; };

    // kElemWise
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kElemWise];
      // horizontal
      relation.horizontal_relation = {{framework::kElemWise, is_same_shape}};
      // vertical
      relation.vertical_relation = {// one-by-one(only one can fuse), fuse
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kElementwise, is_same_shape},
                                    // one-by-one(only one can fuse), need check as may can't use with buffer.
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kBroadcast, elementwise_fuse_broadcast},
                                    // one-by-one(only one can fuse), fuse
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kCommReduce, always_fuse}};
    }
    // kBroadcast
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kBroacast];
      // horizontal
      relation.horizontal_relation = {{framework::kBroadcast, is_same_shape}};
      // vertical
      relation.vertical_relation = {// one-by-one(only one can fuse), fuse
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kElementwise, is_same_shape},
                                    // one-by-one(only one can fuse), need check as may can't use with buffer.
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kCommReduce, broadcast_fuse_reduce}};
    }
    // kInjective
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kInjective];
      // horizontal
      // relation.horizontal_relation = {{OpPatternKind::kInjective, is_same_shape}};
      // vertical
      relation.vertical_relation = {// one-by-one(only one can fuse), fuse
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kElementwise, is_same_shape},
                                    // one-by-one(only one can fuse), need check as may can't use with buffer.
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kCommReduce, injective_fuse_reduce}};
    }
    // kCommReduce
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kCommReduce];
      // horizontal
      relation.horizontal_relation = {{OpPatternKind::kCommReduce, reduce_fuse_reduce}};
      // vertical
      relation.vertical_relation = {// one-by-one(only one can fuse), cost model
                                    // one-by-multi, re-compute + cost model
                                    {OpPatternKind::kElementwise, reduce_fuse_elementwise}};
    }
  }

  Groups fusion_groups_;

  struct Relation {
    std::unordered_map<framework::OpPatternKind, ConditionFunction> vertical_relation;
    std::unordered_map<framework::OpPatternKind, ConditionFunction> horizontal_relation;
  };
  std::unordered_map<framework::OpPatternKind, Relation> fusion_relation_map_;
};  // namespace hlir

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
