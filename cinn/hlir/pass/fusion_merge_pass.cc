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

#include <queue>

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
  FusionMergePassHelper(Graph* graph)
      : FusionHelperBase(graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape"), graph->target_) {
    fusion_groups_ = graph->fusion_groups;
    InitFusionRelation();
  }

  Groups operator()() {
    // run fusion merge untill no update.
    while (DoFusionMerge()) {
    }
    return fusion_groups_;
  }

 private:
  bool DoFusionMerge() {
    VLOG(11) << "DoFusionMerge...!";
    bool updated = false;
    for (int idx = 0; idx < fusion_groups_.size(); ++idx) {
      auto producer = fusion_groups_[idx];
      VLOG(11) << "Fusion Producer Group -> " << producer->group_id;
      // if producer is sub group.
      if (producer->belong_groups.size()) {
        continue;
      }

      updated |= DoFusionMergeHorizontal(producer->consumer_groups);
      updated |= DoFusionMergeVertical(producer, producer->consumer_groups);
    }

    Groups fusion_groups;
    // update fusion_groups_
    for (auto& group : fusion_groups_) {
      if (!group->belong_groups.size()) {
        VLOG(11) << "Fusion Group -> " << group->group_id;
        for (auto& sub_group : group->fused_sub_groups) {
          VLOG(11) << "  Fused Sub-Group -> " << sub_group->group_id;
        }
        fusion_groups.push_back(group);
      }
    }
    fusion_groups_ = fusion_groups;
    return updated;
  }

  bool DoFusionMergeHorizontal(std::unordered_set<Group, Hasher, Comparator>& consumers) {
    Groups candidate_consumers;
    // check consumers exist depency relation
    for (auto& consumer : consumers) {
      if (!IsDepency(consumer, consumers)) {
        candidate_consumers.push_back(consumer);
      }
    }

    std::vector<Groups> fusionable_consumers;
    // fuse consumer groups
    for (auto& consumer : candidate_consumers) {
      // if fusionable consumers is not exist
      if (!fusionable_consumers.size()) {
        fusionable_consumers.push_back({consumer});
        continue;
      }

      // relation
      auto& relation = fusion_relation_map_[consumer->op_pattern_kind];
      // check horizontal relation exist
      if (!relation.horizontal_relation.size()) {
        fusionable_consumers.push_back({consumer});
        continue;
      }

      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!relation.horizontal_relation.count(last->op_pattern_kind)) {
          continue;
        }

        if (!relation.horizontal_relation[last->op_pattern_kind](consumer, last)) {
          continue;
        }

        groups.push_back(consumer);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({consumer});
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

  void DoHorizontalFuse(Groups& consumers) {
    // create fusion group
    auto fused_group = std::make_shared<Graph::Group>();
    // fuse all group into fusion group.
    for (auto consumer : consumers) {
      VLOG(11) << "fuse consumer " << consumer->group_id << " into fused_group!";
      if (fused_group->group_id.size()) {
        fused_group->group_id += "_" + consumer->group_id;
      } else {
        fused_group->group_id = consumer->group_id;
      }
      // set op pattern kind
      fused_group->op_pattern_kind = consumer->op_pattern_kind;
      // input nodes
      for (auto& node : consumer->input_nodes) {
        fused_group->input_nodes.insert(node);
      }
      // output node
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }
      // master node
      for (auto& node : consumer->master_nodes) {
        if (GetOpKind(node) == framework::kCommReduce) {
          fused_group->master_nodes.insert(node);
        }
      }
      // insert sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          fused_group->fused_sub_groups.push_back(sub_group);
          // update belongs group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      // producer group
      for (auto& producer : consumer->producer_groups) {
        fused_group->producer_groups.insert(producer);
        // update producer's consumer
        producer->consumer_groups.erase(consumer);
        producer->consumer_groups.insert(fused_group);
      }
      // consumer group
      for (auto& gconsumer : consumer->consumer_groups) {
        fused_group->consumer_groups.insert(gconsumer);
        // update consumer's producer
        gconsumer->producer_groups.erase(consumer);
        gconsumer->producer_groups.insert(fused_group);
      }
      // belongs group
      consumer->belong_groups.insert(fused_group);
    }

    // Using last group as main group.
    auto& consumer = consumers.back();
    for (auto& node : consumer->master_nodes) {
      fused_group->master_nodes.insert(node);
    }
    // push group to back.
    fusion_groups_.push_back(fused_group);
  }

  bool DoFusionMergeVertical(Group& producer, std::unordered_set<Group, Hasher, Comparator>& consumers) {
    auto& relation = fusion_relation_map_[producer->op_pattern_kind];
    // if producer can't fuse others
    if (!relation.vertical_relation.size()) {
      return false;
    }

    std::unordered_set<Group, Hasher, Comparator> fusionable_consumers;
    for (auto& consumer : consumers) {
      // if can't fuse
      if (!relation.vertical_relation.count(consumer->op_pattern_kind)) {
        VLOG(11) << "Can't fuse producer " << producer->op_pattern_kind << " consumer " << consumer->op_pattern_kind;
        continue;
      }

      // if condition function is false
      if (!relation.vertical_relation[consumer->op_pattern_kind](producer, consumer)) {
        VLOG(11) << "Can't fuse producer " << producer->op_pattern_kind << " consumer " << consumer->op_pattern_kind;
        continue;
      }

      fusionable_consumers.insert(consumer);
    }

    if (fusionable_consumers.size() > 1) {
      RecomputeWithCostModel(producer, fusionable_consumers);
    }

    // if fusionable consumers exist
    if (fusionable_consumers.size()) {
      DoVerticalFuse(producer, fusionable_consumers);
      return true;
    }

    return false;
  }

  void DoVerticalFuse(Group& producer, std::unordered_set<Group, Hasher, Comparator>& fusionable_consumers) {
    Groups fused_groups;
    for (auto& consumer : fusionable_consumers) {
      auto fused_group = std::make_shared<Graph::Group>();
      // update group id
      fused_group->group_id = producer->group_id;
      VLOG(11) << "fuse producer " << producer->group_id << " into consumer " << consumer->group_id;
      // fuse producer into fusion group
      fused_group->op_pattern_kind = producer->op_pattern_kind;
      // input nodes
      fused_group->input_nodes = producer->input_nodes;
      // master nodes
      for (auto& node : producer->master_nodes) {
        if (GetOpKind(node) == framework::kCommReduce) {
          fused_group->master_nodes.insert(node);
        }
      }

      // producer groups
      for (auto& group : producer->producer_groups) {
        fused_group->producer_groups.insert(group);
        // update producer's producer's consumer
        group->consumer_groups.erase(producer);
        group->consumer_groups.insert(fused_group);
      }

      // sub groups
      if (producer->fused_sub_groups.size()) {
        for (auto& group : producer->fused_sub_groups) {
          fused_group->fused_sub_groups.push_back(group);
          // update belong group
          group->belong_groups.erase(producer);
          group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(producer);
      }
      producer->belong_groups.insert(fused_group);

      // fuse consumer into fusion group
      fused_group->op_pattern_kind =
          static_cast<int>(fused_group->op_pattern_kind) > static_cast<int>(consumer->op_pattern_kind)
              ? fused_group->op_pattern_kind
              : consumer->op_pattern_kind;

      fused_group->group_id += "_" + consumer->group_id;
      // input nodes
      for (auto& node : consumer->input_nodes) {
        if (!producer->output_nodes.count(node)) {
          fused_group->input_nodes.insert(node);
        }
      }

      // output nodes
      for (auto& node : consumer->output_nodes) {
        fused_group->output_nodes.insert(node);
      }

      // master nodes
      for (auto& node : consumer->master_nodes) {
        if (GetOpKind(node) == framework::kCommReduce) {
          fused_group->master_nodes.insert(node);
        }
      }

      // producer nodes
      for (auto& group : consumer->producer_groups) {
        if (group.get() != producer.get()) {
          fused_group->producer_groups.insert(group);
          // update consumer's producer's consumer
          group->consumer_groups.erase(consumer);
          group->consumer_groups.insert(fused_group);
        }
      }
      // consumer nodes
      for (auto& group : consumer->consumer_groups) {
        fused_group->consumer_groups.insert(group);
        // update consumer's consumer's producer
        group->producer_groups.erase(consumer);
        group->producer_groups.insert(fused_group);
      }

      // sub group
      if (consumer->fused_sub_groups.size()) {
        for (auto& sub_group : consumer->fused_sub_groups) {
          fused_group->fused_sub_groups.push_back(sub_group);
          // update belong group
          sub_group->belong_groups.erase(consumer);
          sub_group->belong_groups.insert(fused_group);
        }
      } else {
        fused_group->fused_sub_groups.push_back(consumer);
      }
      consumer->belong_groups.insert(fused_group);

      fused_groups.push_back(fused_group);
      fusion_groups_.push_back(fused_group);
    }

    // update output nodes
    if (fused_groups.size()) {
      auto& fused_group = fused_groups.front();
      // update output for others consumer
      for (auto& node : producer->output_nodes) {
        bool be_output = true;
        for (auto& consumer : fusionable_consumers) {
          if (consumer->input_nodes.count(node)) {
            be_output = false;
            break;
          }
        }

        if (be_output) {
          fused_group->output_nodes.insert(node);
        }
      }

      // consumer groups
      for (auto& consumer : producer->consumer_groups) {
        // if consumer is not fusinable.
        if (!fusionable_consumers.count(consumer)) {
          fused_group->consumer_groups.insert(consumer);
          // update consumer's producer
          consumer->producer_groups.erase(producer);
          consumer->producer_groups.insert(fused_group);
        }
      }
    }
  }

  void RecomputeWithCostModel(const Group& producer,
                              std::unordered_set<Group, Hasher, Comparator>& fusionable_consumers) {
    if (producer->op_pattern_kind == framework::kCommReduce) {
      auto consumer = *fusionable_consumers.begin();
      fusionable_consumers.clear();
      fusionable_consumers.insert(consumer);
      return;
    }
  }

  bool IsDepency(Group consumer, const std::unordered_set<Group, Hasher, Comparator>& consumers) {
    std::queue<Group> candidates;
    candidates.push(consumer);

    std::unordered_set<Group, Hasher, Comparator> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      visited_set.insert(candidate);
      candidates.pop();

      for (auto& producer : candidate->producer_groups) {
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          candidates.push(producer);
        }
      }
    }

    return false;
  }

  void InitFusionRelation() {
    VLOG(11) << "InitFusionRelation...!";
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
      CHECK(reducer_0) << "Don't find reduce op in group " << first->group_id;

      Node* reducer_1 = nullptr;
      for (auto& reducer : second->master_nodes) {
        if (GetOpKind(reducer) == OpPatternKind::kCommReduce) {
          reducer_1 = reducer;
          break;
        }
      }
      CHECK(reducer_1) << "Don't find reduce op in group " << second->group_id;

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
          dim = reducer_0_reduce_dim.size() - 1;
        }
      }

      for (auto& dim : reducer_1_reduce_dim) {
        // if dim = -1,  set as shape.size() - 1
        if (dim == -1) {
          dim = reducer_1_reduce_dim.size() - 1;
        }
      }

      // check shape is same
      if (reducer_0_input_shape != reducer_1_input_shape || reducer_0_output_shape != reducer_1_output_shape ||
          reducer_0_reduce_dim != reducer_1_reduce_dim) {
        return false;
      }

      return true;
    };

    // kElemWise
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kElemWise];
      // horizontal
      relation.horizontal_relation = {{framework::kElemWise, is_same_shape}};
      // vertical
      relation.vertical_relation = {{OpPatternKind::kElemWise, is_same_shape},
                                    // TODO(sunli) : Using cost-model.
                                    {OpPatternKind::kBroadcast, always_fuse},
                                    // TODO(sunli) : Check can fuse.
                                    {OpPatternKind::kCommReduce, always_fuse}};
    }
    // kBroadcast
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kBroadcast];
      // horizontal
      relation.horizontal_relation = {{framework::kBroadcast, is_same_shape}};
      // vertical
      relation.vertical_relation = {{OpPatternKind::kElemWise, is_same_shape},
                                    {OpPatternKind::kCommReduce, always_fuse}};
    }
    // kInjective
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kInjective];
      // horizontal
      // relation.horizontal_relation = {{OpPatternKind::kInjective, is_same_shape}};
      // vertical
      relation.vertical_relation = {{OpPatternKind::kElemWise, is_same_shape},
                                    {OpPatternKind::kCommReduce, always_fuse}};
    }
    // kCommReduce
    {
      auto& relation = fusion_relation_map_[OpPatternKind::kCommReduce];
      // horizontal
      relation.horizontal_relation = {{OpPatternKind::kCommReduce, reduce_fuse_reduce}};
      // vertical
      relation.vertical_relation = {// TODO(sunli): Using cost-model.
                                    {OpPatternKind::kElemWise, always_fuse}};
    }
  }

  Groups fusion_groups_;

  struct Relation {
    std::unordered_map<framework::OpPatternKind, ConditionFunction> vertical_relation;
    std::unordered_map<framework::OpPatternKind, ConditionFunction> horizontal_relation;
  };
  std::unordered_map<framework::OpPatternKind, Relation> fusion_relation_map_;
};  // namespace hlir

void FusionMergePassInternal(Graph* graph) {
  VLOG(11) << "FusionMergePass...!";
  if (!graph->fusion_groups.size()) {
    VLOG(11) << "Don't do OpFusoin Pass...!";
    return;
  }

  FusionMergePassHelper fusion_merge_pass_helper(graph);
  graph->fusion_groups = fusion_merge_pass_helper();
}

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
