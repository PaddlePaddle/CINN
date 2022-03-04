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
#include <unordered_set>

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
// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class OpFusionPassHelper {
 public:
  OpFusionPassHelper(std::vector<GraphNode*>& graph_nodes, const absl::flat_hash_map<std::string, shape_t>& shape_dict)
      : shape_dict_(shape_dict) {
    // get op pattern dict
    op_pattern_dict_ = &framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    // filter node data, create group for each node
    for (auto graph_node : graph_nodes) {
      auto node = graph_node->safe_as<Node>();
      if (node) {
        nodes_.push_back(node);
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
          // input data has noe source node
          if (input_node_data->source_node.get()) {
            group->input_nodes.insert(input_node_data->source_node.get());
          }
        }

        // group type
        group->op_pattern_kind = GetOpKind(node);
        // use current node as master node for schedule
        group->master_nodes.insert(node);
        fusion_groups_[node] = group;
      }
    }
    // reverse node for output to input
    std::reverse(nodes_.begin(), nodes_.end());
  }

  // return a vector of groups in topological order.
  Groups operator()() {
    DoOpFusion();
    Groups fusion_groups;
    std::unordered_set<Graph::Group*> groups_set;

    for (auto& p : fusion_groups_) {
      if (groups_set.find(p.second.get()) == groups_set.end()) {
        groups_set.insert(p.second.get());
        // groups set
        fusion_groups.push_back(p.second);
      }
    }

    for (auto& group : fusion_groups) {
      // find producer group
      for (auto& node : group->input_nodes) {
        for (auto& edge : node->inlinks()) {
          auto producer_node      = edge->source();
          auto producer_node_data = producer_node->safe_as<NodeData>();
          CHECK(producer_node_data);
          // input data has no source node
          if (producer_node_data->source_node.get()) {
            auto producer_group = fusion_groups_[producer_node_data->source_node.get()];
            group->producer_groups.insert(producer_group.get());
          }
        }
      }
      // find consumer group
      auto output_node      = *group->output_nodes.begin();
      auto output_node_data = (*output_node->outlinks().begin())->sink();
      for (auto& link : output_node_data->outlinks()) {
        auto consumer_node = link->sink()->safe_as<Node>();
        CHECK(consumer_node);
        auto consumer_group = fusion_groups_[consumer_node];
        group->consumer_groups.insert(consumer_group.get());
      }
    }

    return fusion_groups;
  }

 private:
  OpPatternKind GetOpKind(Node* node) {
    CHECK(op_pattern_dict_->Find(node->op())) << "Don't find the pattern of op : " << node->id();
    auto kind = op_pattern_dict_[0][node->op()];
    if (kind == framework::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be element-wise.
      // TODO : sunli
      if (node->op()->name != "broadcast_to") {
        return framework::kElemWise;
      }
    }

    return kind;
  }

  void DoOpFusion() {
    for (auto node : nodes_) {
      auto group      = fusion_groups_[node];
      auto group_kind = group->op_pattern_kind;
      // group-kind (kInjective and > kCommReduce) not support fusion now.
      if (static_cast<int>(group_kind) > static_cast<int>(framework::kCommReduce) ||
          group_kind == framework::kInjective) {
        continue;
      }

      // if current node is broadcast, do not fuse, left to fusion merge
      auto node_op_kind = GetOpKind(node);
      if (node_op_kind == framework::kBroadcast) {
        continue;
      }

      for (auto& edge : node->inlinks()) {
        auto graph_node = edge->source();
        auto node_data  = graph_node->safe_as<NodeData>();
        CHECK(node_data);

        auto source_node = node_data->source_node;
        // is node data is placeholder
        if (!source_node) {
          continue;
        }

        auto source_node_op_kind = GetOpKind(source_node.get());
        // group-kind (kInjective and >= kCommReduce) not support fusion now.
        // TODO(sunli):support other op kind
        if (static_cast<int>(source_node_op_kind) > static_cast<int>(framework::kCommReduce) ||
            source_node_op_kind == framework::kInjective) {
          continue;
        }

        bool can_merge = true;
        // checkout source node output has all in current group
        for (auto& link : node_data->outlinks()) {
          auto out_node = link->sink()->safe_as<Node>();
          CHECK(out_node);
          // if group can't find node, can't merge
          if (group->nodes_set.find(out_node) == group->nodes_set.end()) {
            can_merge = false;
            break;
          }
        }

        if (!can_merge) continue;

        // if node is reduce node, check can fuse reduce + elementwise
        if (source_node_op_kind == framework::kCommReduce && group_kind == framework::kElemWise) {
          auto shape      = shape_dict_.at(source_node->inlinks_in_order()[0]->source()->id());
          auto reduce_dim = absl::get<std::vector<int>>(source_node->attrs.attr_store.at("dim"));
          // last dimension is in reduce, can't fuse reduce + elementwise, left in fusion merge pass
          if (std::find(reduce_dim.begin(), reduce_dim.end(), shape.size() - 1) != reduce_dim.end()) {
            continue;
          }
        }

        // if reduce fuse reduce, the input shape and output shape should be same.
        if (source_node_op_kind == framework::kCommReduce && group_kind == framework::kCommReduce) {
          // check reduce has same input shape and output shape
          auto source_input_shape  = shape_dict_.at(source_node->inlinks_in_order()[0]->source()->id());
          auto source_output_shape = shape_dict_.at(source_node->outlinks_in_order()[0]->sink()->id());

          const Node* master_node = nullptr;
          // find reduce node
          for (auto& rnode : group->master_nodes) {
            if (GetOpKind(rnode) == framework::kCommReduce) {
              master_node = rnode;
              break;
            }
          }
          auto master_input_shape  = shape_dict_.at(master_node->inlinks_in_order()[0]->source()->id());
          auto master_output_shape = shape_dict_.at(master_node->outlinks_in_order()[0]->sink()->id());

          // check shape is same
          if (source_input_shape != master_input_shape || source_output_shape != master_output_shape) {
            continue;
          }

          // check reduce axis
          if (absl::get<std::vector<int>>(master_node->attrs.attr_store.at("dim")) !=
              absl::get<std::vector<int>>(source_node->attrs.attr_store.at("dim"))) {
            continue;
          }
        }

        // start merge node to fusion group
        if (group->nodes_set.find(source_node.get()) == group->nodes_set.end()) {
          group->nodes.push_back(source_node.get());
        }
        group->nodes_set.insert(source_node.get());
        group->input_nodes.erase(source_node.get());
        group->op_pattern_kind =
            static_cast<int>(group_kind) > static_cast<int>(source_node_op_kind) ? group_kind : source_node_op_kind;

        if (source_node_op_kind == framework::kCommReduce) {
          group->master_nodes.insert(source_node.get());
        }

        if (node_data->outlinks().size() > 1) {
          group->internal_nodes.insert(source_node.get());
        }

        // add input node
        for (auto& edge : source_node->inlinks_in_order()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          if (input_node_data->source_node.get()) {
            group->input_nodes.insert(input_node_data->source_node.get());
          }
        }

        // update node group
        fusion_groups_[source_node.get()] = group;
      }
    }
  }

  void InitFusionRelation() {
    // define consumer -> producer fusion relation
    // 1.kElementwise as consumer
    //  producer: kElementwise、kBroacast、kOutEWiseFusable、kInjective
  }

  std::vector<Node*> nodes_;
  std::unordered_map<Node*, Group> fusion_groups_;
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
  // a node map
  std::unordered_set<Node*> nodes_set_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;

  struct FusionRelation {
    std::unordered_set<framework::OpPatternKind> op_kind        = {};
    std::unordered_set<framework::OpPatternKind> fusion_op_kind = {};
  };
  std::unordered_map<framework::OpPatternKind, FusionRelation> fusion_relation_map_;
};

void OpFusionPassInternal(Graph* graph) {
  // nodes include(node, data node)
  auto nodes = std::get<0>(graph->topological_order());
  // shape
  auto& shape_dict = graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  auto op_fusion_helper = OpFusionPassHelper(nodes, shape_dict);
  graph->fusion_groups  = op_fusion_helper();

  for (auto& Group : graph->fusion_groups) {
    VLOG(11) << "Group Start.";
    for (auto node : Group->nodes) {
      VLOG(11) << "node -> " << node->id();
    }
    VLOG(11) << "Group End.";
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(OpFusionPass) {
  CINN_REGISTER_PASS(OpFusionPass)
      .describe(
          "Op Fusion Pass which performs Ops fusion, Producer Ops are fused into Consumer Ops with certain conditions.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::OpFusionPassInternal);

  return true;
}