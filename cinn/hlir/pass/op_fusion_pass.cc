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
/*********** Op Fusion Pass*************************
 *
 * fuse producer into comsumer.
 * iteration until no group can merge.
 *
 *********************************************************/

struct OpFusionPassHelper {
 public:
  OpFusionPassHelper(std::vector<GraphNode*> graph_nodes, const absl::flat_hash_map<std::string, shape_t>& shape_dict)
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
        group->nodes_.push_back(node);
        group->nodes_set_.insert(node);
        group->output_nodes_.insert(node);
        // input node
        for (auto& edge : node->inlinks()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          group->input_nodes_.insert(input_node_data->source_node.get());
        }

        // group type
        CHECK(op_pattern_dict_->Find(node->op())) << "Don't find the pattern of op : " << node->id();
        group->op_pattern_kind_ = op_pattern_dict_[0][node->op()];
        if (group->op_pattern_kind_ == framework::kCommReduce) {
          group->master_nodes_.insert(node);
        }
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
    return fusion_groups;
  }

 private:
  void DoOpFusion() {
    for (auto node : nodes_) {
      auto group      = fusion_groups_[node];
      auto group_kind = group->op_pattern_kind_;
      // group-kind (kInjective and > kCommReduce) not support fusion now.
      if (static_cast<int>(group_kind) > static_cast<int>(framework::kCommReduce) ||
          group_kind == framework::kInjective) {
        continue;
      }

      for (auto& edge : node->inlinks()) {
        auto graph_node = edge->source();
        auto node_data  = graph_node->safe_as<NodeData>();

        CHECK(node_data);
        auto source_node = node_data->source_node;

        auto source_node_op_kind = op_pattern_dict_[0][source_node->op()];
        // group-kind (kInjective and > kCommReduce) not support fusion now.
        if (static_cast<int>(source_node_op_kind) > static_cast<int>(framework::kCommReduce) ||
            source_node_op_kind == framework::kInjective) {
          continue;
        }

        bool can_merge = true;
        // checkout source node output has all in current group
        for (auto& link : node_data->outlinks()) {
          auto out_node = link->sink()->safe_as<Node>();
          CHECK(out_node);
          if (group->nodes_set_.find(out_node) == group->nodes_set_.end()) {
            can_merge = false;
            break;
          }
        }

        if (!can_merge) continue;
        // checkout node type can be merged into group.
        // element-wise + element-wise
        // element-wise + broadcast
        // element-wise + reduce
        // broadcast + element-wise
        // broadcast + reduce
        // reduce + element-wise
        // reduce + reduce

        // check broadcast + broadcast
        if (source_node_op_kind == framework::kBroadcast && group_kind == framework::kBroadcast) {
          // broadcast + broadcast not support fuse.
          continue;
        }

        // check reduce + broadcast
        if (source_node_op_kind == framework::kCommReduce && group_kind == framework::kBroadcast) {
          // reduce + broadcast not support fuse.
          continue;
        }

        // if reduce fuse reduce, the input shape and output shape should be same.
        if (source_node_op_kind == framework::kCommReduce && group_kind == framework::kCommReduce) {
          // check reduce has same input shape and output shape
          auto source_input_shape  = shape_dict_[source_node->inlinks_in_order()[0]->source()->id()];
          auto source_output_shape = shape_dict_[source_node->outlinks_in_order()[0]->sink()->id()];

          const Node* master_node  = *group->master_nodes_.begin();
          auto master_input_shape  = shape_dict_[master_node->inlinks_in_order()[0]->source()->id()];
          auto master_output_shape = shape_dict_[master_node->outlinks_in_order()[0]->sink()->id()];

          if (source_input_shape != master_input_shape || source_output_shape != master_output_shape) {
            continue;
          }
        }

        // do merge
        group->nodes_.push_back(source_node.get());
        group->nodes_set_.insert(source_node.get());
        group->input_nodes_.erase(source_node.get());
        group->op_pattern_kind_ =
            static_cast<int>(group_kind) > static_cast<int>(source_node_op_kind) ? group_kind : source_node_op_kind;

        if (source_node_op_kind == framework::kCommReduce) {
          group->master_nodes_.insert(source_node.get());
        }
        // add input node
        for (auto& edge : source_node->inlinks_in_order()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          group->input_nodes_.insert(input_node_data->source_node.get());
        }

        // update node group
        fusion_groups_[source_node.get()] = group;
      }
    }
  }

  std::vector<Node*> nodes_;
  std::unordered_map<Node*, Group> fusion_groups_;
  absl::flat_hash_map<std::string, shape_t> shape_dict_;
  // a node map
  std::unordered_set<Node*> nodes_set_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;
};

void OpFusionPassInternal(Graph* graph) {
  // nodes include(node, data node)
  auto nodes = std::get<0>(graph->topological_order());
  // shape
  auto& shape_dict = graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  auto op_fusion_helper = OpFusionPassHelper(nodes, shape_dict);
  graph->fusion_groups_ = op_fusion_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(OpFusionPass) {
  CINN_REGISTER_PASS(OpFusionPass)
      .describe("This pass does op fusion.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::OpFusionPassInternal);

  return true;
}