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

using common::GraphEdge;
using common::GraphNode;

using Group  = std::shared_ptr<Graph::Group>;
using Groups = std::vector<std::shared_ptr<Graph::Group>>;

/*********** Vertical Fusion Pass*************************
 *
 * the condition of two groups can merge: output(group0) + output(group1) <= output(group1)
 * iteration until no group can merge.
 *
 *********************************************************/

struct VerticalFusionHelper {
 public:
  VerticalFusionHelper(std::vector<GraphNode*> graph_nodes, const absl::flat_hash_map<std::string, shape_t>& shape_dict)
      : shape_dict_(shape_dict) {
    // get op pattern dict
    op_pattern_dict_ = framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    // filter node data.
    for (auto graph_node : graph_nodes) {
      auto node = graph_node->safe_as<Node>();
      if (node) {
        nodes_.push_back(node);
        auto group = {std::make_shared<Graph::Group>()};
        // init group
        group->nodes_.push_back(node);
        graph->nodes_set_.push_back(node);
        group->output_nodes_.insert(node);
        for (auto& edge : node->inlinks()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          group->input_nodes_.insert(input_node_data->source_node.get());
        }

        CHECK(op_pattern_dict_.find(node->op())) << "Don't find the pattern of op : " << node->id();
        group->op_pattern_kind_ = op_pattern_dict_[node->op()];
        fusion_groups_[node]    = {group};
      }
    }
    // reverse node for output to input
    std::reverse(node_.begin(), node_.end());
  }

  // return a vector of groups in topological order.
  Groups operator()() {
    VerticalFusion() std::unordered_set<Group*> groups_set;
    std::vector<std::shared_ptr<Group>> fusion_groups;
    for (auto& p : fusion_groups_) {
      if (groups_set.find(p.second[0].get()) == groups_set.end()) {
        fusion_groups.push_back(p.second[0]);
        groups_set.insert(p.second[0].get());
      }
    }

    return fusion_groups;
  }

 private:
  void VerticalFusion() {
    for (auto node : nodes_) {
      auto group      = fusion_groups_[node][0];
      auto group_kind = group->op_pattern_kind_;
      // group kind not support fusion now.
      if (static_cast<int>(group_kind) > static_cast<int>(framework::kCommReduce) ||
          group_kind == framework::kInjective) {
        continue;
      }

      for (auto& edge : node->inlinks()) {
        auto graph_node = edge->source();
        auto node_data  = graph_node->safe_as<NodeData>();

        CHECK(node_data);
        std::shared_ptr<Node> source_node = node_data->source_node;

        auto source_node_op_kind = op_pattern_dict_[source_node->op()];
        if (static_cast<int>(source_node_op_kind) > static_cast<int>(framework::kCommReduce) ||
            source_node_op_kind == framework::kInjective) {
          continue;
        }

        bool can_merge = true;
        // checkout source node output has all in current group
        for (auto& link : node_data->outlinks()) {
          auto out_node = link->sink()->safe_as<Node>();
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
        // broadcast + broadcast
        // broadcast + reduce
        // reduce + element-wise
        // reduce + reduce
        if (source_node_op_kind == framework::kCommReduce && group_kind == framework::kBroadcast) {
          // reduce + broadcast not support fuse.
          continue;
        }

        // if reduce fuse reduce, the input shape and output shape should be same.
        if (source_node_op_kind == framework::kCommReduce && group_kind == framework::kCommReduce) {
          // check reduce has same input shape and output shape
          auto source_input_shape  = shape_dict_[source_node->inlinks()[0]->source()->id()];
          auto source_output_shape = shape_dict_[source_node->outlinks()[0]->sink()->id()];

          auto master_node         = group->master_nodes_.begin();
          auto master_input_shape  = shape_dict_[master_node->inlinks()[0]->source()->id()];
          auto master_output_shape = shape_dict_[master_node->outlinks()[0]->sink()->id()];

          if (source_input_shape != master_input_shape || source_output_shape != master_output_shape) {
            continue;
          }
        }

        // do merge
        group->nodes_.push_back(source_node);
        group->nodes_set_.insert(source_node);
        group->input_nodes_.erase(source);
        group->op_pattern_kind_ =
            static_cast<int>(group_kind) > static_cast<int>(source_node_op_kind) ? group_kind : source_node_op_kind;

        if (source_node_op_kind == framework::kCommReduce) {
          group->master_nodes_.insert(source_node);
        }
        // add input node
        for (auto& edge : source_node->inlinks_in_order()) {
          auto input_graph_node = edge->source();
          auto input_node_data  = input_graph_node->safe_as<NodeData>();
          CHECK(input_node_data);
          group->input_nodes_.insert(input_node_data->source_node.get());
        }
      }
    }
  }

  std::vector<Node*> nodes_;
  std::unordered_map<Node*, Groups> fusion_groups_;
  absl::flat_hash_map<std::string, shape_t> shape_dict_;
  // a node map
  std::unordered_set<Node*> nodes_set_;
  // op pattern dict
  framework::OpValueType<OpPatternKind> op_pattern_dict_;
};

void VerticalFusionPass(Graph* graph) {
  auto nodes        = std::get<0>(graph->topological_order());
  auto output_nodes = graph->outputs;

  for (auto node_data : output_nodes) {
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(VerticalFusion) {
  CINN_REGISTER_PASS(VerticalFusion)
      .describe("This pass fusion ops by vertical.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::VerticalFusionPass);

  return true;
}
