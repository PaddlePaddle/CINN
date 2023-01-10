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

#include <string>
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

using InputToNodeMap = std::unordered_map<std::string, std::unordered_set<Node*>>;
using shape_dict_t   = absl::flat_hash_map<std::string, framework::shape_t>;

std::unordered_set<std::string> unordered_ops = {
    "elementwise_add",
    "elementwise_mul",
    "max",
    "min",
    "logical_and",
    "logical_or",
    "logical_xor",
    "equal",
    "not_equal",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_and",
};

bool IsSameSubexpression(Node* op1, Node* op2, shape_dict_t& shape_dict) {
  auto op1_in_edges    = op1->inlinks_in_order(true);
  auto op2_in_edges    = op2->inlinks_in_order(true);
  auto op1_inputs_size = op1_in_edges.size();
  auto op2_inputs_size = op2_in_edges.size();
  if (op1_inputs_size != op2_inputs_size) {
    return false;
  }
  auto op1_attrs_size = op1->attrs.attr_store.size();
  auto op2_attrs_size = op2->attrs.attr_store.size();
  if (op1_attrs_size != op2_attrs_size) {
    return false;
  }
  if (unordered_ops.count(op1->op()->name)) {
    for (auto& op1_edge : op1_in_edges) {
      auto* op1_source_node = op1_edge->source()->safe_as<NodeData>();
      CHECK(op1_source_node);
      bool op1_equal_op2 = std::any_of(op2_in_edges.begin(), op2_in_edges.end(), [&](common::Shared<GraphEdge>& edge) {
        auto* op2_source_node = edge->source()->safe_as<NodeData>();
        CHECK(op2_source_node);
        if (op1_source_node->id() == op2_source_node->id()) {
          return true;
        }
        return false;
      });
      if (!op1_equal_op2) {
        return false;
      }
    }
  } else {
    for (int i = 0; i < op1_inputs_size; ++i) {
      auto* op1_source_node = op1_in_edges[i]->source()->safe_as<NodeData>();
      auto* op2_source_node = op2_in_edges[i]->source()->safe_as<NodeData>();
      CHECK(op1_source_node);
      CHECK(op2_source_node);
      if (op1_source_node->id() != op2_source_node->id()) {
        return false;
      }
    }
  }
  if (op1->op()->name == "reshape") {
    auto* op1_sink_node = op1->outlinks_in_order(true)[0]->sink()->safe_as<NodeData>();
    auto* op2_sink_node = op2->outlinks_in_order(true)[0]->sink()->safe_as<NodeData>();
    return shape_dict[op1_sink_node->id()] == shape_dict[op2_sink_node->id()];
  } else {
    auto* op1_sink_node = op1->outlinks_in_order(true)[0]->sink()->safe_as<NodeData>();
    auto* op2_sink_node = op2->outlinks_in_order(true)[0]->sink()->safe_as<NodeData>();
    if (shape_dict[op1_sink_node->id()].size() != shape_dict[op2_sink_node->id()].size()) {
      return false;
    }
    return std::all_of(op1->attrs.attr_store.begin(), op1->attrs.attr_store.end(), [&](auto attr) {
      if (!op2->attrs.attr_store.count(attr.first) || op2->attrs.attr_store[attr.first] != attr.second) {
        if (attr.first == "axis" || attr.first == "dim") {
          auto op1_axis = absl::get<int>(attr.second);
          auto op2_axis = absl::get<int>(op2->attrs.attr_store[attr.first]);
          if (op1_axis < 0) {
            op1_axis += shape_dict[op1_sink_node->id()].size();
          }
          if (op2_axis < 0) {
            op2_axis += shape_dict[op1_sink_node->id()].size();
          }
          return op2_axis == op1_axis;
        }
        return false;
      }
      return true;
    });
  }
}

void RemoveNode(framework::Graph* graph, Node* node) {
  auto in_edges = node->inlinks();
  for (auto& edge : in_edges) {
    auto* in_node = edge->source()->safe_as<NodeData>();
    in_node->UnLinkSingleTo(node);
  }
  auto out_edges = node->outlinks();
  for (auto& edge : out_edges) {
    auto* out_node = edge->sink()->safe_as<NodeData>();
    CHECK(out_node);
    node->UnLinkSingleTo(out_node);
    graph->DropNode(out_node);
  }
  graph->DropNode(node);
  LOG(INFO) << "remove " << node->id() << " node.";
}

void ReplaceNode(NodeData* src_new, NodeData* src_old, Node* trt) {
  std::vector<NodeData*> in_nodes;
  for (auto& in_edge : trt->inlinks_in_order(true)) {
    auto* in_node = in_edge->source()->safe_as<NodeData>();
    if (in_node->id() == src_old->id()) {
      in_nodes.emplace_back(src_new);
    } else {
      in_nodes.emplace_back(in_node);
    }
    in_node->UnLinkSingleTo(trt);
  }
  for (auto in_node : in_nodes) {
    in_node->LinkTo(trt);
  }
}

int CommonSubexpressionElimination(Graph* graph, std::vector<GraphNode*>& store_nodes, InputToNodeMap in2node) {
  std::unordered_map<std::string, std::vector<Node*>> expr_map;
  auto shape_dict = graph->GetAttrs<absl::flat_hash_map<std::string, framework::shape_t>>("infershape");
  int remove_num  = 0;
  for (auto& graph_node : store_nodes) {
    auto node = graph_node->safe_as<Node>();
    if (node) {
      auto& node_type  = node->op()->name;
      auto& candidates = expr_map[node_type];
      bool found       = false;
      for (auto* candidate_node : candidates) {
        if (!IsSameSubexpression(node, candidate_node, shape_dict)) continue;
        found = true;
        for (int k = 0; k < node->outlinks_in_order(true).size(); ++k) {
          CHECK(node->outlinks_in_order(true).size() == candidate_node->outlinks_in_order(true).size());
          auto* sink_node           = node->outlinks_in_order(true)[k]->sink()->safe_as<NodeData>();
          auto* candidate_sink_node = candidate_node->outlinks_in_order(true)[k]->sink()->safe_as<NodeData>();
          CHECK(sink_node);
          CHECK(candidate_sink_node);
          auto out_nodes = in2node[sink_node->id()];
          for (auto out_node : out_nodes) {
            ReplaceNode(candidate_sink_node, sink_node, out_node);
            out_nodes.erase(node);
            out_nodes.insert(candidate_node);
          }
        }
        RemoveNode(graph, node);
        remove_num++;
        break;
      }
      if (!found) {
        expr_map[node_type].push_back(node);
      }
    }
  }
  return remove_num;
}

void CommonSubexpressionEliminationPass(Graph* graph) {
  VLOG(3) << "CommonSubexpressionEliminationPass...!";
  std::unordered_map<std::string, std::vector<Node*>> expr_map;
  InputToNodeMap in2node;
  auto store_nodes = std::get<0>(graph->topological_order());

  for (auto& graph_node : store_nodes) {
    auto node = graph_node->safe_as<Node>();
    if (node) {
      for (auto& in_edge : node->inlinks_in_order(true)) {
        auto* source_node = in_edge->source()->safe_as<NodeData>();
        in2node[source_node->id()].insert(node);
      }
    }
  }

  int remove_num = CommonSubexpressionElimination(graph, store_nodes, in2node);
  while (remove_num) {
    store_nodes = std::get<0>(graph->topological_order());
    remove_num  = CommonSubexpressionElimination(graph, store_nodes, in2node);
  }
  VLOG(3) << "CommonSubexpressionEliminationPass Finish...!";
}
}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(CommonSubexpressionEliminationPass) {
  CINN_REGISTER_PASS(CommonSubexpressionEliminationPass)
      .describe("This pass  will remove these same sub-expression.")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::CommonSubexpressionEliminationPass);

  return true;
}
