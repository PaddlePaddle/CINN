// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/common/type.h"
#include "cinn/hlir/pass/op_fusion_pass_util.h"

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
using common::Type;

using DtypeDict = absl::flat_hash_map<std::string, Type>;
using ShapeDict = absl::flat_hash_map<std::string, shape_t>;

class CastRecomputePass {
 public:
  CastRecomputePass(Graph* graph)
      : graph_(graph),
        dtype_dict_(&graph->GetMutableAttrs<DtypeDict>("inferdtype")),
        shape_dict_(&graph->GetMutableAttrs<ShapeDict>("infershape")) {}

  void CreateCastNode(NodeData* input_data, const int index, const Type& dtype, Node* output_op) {
    // create node
    auto tmp_node = new Node(framework::Operator::Get("cast"), "cast", "cast_" + std::to_string(index));
    tmp_node->attrs.attr_store["dtype"] = common::Type2Str(dtype);
    input_data->LinkTo(tmp_node);
    graph_->RegisterNode(tmp_node->id(), tmp_node);

    // create node data
    auto tmp_node_data = new NodeData(Shared<Node>(tmp_node), 0, 0, common::UniqName("var"), false);
    tmp_node->LinkTo(tmp_node_data);
    tmp_node_data->LinkTo(output_op);
    graph_->RegisterNode(tmp_node_data->id(), tmp_node_data);

    // update shape_dict
    CHECK(shape_dict_->count(input_data->id())) << "Cannot found node " << input_data->id() << " in shape_dict_!";
    CHECK(!shape_dict_->count(tmp_node_data->id())) << "Node " << tmp_node_data->id() << " existed in shape_dict_!";
    shape_dict_->emplace(std::make_pair(tmp_node_data->id(), shape_dict_->at(input_data->id())));
    // update dtype_dict
    dtype_dict_->emplace(std::make_pair(tmp_node_data->id(), dtype));
  }

  void operator()() {
    int new_cast_num   = 0;
    auto nodes_inorder = std::get<0>(graph_->topological_order());

    for (auto* graph_node : nodes_inorder) {
      auto node = graph_node->safe_as<Node>();
      // if node is NodeData or not op, continue.
      if (!node || node->op() == nullptr) {
        continue;
      }

      // if not cast op, continue
      if ("cast" != node->op()->name) {
        continue;
      }

      const auto& cast_in_edges = node->inlinks();
      CHECK_EQ(cast_in_edges.size(), 1) << "The cast op should only has one output, but " << node->id() << " not!";
      auto* cast_in_data = (*cast_in_edges.begin())->source()->safe_as<NodeData>();
      CHECK(cast_in_data) << "The cast's input node type should be NodeData, but " << node->id() << " not!";

      const auto& cast_out_edges = node->outlinks();
      CHECK_EQ(cast_out_edges.size(), 1) << "The cast op should only has one output, but " << node->id() << " not!";
      auto* cast_out_data = (*cast_out_edges.begin())->sink()->safe_as<NodeData>();
      CHECK(cast_out_data) << "The cast's output node type should be NodeData, but " << node->id() << " not!";

      const auto& next_op_edges = cast_out_data->outlinks();
      if (next_op_edges.size() == 1) {
        // if the cast only link to one node, continue
        continue;
      }

      CHECK(dtype_dict_->count(cast_out_data->id()))
          << "Cannot found node " << cast_out_data->id() << " in dtype_dict_!";
      const auto& dtype = dtype_dict_->at(cast_out_data->id());
      bool need_cast    = false;
      std::vector<Node*> need_cast_nodes;
      for (auto edge : next_op_edges) {
        if (!need_cast) {
          // first node use old cast, do not need create new cast, continue
          need_cast = true;
          continue;
        }

        auto* next_op = edge->sink()->safe_as<Node>();
        CHECK(next_op) << cast_out_data->id() << "'s output should be op node!";
        need_cast_nodes.emplace_back(next_op);
      }
      for (auto* node : need_cast_nodes) {
        CreateCastNode(cast_in_data, nodes_inorder.size() + new_cast_num, dtype, node);
        cast_out_data->UnLinkSingleTo(node);
        new_cast_num++;
      }
    }

    VLOG(3) << "Total add " << new_cast_num << " cast op in graph.";
  }

  framework::Graph* graph_;
  DtypeDict* dtype_dict_;
  ShapeDict* shape_dict_;
};

void CastRecomputePassImpl(Graph* graph) {
  std::unordered_set<std::string> fetch_ids;
  std::transform(graph->outputs.begin(),
                 graph->outputs.end(),
                 std::inserter(fetch_ids, fetch_ids.begin()),
                 [](const NodeData* node) { return node->id(); });
  VLOG(3) << "Before CastRecomputePass:\n" << graph->DebugGroupedGraph(fetch_ids);

  CastRecomputePass pass(graph);
  pass();

  VLOG(3) << "After CastRecomputePass:\n" << graph->DebugGroupedGraph(fetch_ids);
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(CastRecomputePass) {
  CINN_REGISTER_PASS(CastRecomputePass)
      .describe("CastRecompute Pass which performs \"operator recompute\"")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::CastRecomputePassImpl);
  return true;
}
