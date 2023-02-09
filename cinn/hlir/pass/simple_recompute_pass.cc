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

#include <functional>
#include <queue>
#include <unordered_map>

#include "cinn/common/type.h"
#include "cinn/hlir/pass/op_fusion_pass_util.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::AttrType;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::shape_t;

using common::GraphEdge;
using common::GraphNode;
using common::Type;

using DtypeDict    = absl::flat_hash_map<std::string, Type>;
using ShapeDict    = absl::flat_hash_map<std::string, shape_t>;
using NodeAttrType = absl::flat_hash_map<std::string, AttrType>;

namespace {
using CreateNewNodeFunc = std::function<void(Graph*, Node*, const int index, const NodeAttrType&, Node*)>;

class GraphPassHelper {
 public:
  GraphPassHelper(Graph* graph)
      : graph_(graph),
        dtype_dict_(&graph->GetMutableAttrs<DtypeDict>("inferdtype")),
        shape_dict_(&graph->GetMutableAttrs<ShapeDict>("infershape")) {}

  Node* CreateOpNode(const std::string& op_name, const int index, const NodeAttrType& attrs) {
    auto tmp_node = new Node(framework::Operator::Get(op_name), op_name, op_name + "_" + std::to_string(index));
    tmp_node->attrs.attr_store = attrs;
    graph_->RegisterNode(tmp_node->id(), tmp_node);
    return tmp_node;
  }

  NodeData* CreateOutputNode(Node* op_node, const shape_t& shape, const Type& dtype) {
    auto tmp_node_data = new NodeData(Shared<Node>(op_node), 0, 0, common::UniqName("var"), false);
    op_node->LinkTo(tmp_node_data);
    graph_->RegisterNode(tmp_node_data->id(), tmp_node_data);

    // update shape_dict
    CHECK(!shape_dict_->count(tmp_node_data->id())) << "Node " << tmp_node_data->id() << " existed in shape_dict_!";
    shape_dict_->emplace(std::make_pair(tmp_node_data->id(), shape));
    // update dtype_dict
    CHECK(!dtype_dict_->count(tmp_node_data->id())) << "Node " << tmp_node_data->id() << " existed in dtype_dict_!";
    dtype_dict_->emplace(std::make_pair(tmp_node_data->id(), dtype));
    return tmp_node_data;
  }

  NodeData* GetOpNodeInput(Node* node, int index) {
    CHECK(node->op()) << "Only support get input nodedata for op node";

    const auto& in_edges = node->inlinks_in_order(true);
    CHECK_GT(in_edges.size(), index) << "The " << node->op()->name << " op should has at least " << index + 1
                                     << " input, but " << node->id() << " not!";

    auto* in_data = in_edges.at(index)->source()->safe_as<NodeData>();
    CHECK(in_data) << "The " << node->op()->name << "'s input node type should be NodeData, but " << node->id()
                   << " not!";

    return in_data;
  }

  NodeData* GetOpNodeOutput(Node* node, int index) {
    CHECK(node->op()) << "Only support get output nodedata for op node";

    const auto& out_edges = node->outlinks_in_order(true);
    CHECK_GT(out_edges.size(), index) << "The " << node->op()->name << " op should has at least " << index + 1
                                      << " output, but " << node->id() << " not!";

    auto* out_data = out_edges.at(index)->sink()->safe_as<NodeData>();
    CHECK(out_data) << "The " << node->op()->name << "'s output node type should be NodeData, but " << node->id()
                    << " not!";

    return out_data;
  }

  const shape_t& GetNodeDataShape(NodeData* node_data) const {
    CHECK(shape_dict_->count(node_data->id())) << "Cannot found node " << node_data->id() << " in shape_dict_!";
    return shape_dict_->at(node_data->id());
  }

  const Type& GetNodeDataType(NodeData* node_data) const {
    CHECK(dtype_dict_->count(node_data->id())) << "Cannot found node " << node_data->id() << " in dtype_dict_!";
    return dtype_dict_->at(node_data->id());
  }

 private:
  framework::Graph* graph_;
  DtypeDict* dtype_dict_;
  ShapeDict* shape_dict_;
};

void IdentityRecomputeImpl(
    Graph* graph, Node* recompute_op, const int index, const NodeAttrType& attrs, Node* output_node) {
  GraphPassHelper helper(graph);

  auto* in_data  = helper.GetOpNodeInput(recompute_op, 0);
  auto* out_data = helper.GetOpNodeOutput(recompute_op, 0);

  auto* new_node = helper.CreateOpNode(recompute_op->op()->name, index, attrs);
  in_data->LinkTo(new_node);

  auto* new_output_data =
      helper.CreateOutputNode(new_node, helper.GetNodeDataShape(out_data), helper.GetNodeDataType(out_data));
  new_output_data->LinkTo(output_node);
  out_data->UnLinkSingleTo(output_node);
}

void ConstRecomputeImpl(
    Graph* graph, Node* recompute_op, const int index, const NodeAttrType& attrs, Node* output_node) {
  GraphPassHelper helper(graph);

  // hasn't input node
  auto* out_data = helper.GetOpNodeOutput(recompute_op, 0);

  auto* new_node = helper.CreateOpNode(recompute_op->op()->name, index, attrs);

  auto* new_output_data =
      helper.CreateOutputNode(new_node, helper.GetNodeDataShape(out_data), helper.GetNodeDataType(out_data));
  new_output_data->LinkTo(output_node);
  out_data->UnLinkSingleTo(output_node);
}

static std::unordered_map<std::string, CreateNewNodeFunc> need_recompute_op_list = {
    {"identity", IdentityRecomputeImpl},
    {"cast", IdentityRecomputeImpl},
    {"fill_constant", ConstRecomputeImpl},
    {"const_scalar", ConstRecomputeImpl},
    {"arange", ConstRecomputeImpl}};
}  // namespace

class SimpleRecomputePass {
 public:
  SimpleRecomputePass(Graph* graph) : graph_(graph) {}

  void operator()() {
    GraphPassHelper helper(graph_);

    int new_recompute_num = 0;
    auto nodes_inorder    = std::get<0>(graph_->topological_order());

    std::unordered_map<std::string, int> new_recompute_ops;
    for (auto* graph_node : nodes_inorder) {
      auto node = graph_node->safe_as<Node>();
      // if node is NodeData or not op, continue.
      if (!node || node->op() == nullptr) {
        continue;
      }

      // if not cast op, continue
      if (!need_recompute_op_list.count(node->op()->name)) {
        continue;
      }

      auto* out_data            = helper.GetOpNodeOutput(node, 0);
      const auto& next_op_edges = out_data->outlinks();
      if (next_op_edges.size() == 1) {
        // if the cast only link to one node, continue
        continue;
      }

      bool need_cast = false;
      std::vector<Node*> need_recompute_node;
      for (auto edge : next_op_edges) {
        if (!need_cast) {
          // first node use old cast, do not need create new cast, continue
          need_cast = true;
          continue;
        }

        auto* next_op = edge->sink()->safe_as<Node>();
        CHECK(next_op) << out_data->id() << "'s output should be op node!";
        need_recompute_node.emplace_back(next_op);
      }
      for (auto* out_node : need_recompute_node) {
        need_recompute_op_list.at(node->op()->name)(
            graph_, node, nodes_inorder.size() + new_recompute_num, node->attrs.attr_store, out_node);
        new_recompute_ops[node->op()->name]++;
        new_recompute_num++;
      }
    }

    auto debug_recompute_info = [](const std::unordered_map<std::string, int>& new_recompute_ops) {
      std::string info;
      for (const auto& map_pair : new_recompute_ops) {
        info += map_pair.first + " : " + std::to_string(map_pair.second) + ", ";
      }
      return info;
    };

    VLOG(3) << "Total add " << new_recompute_num
            << " recompute op in graph, in case: " << debug_recompute_info(new_recompute_ops);
  }

  framework::Graph* graph_;
};

void SimpleRecomputePassImpl(Graph* graph) {
  std::unordered_set<std::string> fetch_ids;
  std::transform(graph->outputs.begin(),
                 graph->outputs.end(),
                 std::inserter(fetch_ids, fetch_ids.begin()),
                 [](const NodeData* node) { return node->id(); });
  VLOG(3) << "Before SimpleRecomputePass:\n" << graph->DebugGroupedGraph(fetch_ids);

  SimpleRecomputePass pass(graph);
  pass();

  VLOG(3) << "After SimpleRecomputePass:\n" << graph->DebugGroupedGraph(fetch_ids);
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(SimpleRecomputePass) {
  CINN_REGISTER_PASS(SimpleRecomputePass)
      .describe("SimpleRecomputePass Pass which performs \"copy new op for each output op\"")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::SimpleRecomputePassImpl);
  return true;
}
