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

#include "cinn/common/type.h"
#include "cinn/hlir/pass/fusion_merge_base.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeAttr;

// Dense Merge Pass: merge those gemm which has same var as input into a batched cubals call op.
// A * B, A * C, A * D,...
// after
// A * [B, C, D,...]
// Using cublas batched gemm can avoid do concat and slice.

class DenseMergePassHelper : public FusionHelperBase {
 public:
  DenseMergePassHelper(Graph* graph) : FusionHelperBase(graph), graph_(graph) {}

  void operator()() {
    auto nodes_inorder = std::get<0>(graph_->topological_order());
    for (auto node : nodes_inorder) {
      if (node->safe_as<NodeData>()) {
        MergeDense(node);
      }
    }
  }

 private:
  void MergeDense(NodeData* node) {
    auto dense_ops = GetDenseOp(node);
    if (dense_ops.size() <= 1) {
      return;
    }

    std::vector<Node*> lhs_ops, rhs_ops;
    for (auto op : dense_ops) {
      if (op->inlinks_in_order()[0]->source == node) {
        lhs_ops.push_back(op);
      } else {
        rhs_ops.push_back(op);
      }
    }

    if (lhs_ops.size() > 1) LeftMerge(node, lhs_ops);
    if (rhs_ops.size() > 1) RightMerge(node, rhs_ops);
  }

  std::vector<Node*> GetDenseOp(NodeData* node) {
    std::vector<Node*> dense_ops;
    auto outlinks = node->outlinks();
    for (auto link : outlinks) {
      auto sink = link->sink()->safe_as<Node>();
      if (node->op()->name == "matmul" || node->op()->name == "mul" || node->op()->name == "cublas_gemm" ||
          node->op()->name == "cublas_matmul") {
        dense_ops.push_back(sink);
      }
    }
    return dense_ops;
  }

  void LeftMerge(NodeData* node, std::vector<Node*> dense_ops) { DoMerge(node, dense_ops, 1, "left"); }

  void RightMerge(NodeData* node, std::vector<Node*> dense_ops) { DoMerge(node, dense_ops, 0, "right"); }

  void DoMerge(NodeData* node, std::vector<Node*> dense_ops, int pos, std::string side) {
    // split dense op by it's attr
    std::unordered_map<std::string, std::vector<Node*>> dense_op_map;
    for (auto dense_op : dense_ops) {
      auto sign = GenOpSign(dense_op->inlinks_in_order()[pos]->safe_as<NodeData>(), dense_op->attrs);
      if (dense_op_map.count(sign)) {
        dense_op_map[sign].push_back(dense_op);
      } else {
        dense_op_map[sign] = {dense_op};
      }
    }

    for (auto dense_op : dense_op_map) {
      if (dense_op.second.size() <= 1) {
        continue;
      }

      // create custom call node
      Node* node_tmp = new Node(Operator::Get("custom_call"), "custom_call", common::UniqName("custom_call"));
      graph_->RegisterNode(node_tmp);
      node_tmp->attrs         = dense_op.second[0]->attrs;
      node_tmp->attrs["side"] = side node_tmp->attrs["custom_call"] = "cinn_call_batched_cublas";

      // update inlink.
      node->LinkTo(node_tmp);
      for (auto dense_op : op.second) {
        node->UnLinkSingleTo(dense_op);
        // link to new node
        dense_op->inlinks_in_order()[pos]->source()->LinkTo(node_tmp);
        // unlink old dense node
        dense_op->inlinks_in_order()[pos]->source()->UnLinkSingleTo(dense_op);
        // dense_node_data link to node_tmp
        auto dense_node_data = GetNodeData(dense_op);
        dense_op->UnLinkSingleTo(dense_node_data);
        node_tmp->LinkTo(dense_node_data);
        // update node tmp.
        dense_node_data->Source_node.Reset(node_tmp);

        // drop dense op.
        graph_->DropNode(dense_op);
      }
    }
  }

  std::string GenOpSign(const NodeData* node, const NodeAttr& attr) {
    auto shape         = shape_dict_.at(node0->id());
    bool trans_a       = attr_store.count("trans_a") ? absl::get<bool>(attr_store.at("trans_a")) : false;
    bool trans_b       = attr_store.count("trans_b") ? absl::get<bool>(attr_store.at("trans_b")) : false;
    bool trans_out     = attr_store.count("trans_out") ? absl::get<bool>(attr_store.at("trans_out")) : false;
    float alpha        = attr_store.count("alpha") ? absl::get<float>(attr_store.at("alpha")) : 1.0f;
    float beta         = attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;
    int x_num_col_dims = attr_store.count("x_num_col_dims") ? absl::get<int>(attr_store.at("x_num_col_dims")) : 0;
    int y_num_col_dims = attr_store.count("y_num_col_dims") ? absl::get<int>(attr_store.at("y_num_col_dims")) : 0;

    std::string sign = "";
    sign += std::to_string(trans_a);
    sign += "_" + std::to_string(trans_b);
    sign += "_" + std::to_string(trans_out);
    sign += "_" + std::to_string(alpha);
    sign += "_" + std::to_string(beta);
    sign += "_" + std::to_string(x_num_col_dims);
    sign += "_" + std::to_string(y_num_col_dims);
    for (auto s : shape) {
      sign += "_" + std::to_string(s);
    }

    return sign;
  }

  Graph* graph_;
};

void DenseMergePassInternal(Graph* graph) {
  DenseMergePassHelper dense_merge_pass_helper(graph);
  dense_merge_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(DenseMerge) {
  CINN_REGISTER_PASS(DenseMerge)
      .describe("")
      .set_change_structure(true)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::DenseMergePassInternal);
  return true;
}
