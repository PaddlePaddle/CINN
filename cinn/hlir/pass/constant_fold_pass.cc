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

#include "cinn/common/type.h"
#include "cinn/hlir/pass/constant_fold_pass_util.h"

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

using AlterFunction = std::function<void(const FusionHelperBase*, Graph*, Node*)>;

// Constant Fold Pass
//
class ConstantFoldPassHelper : public FusionHelperBase {
 public:
  ConstantFoldPassHelper(Graph* graph) : FusionHelperBase(graph), graph_(graph) {}

  void operator()() {
    auto nodes_inorder = std::get<0>(graph_->topological_order());
    bool update        = false;
    do {
      update = false;
      for (auto node : nodes_inorder) {
        if (!node->safe_as<Node>()) {
          continue;
        }
        // check producer's type
        auto producers = GetProducerNode(node->safe_as<Node>());
        if (producers.empty()) {
          continue;
        }

        bool can_fold = true;
        for (auto producer : producers) {
          if (!IsConstOp(producer)) {
            can_fold = false;
            break;
          }
        }

        if (!can_fold) continue;

        auto key = GetTypeName(node->safe_as<Node>());
        if (alter_function_.count(key)) {
          alter_function_[key](this, graph_, node->safe_as<Node>());
        }

        update = true;
      }
    } while (update);
  }

 private:
  void RegisterAlterFunction() {
    alter_function_ = {{"broadcast_to_const_scalar", fold_broadcast_to_constant},
                       {"broadcast_to_fill_constant", fold_broadcast_to_constant},
                       {"reshape_fill_constant", fold_reshape_fill_constant},
                       {"squeeze_fill_constant", fold_squeeze_fill_constant},
                       {"expand_dims_to_fill_constant", fold_expand_dims_to_fill_constant}};
  }

  std::string GetTypeName(Node* node) {
    auto producers  = GetProducerNode(node->safe_as<Node>());
    std::string key = node->op()->name;
    for (auto producer : producers) {
      key += std::string("_") + producer->op()->name;
    }
    return key;
  }

  std::unordered_map<std::string, AlterFunction> alter_function_;
  Graph* graph_;
};

void ConstantFoldPassInternal(Graph* graph) {
  ConstantFoldPassHelper constant_fold_pass_helper(graph);
  constant_fold_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(ConstantFoldPass) {
  CINN_REGISTER_PASS(ConstantFoldPass)
      .describe("Op Fusion Pass which performs \"fold constant\"")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::ConstantFoldPassInternal);

  return true;
}
