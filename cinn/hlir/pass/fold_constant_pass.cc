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

using GroupPtr  = std::shared_ptr<Graph::Group>;
using GroupList = std::vector<GroupPtr>;

using AlterFunction = std::function<bool(Graph*, const Node*, const Node*)>;

// Fold Constant Pass
//
class FoldConstantPassHelper : public FusionHelperBase {
 public:
  FoldConstantPassHelper(Graph* graph) : FusionHelperBase(graph), graph_(graph) {}

  void operator()() {
    auto nodes_inorder = std::get<0>(graph->topological_order());
    bool update        = false;
    do {
      for (auto node : nodes_inorder) {
      }
    } while (update);
  }

 private:
  std::unordered_map<std::string, AlterFunction> alter_function_;
  Graph* graph_;
};

void FoldConstantPassInternal(Graph* graph) {
  FoldConstantPassHelper fold_constant_pass_helper(graph);
  fold_constant_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(ConstantFoldPass) {
  CINN_REGISTER_PASS(ConstantFoldPass)
      .describe("Op Fusion Pass which performs \"fold constant\"")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::FoldConstantPassInternal);

  return true;
}
