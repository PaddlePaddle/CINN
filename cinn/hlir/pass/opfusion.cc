#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
using framework::OpPatternKind;

void OpFusionPass(Graph* graph) {
  auto store_nodes      = std::get<0>(graph->topological_order());
  auto& op_pattern_dict = Operator::GetAttrs<OpPatternKind>("OpPattern");
  for (int i = 0; i < store_nodes.size(); i++) {
    auto node = store_nodes[i]->safe_as<Node>();
    if (node) {
      auto op_pattern = op_pattern_dict[node->op()];
      if (op_pattern <= framework::kOutEWiseFusable) {
        int fuse_number = 1;
        while (i + 2 < store_nodes.size() && store_nodes[i + 2]->safe_as<Node>()) {
          auto temp_node = store_nodes[i + 2]->safe_as<Node>();
          if (op_pattern_dict[temp_node->op()] <= framework::kElemWise) {
            i = i + 2;
            fuse_number++;
          } else {
            break;
          }
        }
        if (fuse_number > 1) {
          node->attrs.attr_store["FuseNumber"] = fuse_number;
        }
      }
    }
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
CINN_REGISTER_HELPER(OpFusion) {
  CINN_REGISTER_PASS(OpFusion)
      .describe("This pass traverse the graph and fuse all ops.")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::OpFusionPass);

  return true;
}
