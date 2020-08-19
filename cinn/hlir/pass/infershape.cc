#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
namespace cinn {
namespace hlir {
namespace pass {
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
void InferShapePass(Graph* src) {
  auto res        = src->GetAttr<std::unordered_map<std::string, std::vector<int>>>("infershape");
  auto store_node = std::get<0>(src->topological_order());
  auto op_infershape =
      Operator::GetAttr<std::function<std::vector<std::vector<int>>(std::vector<std::vector<int>>)>>("infershape");
  for (auto i : store_node) {
    if (i->check_type<Node>()) {
      std::vector<std::vector<int>> inputs_shape;
      for (auto j : i->inlinks()) {
        inputs_shape.push_back(res[j->source()->safe_as<NodeData>()->id()]);
      }
      auto out_shape = op_infershape[i->safe_as<Node>()->op()](inputs_shape);
      int counter    = 0;
      CHECK_EQ(i->outlinks().size(), out_shape.size())
          << "The output number of node " << i->id() << " is " << i->outlinks().size()
          << " , which is different with the output shape size " << out_shape.size() << " . And the op type is "
          << i->safe_as<Node>()->op()->name;
      for (auto j : i->outlinks()) {
        res[j->sink()->safe_as<NodeData>()->id()] = out_shape[counter++];
      }
    }
  }
  src->attrs["infershape"] = std::make_shared<std::any>(res);
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
CINN_REGISTER_HELPER(passes) {
  CINN_REGISTER_PASS(InferShape)
      .describe("This pass infer the shape of tensor and save to g.attrs[\"infer_shape\"].")
      .set_change_structure(false)
      .provide_graph_attr("infershape")
      .set_body(cinn::hlir::pass::InferShapePass);
}
