#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;

void InferShapePass(Graph* src) {
  auto shape_dict = src->GetAttr<std::unordered_map<std::string, std::vector<int>>>("infershape");
  auto dtype_dict = src->GetAttr<std::unordered_map<std::string, Type>>("inferdtype");
  auto store_node = std::get<0>(src->topological_order());
  auto op_infershape =
      Operator::GetAttr<std::function<std::vector<std::vector<int>>(const std::vector<std::vector<int>>&)>>(
          "infershape");
  auto op_inferdtype =
      Operator::GetAttr<std::function<std::vector<Type>(const std::vector<Type>&, const framework::NodeAttr&)>>(
          "inferdtype");
  for (auto i : store_node) {
    if (i->check_type<Node>()) {
      std::vector<std::vector<int>> inputs_shape;
      std::vector<Type> inputs_dtype;
      for (auto j : i->inlinks()) {
        inputs_shape.push_back(shape_dict[j->source()->safe_as<NodeData>()->id()]);
        inputs_dtype.push_back(dtype_dict[j->source()->safe_as<NodeData>()->id()]);
      }
      auto out_shape = op_infershape[i->safe_as<Node>()->op()](inputs_shape);
      auto out_dtype = op_inferdtype[i->safe_as<Node>()->op()](inputs_dtype, i->safe_as<Node>()->attrs);
      int counter    = 0;
      CHECK_EQ(i->outlinks().size(), out_shape.size())
          << "The output number of node " << i->id() << " is " << i->outlinks().size()
          << " , which is different with the output shape size " << out_shape.size() << " . And the op type is "
          << i->safe_as<Node>()->op()->name;
      CHECK_EQ(i->outlinks().size(), out_dtype.size())
          << "The output number of node " << i->id() << " is " << i->outlinks().size()
          << " , which is different with the output dtype size " << out_dtype.size() << " . And the op type is "
          << i->safe_as<Node>()->op()->name;
      for (auto j : i->outlinks()) {
        shape_dict[j->sink()->safe_as<NodeData>()->id()] = out_shape[counter];
        dtype_dict[j->sink()->safe_as<NodeData>()->id()] = out_dtype[counter];
        counter++;
      }
    }
  }
  src->attrs["infershape"] = std::make_shared<std::any>(shape_dict);
  src->attrs["inferdtype"] = std::make_shared<std::any>(dtype_dict);
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
CINN_REGISTER_HELPER(passes) {
  CINN_REGISTER_PASS(InferShape)
      .describe(
          "This pass infer the shape and data type of tensor and save to g.attrs[\"infershape\"] and "
          "g.attrs[\"inferdtype\"].")
      .set_change_structure(false)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::InferShapePass);
}
