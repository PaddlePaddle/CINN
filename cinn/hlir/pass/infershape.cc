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
  auto& shape_dict = src->GetMutableAttrs<std::unordered_map<std::string, std::vector<int>>>("infershape");
  auto& dtype_dict = src->GetMutableAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  auto store_node  = std::get<0>(src->topological_order());
  auto& op_infershape =
      Operator::GetAttrs<std::function<std::vector<std::vector<int>>(const std::vector<std::vector<int>>&)>>(
          "infershape");
  auto& op_inferdtype =
      Operator::GetAttrs<std::function<std::vector<Type>(const std::vector<Type>&, const framework::NodeAttr&)>>(
          "inferdtype");
  for (auto& node : store_node) {
    if (node->check_type<Node>()) {
      std::vector<std::vector<int>> inputs_shape;
      std::vector<Type> inputs_dtype;
      for (auto& in_edge : node->inlinks()) {
        inputs_shape.push_back(shape_dict[in_edge->source()->safe_as<NodeData>()->id()]);
        inputs_dtype.push_back(dtype_dict[in_edge->source()->safe_as<NodeData>()->id()]);
      }
      auto out_shape = op_infershape[node->safe_as<Node>()->op()](inputs_shape);
      auto out_dtype = op_inferdtype[node->safe_as<Node>()->op()](inputs_dtype, node->safe_as<Node>()->attrs);
      int counter    = 0;
      CHECK_EQ(node->outlinks().size(), out_shape.size())
          << "The output number of node " << node->id() << " is " << node->outlinks().size()
          << " , which is different with the output shape size " << out_shape.size() << " . And the op type is "
          << node->safe_as<Node>()->op()->name;
      CHECK_EQ(node->outlinks().size(), out_dtype.size())
          << "The output number of node " << node->id() << " is " << node->outlinks().size()
          << " , which is different with the output dtype size " << out_dtype.size() << " . And the op type is "
          << node->safe_as<Node>()->op()->name;
      for (auto& out_edge : node->outlinks()) {
        shape_dict[out_edge->sink()->safe_as<NodeData>()->id()] = out_shape[counter];
        dtype_dict[out_edge->sink()->safe_as<NodeData>()->id()] = out_dtype[counter];
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
