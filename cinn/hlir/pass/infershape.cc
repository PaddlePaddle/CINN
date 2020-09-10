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

void InferShapePass(Graph* graph) {
  auto& shape_dict    = graph->GetMutableAttrs<std::unordered_map<std::string, std::vector<int>>>("infershape");
  auto& dtype_dict    = graph->GetMutableAttrs<std::unordered_map<std::string, Type>>("inferdtype");
  auto store_nodes    = std::get<0>(graph->topological_order());
  auto& op_infershape = Operator::GetAttrs<
      std::function<std::vector<std::vector<int>>(const std::vector<std::vector<int>>&, const framework::NodeAttr&)>>(
      "infershape");
  auto& op_inferdtype =
      Operator::GetAttrs<std::function<std::vector<Type>(const std::vector<Type>&, const framework::NodeAttr&)>>(
          "inferdtype");

  for (auto& n : store_nodes) {
    auto node = n->safe_as<Node>();
    if (node) {
      std::vector<std::vector<int>> inputs_shape;
      std::vector<Type> inputs_dtype;
      for (auto& in_edge : node->inlinks_in_order()) {
        auto* source_node = in_edge->source()->safe_as<NodeData>();
        CHECK(source_node);
        CHECK(shape_dict.count(source_node->id())) << "No shape for " << source_node->id();
        CHECK(dtype_dict.count(source_node->id())) << "No dtype for " << source_node->id();
        inputs_shape.push_back(shape_dict[source_node->id()]);
        inputs_dtype.push_back(dtype_dict[source_node->id()]);
      }

      auto out_shape = op_infershape[node->safe_as<Node>()->op()](inputs_shape, node->safe_as<Node>()->attrs);
      auto out_dtype = op_inferdtype[node->safe_as<Node>()->op()](inputs_dtype, node->safe_as<Node>()->attrs);
      CHECK_EQ(node->outlinks_in_order().size(), out_shape.size())
          << "The output number of node " << node->id() << " is " << node->outlinks_in_order().size()
          << " , which is different with the output shape size " << out_shape.size() << " . And the op type is "
          << node->safe_as<Node>()->op()->name;
      CHECK_EQ(node->outlinks_in_order().size(), out_dtype.size())
          << "The output number of node " << node->id() << " is " << node->outlinks_in_order().size()
          << " , which is different with the output dtype size " << out_dtype.size() << " . And the op type is "
          << node->safe_as<Node>()->op()->name;

      int counter = 0;
      for (auto& out_edge : node->outlinks_in_order()) {
        auto* sink_node = out_edge->sink()->safe_as<NodeData>();
        CHECK(sink_node);

        VLOG(3) << "Infershape: " << sink_node->id() << " " << utils::Join(out_shape[counter], ",");
        shape_dict[sink_node->id()] = out_shape[counter];
        dtype_dict[sink_node->id()] = out_dtype[counter];
        counter++;
      }
    }
  }
  graph->attrs["infershape"] = std::make_shared<std::any>(shape_dict);
  graph->attrs["inferdtype"] = std::make_shared<std::any>(dtype_dict);
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

  return true;
}
