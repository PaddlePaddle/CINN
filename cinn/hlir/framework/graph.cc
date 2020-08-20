#include "cinn/hlir/framework/graph.h"

namespace cinn {
namespace hlir {
namespace framework {

Graph::Graph(frontend::Program prog) {
  std::unordered_map<std::string, std::vector<int>> shape_dict;
  std::unordered_map<std::string, common::Type> dtype_dict;
  int counter = 0;
  for (size_t i = 0; i < prog.size(); i++) {
    auto temp = prog[i];
    Node* node_tmp =
        new Node(Operator::Get(temp->op_type), temp->op_type, temp->op_type + "_" + std::to_string(counter++));
    std::shared_ptr<Node> node_ptr(node_tmp);
    node_tmp->attrs.attr_store = temp->attrs;
    for (frontend::Variable input_v : temp->inputs) {
      NodeData* input_data = this->RetriveNode(input_v->id)->as<NodeData>();
      if (!input_data) {
        dtype_dict[input_v->id] = input_v->type;
        shape_dict[input_v->id] = input_v->shape;
        input_data              = new NodeData(nullptr, 0, 0, input_v->id);
        input_data->LinkTo(node_tmp);
        this->RegisterNode(input_v->id, input_data);
      } else {
        input_data->LinkTo(node_tmp);
      }
    }
    for (frontend::Variable output_v : temp->outputs) {
      int out_idx           = 0;
      NodeData* output_data = new NodeData(node_ptr, out_idx++, 0, output_v->id);
      node_tmp->LinkTo(output_data);
      this->RegisterNode(output_v->id, output_data);
    }
    this->RegisterNode(node_tmp->id(), node_tmp);
  }
  this->attrs["infershape"] = std::make_shared<std::any>(shape_dict);
  this->attrs["inferdtype"] = std::make_shared<std::any>(dtype_dict);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
