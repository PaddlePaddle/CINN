#include "cinn/hlir/framework/graph.h"

namespace cinn {
namespace hlir {
namespace framework {

Graph::Graph(frontend::Program prog) {
  std::unordered_map<std::string, std::vector<int>> res;
  int counter = 0;
  for (size_t i = 0; i < prog.size(); i++) {
    auto temp = prog[i];
    Node* node_tmp =
        new Node(Operator::Get(temp->op_type), temp->op_type, temp->op_type + "_" + std::to_string(counter++));
    std::shared_ptr<Node> node_ptr(node_tmp);
    node_tmp->attrs.attr_store = temp->attrs;
    for (frontend::Variable j : temp->inputs) {
      NodeData* input_data = this->RetriveNode(j->id)->as<NodeData>();
      if (!input_data) {
        res[j->id] = j->shape;
        input_data = new NodeData(nullptr, 0, 0, j->id);
        input_data->LinkTo(node_tmp);
        this->RegisterNode(j->id, input_data);
      } else {
        input_data->LinkTo(node_tmp);
      }
    }
    for (frontend::Variable j : temp->outputs) {
      int out_idx           = 0;
      NodeData* output_data = new NodeData(node_ptr, out_idx++, 0, j->id);
      node_tmp->LinkTo(output_data);
      this->RegisterNode(j->id, output_data);
    }
    this->RegisterNode(node_tmp->id(), node_tmp);
  }
  this->attrs["infer_shape"] = std::make_shared<std::any>(res);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
