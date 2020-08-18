#pragma once
#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/common/graph_utils.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph : public cinn::common::Graph {
 public:
  explicit Graph(frontend::Program prog) {
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

  /** \brief outputs of the computation graph. */
  std::vector<NodeData*> outputs;

  /** \brief attributes of a graph */
  std::unordered_map<std::string, std::shared_ptr<std::any>> attrs;

  void RegisterNode(size_t key, Node* node) { this->common::Graph::RegisterNode(key, node->as<common::GraphNode>()); }
  void RegisterNode(size_t key, NodeData* node) {
    this->common::Graph::RegisterNode(key, node->as<common::GraphNode>());
  }
  void RegisterNode(const std::string& key, Node* node) {
    this->common::Graph::RegisterNode(key, node->as<common::GraphNode>());
  }
  void RegisterNode(const std::string& key, NodeData* node) {
    this->common::Graph::RegisterNode(key, node->as<common::GraphNode>());
  }

  /**
   * \brief Get the immutable attribute from attrs.
   * @param attr_name the name of the attribute
   * @return the reference to corresponding attribute
   * @tparam T the type of the attribute.
   */
  template <typename T>
  inline const T& GetAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    CHECK(it != attrs.end()) << "Cannot find attribute [" << attr_name << "] in the graph";
    return std::any_cast<const T&>(*it->second);
  }

  /**
   * \brief Check whether has a specific attribute.
   * @param attr_name the name of the attribute
   * @return a boolean result
   */
  inline bool HasAttr(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    return it != attrs.end();
  }
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
