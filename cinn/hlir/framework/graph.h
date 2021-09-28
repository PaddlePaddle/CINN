#pragma once
#include <absl/container/flat_hash_map.h>
#include <memory>
#include <string>
#include <vector>

#include <absl/types/any.h>

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
  Graph(const frontend::Program& prog, const Target& target);

  Target target_;
  /** \brief outputs of the computation graph. */
  std::vector<NodeData*> outputs;

  /** \brief attributes of a graph */
  absl::flat_hash_map<std::string, std::shared_ptr<absl::any>> attrs;

  std::vector<std::vector<Node*>> groups;

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
  inline const T& GetAttrs(const std::string& attr_name) const {
    auto it = attrs.find(attr_name);
    CHECK(it != attrs.end()) << "Cannot find attribute [" << attr_name << "] in the graph";
    return absl::any_cast<const T&>(*it->second);
  }

  /**
   * \brief Get the mutable attribute from attrs.
   * @param attr_name the name of the attribute
   * @return the reference to corresponding attribute
   * @tparam T the type of the attribute.
   */
  template <typename T>
  inline T& GetMutableAttrs(const std::string& attr_name) {
    auto it = attrs.find(attr_name);
    CHECK(it != attrs.end()) << "Cannot find attribute [" << attr_name << "] in the graph";
    return absl::any_cast<T&>(*it->second);
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

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(Graph);
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
