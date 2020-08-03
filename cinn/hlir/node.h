#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cinn/common/graph_utils.h"
#include "cinn/hlir/op.h"

namespace cinn {
namespace hlir {
class Node;
using NodePtr = std::shared_ptr<Node>;
// NodeAttr represents the additional attribute of a node.
struct NodeAttr {
  const Operator *op{nullptr};
  std::string node_name;
  std::unordered_map<std::string, std::string> attr_store;
};
// NodeData represents the output of each node.
class NodeData : public cinn::common::GraphNode {
  NodeData(NodePtr node, uint32_t index, uint32_t version)
      : source_node(std::move(node)), output_index(index), version(version) {}

  NodeData() : source_node(), output_index(), version() {}
  std::string id() { return node_id; }
  NodePtr source_node;
  uint32_t output_index;
  uint32_t version;
  std::string node_id;
};

class Node : public cinn::common::GraphNode {
 public:
  Node() = default;
  Node(const Operator *op, const std::string &name) {
    this->attrs.op        = op;
    this->attrs.node_name = name;
  }
  ~Node();
  std::string id() { return node_id; }
  // The attributes of this node
  NodeAttr attrs;

  std::string node_id;

  inline const Operator *op() const { return this->attrs.op; }
  inline bool is_variable() { return (this->attrs.op == nullptr); }
  inline uint32_t num_outputs() {
    if (is_variable())
      return 1;
    else
      return this->op()->num_outputs;
  }
  inline uint32_t num_inputs() {
    if (is_variable())
      return 1;
    else
      return this->op()->num_inputs;
  }
  template <class... Args>
  static NodePtr Create(Args &&... args) {
    return std::make_shared<Node>(std::forward<Args>(args)...);
  }
};

inline NodeData MakeNode(
    const char *op_name,
    std::string node_name,
    std::vector<NodeData> inputs,
    std::unordered_map<std::string, std::string> attrs = std::unordered_map<std::string, std::string>()) {
  NodePtr p           = Node::Create();
  p->attrs.op         = Operator::Get(op_name);
  p->attrs.node_name  = std::move(node_name);
  p->attrs.attr_store = attrs;
  return NodeData(p, 0, 0);
}

}  // namespace hlir
}  // namespace cinn
