#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cinn/hlir/op.h"

namespace cinn {
namespace hlir {
class Node;
using NodePtr = std::shared_ptr<Node>;
// NodeAttr represents the additional attribute of a node.
struct NodeAttr {
  const Op *op{nullptr};
  std::string nodeName;
  std::unordered_map<std::string, std::string> attrStore;
};
// NodeData represents the output of each node.
struct NodeData {
  NodeData(NodePtr node, uint32_t index, uint32_t version)
      : sourceNode(std::move(node)), outputIndex(index), version(version) {}

  NodeData() : sourceNode(), outputIndex(), version() {}

  NodePtr sourceNode;
  int outputIndex;
  int version;
};

class Node {
 public:
  Node() = default;
  Node(const Op *op, const std::string &name) {
    this->attrs.op       = op;
    this->attrs.nodeName = name;
  }
  ~Node();
  NodeAttr attrs;
  std::vector<NodeData> inputs;

  inline const Op &op() const { return this->attrs.op; }
  inline is_variable() { return (this->attrs.op == nullptr); }
  inline int num_outputs() {
    if (is_variable())
      return 1;
    else
  }
};

}  // namespace hlir
}  // namespace cinn
