#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace hlir {
namespace framework {

std::tuple<common::GraphEdge *, common::GraphEdge *> Node::LinkTo(NodeData *other) {
  return this->common::GraphNode::LinkTo(other->as<common::GraphNode>());
}

std::tuple<common::GraphEdge *, common::GraphEdge *> NodeData::LinkTo(Node *other) {
  return this->common::GraphNode::LinkTo(other->as<common::GraphNode>());
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
