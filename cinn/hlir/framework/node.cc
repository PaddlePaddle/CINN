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

namespace {

struct PyBindNodeAttrVisitor {
  std::stringstream &out;
  PyBindNodeAttrVisitor(std::stringstream &out) : out(out) {}

  void operator()(int v) { out << "int: " << v; }
  void operator()(float v) { out << "float: " << v; }
  void operator()(bool v) { out << "bool: " << v; }
  void operator()(const std::string &v) { out << "string: " << v; }
#define VISIT_ELEMENTS(T__)                                      \
  void operator()(const std::vector<T__> &vs) {                  \
    if (vs.empty()) return;                                      \
    for (int i = 0; i < vs.size() - 1; i++) out << vs[i] << ","; \
    out << vs.back();                                            \
  }
  VISIT_ELEMENTS(int)
  VISIT_ELEMENTS(float)
  VISIT_ELEMENTS(bool)
  VISIT_ELEMENTS(std::string)
};

}  // namespace

std::ostream &operator<<(std::ostream &os, const NodeAttr &node_attr) {
  std::stringstream ss;
  ss << "NodeAttr:\n";
  for (auto &item : node_attr.attr_store) {
    std::stringstream os;
    PyBindNodeAttrVisitor visitor(os);
    std::visit(visitor, item.second);
    ss << "- " << os.str() << "\n";
  }
  os << ss.str();
  return os;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
