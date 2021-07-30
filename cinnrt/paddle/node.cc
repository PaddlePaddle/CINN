#include "cinnrt/paddle/node.h"

#include <algorithm>

namespace cinnrt {
namespace paddle {

std::tuple<common::GraphEdge*, common::GraphEdge*> Node::LinkTo(NodeData* other) {
  return this->common::GraphNode::LinkTo(other->as<common::GraphNode>());
}

std::tuple<common::GraphEdge*, common::GraphEdge*> NodeData::LinkTo(Node* other) {
  return this->common::GraphNode::LinkTo(other->as<common::GraphNode>());
}

namespace {

struct PyBindNodeAttrVisitor {
  std::stringstream& out;
  explicit PyBindNodeAttrVisitor(std::stringstream& out) : out(out) {}

  void operator()(int v) { out << "int: " << v; }
  void operator()(float v) { out << "float: " << v; }
  void operator()(bool v) { out << "bool: " << v; }
  void operator()(const std::string& v) { out << "string: " << v; }
#define VISIT_ELEMENTS(T__)                                      \
  void operator()(const std::vector<T__>& vs) {                  \
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

std::ostream& operator<<(std::ostream& os, const NodeAttr& node_attr) {
  std::stringstream ss;
  ss << "NodeAttr:\n";
  for (auto& item : node_attr.attr_store) {
    std::stringstream os;
    PyBindNodeAttrVisitor visitor(os);
    std::visit(visitor, item.second);
    ss << "- " << os.str() << "\n";
  }
  os << ss.str();
  return os;
}

//! Using index to sort the input/output tensors
bool edge_index_compare(const common::Shared<common::GraphEdge>& a, const common::Shared<common::GraphEdge>& b) {
  return a->index() < b->index();
}

const std::vector<common::Shared<common::GraphEdge>>& Node::inlinks_in_order() const {
  if (inlinks_in_order_.empty()) {
    for (auto& in_edge : this->inlinks()) {
      inlinks_in_order_.push_back(in_edge);
      CHECK_GE(in_edge->index(), 0) << "The index of a node's inlinks should be >= 0! Now index is: "
                                    << in_edge->index() << ". Please check.";
    }
    std::sort(inlinks_in_order_.begin(), inlinks_in_order_.end(), edge_index_compare);
  }
  return inlinks_in_order_;
}

const std::vector<common::Shared<common::GraphEdge>>& Node::outlinks_in_order() const {
  if (outlinks_in_order_.empty()) {
    for (auto& out_edge : this->outlinks()) {
      outlinks_in_order_.push_back(out_edge);
      CHECK_GE(out_edge->index(), 0) << "The index of a node's outlinks should be >= 0! Now index is: "
                                     << out_edge->index() << ". Please check.";
    }
    std::sort(outlinks_in_order_.begin(), outlinks_in_order_.end(), edge_index_compare);
  }
  return outlinks_in_order_;
}

}  // namespace paddle
}  // namespace cinnrt
