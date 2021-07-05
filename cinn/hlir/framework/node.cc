#include "cinn/hlir/framework/node.h"

#include <algorithm>

#include "cinn/common/context.h"

namespace cinn {
namespace hlir {
namespace framework {

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

const std::vector<common::Shared<common::GraphEdge>>& Node::inlinks_in_order(bool refresh) const {
  if (inlinks_in_order_.empty() || refresh) {
    if (refresh) {
      inlinks_in_order_.clear();
    }
    for (auto& in_edge : this->inlinks()) {
      inlinks_in_order_.push_back(in_edge);
      CHECK_GE(in_edge->index(), 0) << "The index of a node's inlinks should be >= 0! Now index is: "
                                    << in_edge->index() << ". Please check.";
    }
    std::sort(inlinks_in_order_.begin(), inlinks_in_order_.end(), edge_index_compare);
  }
  return inlinks_in_order_;
}

const std::vector<common::Shared<common::GraphEdge>>& Node::outlinks_in_order(bool refresh) const {
  if (outlinks_in_order_.empty() || refresh) {
    if (refresh) {
      outlinks_in_order_.clear();
    }
    for (auto& out_edge : this->outlinks()) {
      outlinks_in_order_.push_back(out_edge);
      CHECK_GE(out_edge->index(), 0) << "The index of a node's outlinks should be >= 0! Now index is: "
                                     << out_edge->index() << ". Please check.";
    }
    std::sort(outlinks_in_order_.begin(), outlinks_in_order_.end(), edge_index_compare);
  }
  return outlinks_in_order_;
}

void ReplaceGraphOpNode(common::Graph* graph, Node* old_node, Node* new_node, int new_out_nums) {
  CHECK(graph);
  CHECK(old_node);
  CHECK(new_node);
  auto old_inlinks  = old_node->inlinks_in_order(true);
  auto old_outlinks = old_node->outlinks_in_order(true);
  for (auto& link : old_inlinks) {
    auto* source = link->source();
    source->LinkTo(new_node);
    source->UnLinkTo(old_node);
  }
  std::shared_ptr<Node> node_ptr(new_node);
  // unlink and delete old outnodes except the first one
  for (int i = 0; i < old_outlinks.size(); i++) {
    auto* sink = old_outlinks[i]->sink();
    old_node->UnLinkTo(sink);
    if (i == 0) {
      new_node->as<common::GraphNode>()->LinkTo(sink);
    } else {
      // graph->DropNode(sink);
    }
  }
  for (int i = 1; i < new_out_nums; i++) {
    auto* new_out = new NodeData(node_ptr, 0, 0, common::UniqName(new_node->id() + "_out_" + std::to_string(i)));
    graph->RegisterNode(new_out->id(), new_out);
    new_node->as<common::GraphNode>()->LinkTo(new_out);
  }
  graph->DropNode(old_node);
  graph->RegisterNode(new_node->id(), new_node);
}

NodeData* InsertGraphOpNode(common::Graph* graph, Node* insert_node, NodeData* input_nodedata, Node* out_node) {
  CHECK(graph);
  CHECK(insert_node);
  CHECK(input_nodedata);
  input_nodedata->LinkTo(insert_node);
  std::shared_ptr<Node> node_ptr(insert_node);
  auto* out_nodedata = new NodeData(node_ptr, 0, 0, common::UniqName(insert_node->id() + "_out"));
  insert_node->LinkTo(out_nodedata);
  if (out_node) {
    out_nodedata->LinkTo(out_node);
    input_nodedata->UnLinkTo(out_node);
  }
  graph->RegisterNode(insert_node->id(), insert_node);
  graph->RegisterNode(out_nodedata->id(), out_nodedata);
  return out_nodedata;
}

void DeleteGraphOpNode(common::Graph* graph, Node* op_node, NodeData* input_nodedata, NodeData* out_nodedata) {
  CHECK(graph);
  CHECK(op_node);
  CHECK(input_nodedata);
  CHECK(out_nodedata);
  input_nodedata->UnLinkTo(op_node);
  op_node->UnLinkTo(out_nodedata);
  graph->DropNode(op_node);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
