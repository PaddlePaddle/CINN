#pragma once
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cinn/common/graph_utils.h"
#include "cinn/hlir/op.h"

namespace cinn {
namespace hlir {
class Node;
class NodeData;

using NodePtr = std::shared_ptr<Node>;

/**
 * \brief Attributes of each node in graph.
 *  The attributes include the node's name, the corresponding operator
 *  and other parameters like axis.
 */
struct NodeAttr {
  /**
   * \brief The operator this node uses.
   */
  const Operator *op{nullptr};

  /**
   * \brief The name of this node.
   */
  std::string node_name;

  /**
   * \brief The attributes stored as string in dictionary.
   */
  std::unordered_map<std::string, std::string> attr_store;
};

/**
 * \brief Node represents an operation in a computation graph.
 */
class Node : public cinn::common::GraphNode {
 public:
  Node() = default;
  Node(const Operator *op, const std::string &name, std::string id = nullptr) {
    this->attrs.op        = op;
    this->attrs.node_name = name;
    this->id_             = std::move(id);
  }

  std::tuple<cinn::common::GraphEdge *, cinn::common::GraphEdge *> LinkTo(cinn::hlir::NodeData *other);
  /**
   * \brief Get the unique id of this NodeData.
   */
  std::string id() const override { return id_; }

  /**
   * \brief The attributes in the node.
   */
  NodeAttr attrs;

  inline const Operator *op() const { return this->attrs.op; }

  inline bool is_variable() { return (this->attrs.op == nullptr); }

  inline uint32_t num_outputs() { return is_variable() ? 1 : this->op()->num_outputs; }

  inline uint32_t num_inputs() { return is_variable() ? 1 : this->op()->num_inputs; }

  template <class... Args>
  static NodePtr Create(Args &&... args) {
    return std::make_shared<Node>(std::forward<Args>(args)...);
  }

 private:
  /**
   * \brief The unique id of the node.
   */
  std::string id_;
};

/**
 * \brief NodeData represents the output data from an operator.
 */
class NodeData : public cinn::common::GraphNode {
 public:
  NodeData(NodePtr node, uint32_t index, uint32_t version, std::string id)
      : source_node(std::move(node)), output_index(index), version(version), id_(std::move(id)) {}

  NodeData() : source_node(), output_index(), version(), id_() {}

  std::tuple<cinn::common::GraphEdge *, cinn::common::GraphEdge *> LinkTo(Node *other);
  static std::shared_ptr<NodeData> Create(
      const char *op_name,
      std::string node_name,
      std::vector<NodeData> inputs,
      std::string id                                     = nullptr,
      std::unordered_map<std::string, std::string> attrs = std::unordered_map<std::string, std::string>()) {
    auto res                           = std::make_shared<NodeData>();
    res->id_                           = std::move(id);
    res->source_node                   = Node::Create();
    res->source_node->attrs.op         = Operator::Get(op_name);
    res->source_node->attrs.node_name  = std::move(node_name);
    res->source_node->attrs.attr_store = attrs;
    return res;
  }

  /**
   * \brief Get the unique id of this NodeData.
   */
  std::string id() const override { return id_; }

  /**
   * \brief Source_node represents the operator this NodeData comes from.
   */
  NodePtr source_node;

  /**
   * \brief Output_index represents the index of this output data
   *  among all the outputs of the operator.
   *  For example, if an operator has 2 outputs, the index of
   *  the 2 NodeData should be 0 and 1.
   */
  uint32_t output_index;

  /**
   * \brief The version of input Variable.
   *  This field can only be nonzero when this->node is a Variable node.
   *  version is increased by one each time a Variable get composed to a mutation Op.
   */
  uint32_t version;

  /**
   * \brief The unique id of this NodeData.
   */
 private:
  std::string id_;
};

std::tuple<cinn::common::GraphEdge *, cinn::common::GraphEdge *> Node::LinkTo(NodeData *other) {
  return this->cinn::common::GraphNode::LinkTo(other->as<cinn::common::GraphNode>());
}

std::tuple<cinn::common::GraphEdge *, cinn::common::GraphEdge *> NodeData::LinkTo(Node *other) {
  return this->cinn::common::GraphNode::LinkTo(other->as<cinn::common::GraphNode>());
}

}  // namespace hlir
}  // namespace cinn
