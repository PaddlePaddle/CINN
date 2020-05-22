#pragma once
//! \file This file contains the utilities of graph.

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "cinn/common/object.h"
#include "cinn/common/shared.h"

namespace cinn {
namespace common {

#ifdef As
#undef As
#endif

class GraphNode;

/**
 * Edge in the graph, which can hold some attributes.
 */
class GraphEdge : public Object {
 public:
  GraphEdge(GraphNode* source, GraphNode* sink) : source_(source), sink_(sink) {}

  GraphNode* source() const { return source_; }
  GraphNode* sink() const { return sink_; }
  const char* type_info() const override { return "graph_edge"; }

 private:
  //! Source of this edge.
  GraphNode* source_{};
  //! End of this edge.
  GraphNode* sink_{};
};

}  // namespace common
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::common::Shared<cinn::common::GraphEdge>> {
  size_t operator()(const cinn::common::Shared<cinn::common::GraphEdge>& key) {
    return std::hash<std::string>()(std::to_string(reinterpret_cast<size_t>(key->source())) +
                                    std::to_string(reinterpret_cast<size_t>(key->sink())));
  }
};

}  // namespace std

namespace cinn {
namespace common {

/**
 * @brief The base class of all node of graph.
 * This is used to normalize and share the graph operations.
 */
class GraphNode : public Object {
 public:
  //! The unique identifier of the node.
  virtual std::string id() const = 0;

  //! Links from this to other.
  template <typename EdgeT = GraphEdge>
  std::tuple<EdgeT*, EdgeT*> LinkTo(GraphNode* other) {
    EdgeT *a, *b;
    CHECK(other);
    CHECK_NE(other, this) << "cannot link to itself";
    auto source_edge = make_shared<GraphEdge>(this, other);
    auto sink_edge   = make_shared<GraphEdge>(this, other);

    outlinks_.insert(source_edge);
    other->inlinks_.insert(sink_edge);

    for (auto& item : outlinks_) {
      if (item->sink()->id() == other->id()) {
        a = static_cast<EdgeT*>(item.get());
        break;
      }
    }
    for (auto& item : other->inlinks_) {
      if (item->sink()->id() == other->id()) {
        b = static_cast<EdgeT*>(item.get());
        break;
      }
    }
    return std::make_tuple(a, b);
  }

  //! Get the input links of the node.
  virtual std::set<Shared<GraphEdge>> inlinks() const { return inlinks_; }
  //! Get the output links of the node.
  virtual std::set<Shared<GraphEdge>> outlinks() const { return outlinks_; }
  //! Get a derived pointer.
  template <typename Derived>
  Derived* As() {
    static_assert(std::is_base_of<GraphNode, Derived>::value);
    return static_cast<Derived*>(this);
  }
  template <typename Derived>
  const Derived* As() const {
    static_assert(std::is_base_of<GraphNode, Derived>::value);
    return static_cast<const Derived*>(this);
  }

  //! Reset graph traversal meta info.
  void ResetVisitMeta() { visited_time_ = 0; }
  void VisitOnce() const { visited_time_++; }
  bool visited() const { return inlinks_.empty() || visited_time_ == inlinks_.size(); }

  const char* type_info() const override { return "GraphNode"; }

  GraphNode() = default;

 protected:
  //! The input links of the node.
  //! \note We record the raw pointer rather than the shared pointer to avoid cycle reference.
  std::set<common::Shared<GraphEdge>> inlinks_;
  //! The output links of the node.
  //! \note We record the raw pointer rather than the shared pointer to avoid cycle reference.
  std::set<common::Shared<GraphEdge>> outlinks_;

  mutable int visited_time_{};
};

/**
 * @brief The base class of all the graph.
 */
class Graph {
 public:
  using node_order_t = std::vector<GraphNode*>;
  using edge_order_t = std::vector<GraphEdge*>;

  //! Add a node to the graph.
  //! @{
  void RegisterNode(size_t key, GraphNode* node);
  void RegisterNode(const std::string& key, GraphNode* node);
  //! @}

  //! Retrive a node.
  //! @{
  GraphNode* RetriveNode(size_t key) const;
  GraphNode* RetriveNode(const std::string& key) const;
  //! @}

  //! Get the start point of the graph (the nodes those has no inlinks).
  std::vector<const GraphNode*> start_points() const;
  std::vector<GraphNode*> start_points();

  //! Return the graph's nodes and edges(visited) in topological order.
  std::tuple<std::vector<GraphNode*>, std::vector<GraphEdge*>> topological_order();

  //! Return the graph's DFS order.
  std::vector<GraphNode*> dfs_order();

  //! Return the dependency nodes of a set of nodes.
  std::set<GraphNode*> dependencies(const std::vector<GraphNode*>& nodes);

  std::vector<const GraphNode*> nodes() const;
  std::vector<GraphNode*> nodes();

  //! Get a string representation to visualize a graph.
  std::string Visualize() const;

  size_t num_nodes() const { return nodes_.size(); }

 protected:
  //! A lookup table that map from hash key to graph node, note that it doesn't own the graph node.
  std::map<size_t, GraphNode*> registry_;
  //! A list owns the graph nodes.
  std::vector<Shared<GraphNode>> nodes_;
};

}  // namespace common
}  // namespace cinn

namespace std {
template <>
struct hash<cinn::common::GraphNode> {
  size_t operator()(const cinn::common::GraphNode& x) { return reinterpret_cast<size_t>(hash<std::string>()(x.id())); }
};

}  // namespace std
