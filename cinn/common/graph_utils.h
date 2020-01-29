#pragma once
//! \file This file contains the utilities of graph.

#include <list>
#include <vector>

#include "cinn/common/shared.h"

namespace cinn {
namespace common {

/**
 * @brief The base class of all node of graph.
 * This is used to normalize and share the graph operations.
 */
class GraphNode {
 public:
  GraphNode() = default;

  //! Get the input links of the node.
  virtual std::list<GraphNode*> inlinks() const { return inlinks_; }
  //! Get the output links of the node.
  virtual std::list<GraphNode*> outlinks() const { return outlinks_; }

 protected:
  //! The input links of the node.
  //! \note We record the raw pointer rather than the shared pointer to avoid cycle reference.
  std::list<GraphNode*> inlinks_;
  //! The output links of the node.
  //! \note We record the raw pointer rather than the shared pointer to avoid cycle reference.
  std::list<GraphNode*> outlinks_;
};

/**
 * @brief The base class of all the graph.
 */
class Graph {
 public:
  size_t num_nodes() const { return nodes_.size(); }

  //! Return the graph's topological order.
  std::vector<GraphNode*> topological_order() const;

  //! Return the graph's DFS order.
  std::vector<GraphNode*> dfs_order() const;

 protected:
  std::vector<Shared<GraphNode>> nodes_;
};

}  // namespace common
}  // namespace cinn
