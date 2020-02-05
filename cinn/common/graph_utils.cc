#include "cinn/common/graph_utils.h"
#include <glog/logging.h>
#include <functional>
#include <set>
#include <stack>

namespace cinn {
namespace common {

namespace {

void TopologicalSortUtil(GraphNode *node,
                         std::set<GraphNode *> *visited,
                         std::stack<GraphNode *> *stack,
                         std::vector<GraphNode *> *order,
                         std::vector<GraphEdge *> *edge_order) {
  node->VisitOnce();
  if (!node->visited()) return;
  CHECK(!visited->count(node)) << "duplicate visit current node";

  // Mark the current node as visited.
  visited->insert(node);
  order->push_back(node);

  for (auto &e : node->outlinks()) {
    if (!visited->count(e->sink())) {
      edge_order->push_back(e.get());
      TopologicalSortUtil(e->sink(), visited, stack, order, edge_order);
    }
  }

  stack->push(node);
}

std::tuple<Graph::node_order_t, Graph::edge_order_t> TopologicalSort(const std::vector<GraphNode *> &nodes) {
  std::stack<GraphNode *> stack;
  std::set<GraphNode *> visited;   // Tell whether a node is visited
  std::vector<GraphNode *> order;  // nodes visited in order
  std::vector<GraphEdge *> edges;  // edges visited in order

  for (auto *node : nodes) {
    if (!visited.count(node)) {
      TopologicalSortUtil(node, &visited, &stack, &order, &edges);
    }
  }
  return std::make_tuple(std::move(order), std::move(edges));
}

void DFSSortUtil(const GraphNode *node, std::vector<GraphNode *> *order) {}

std::vector<GraphNode *> DFSSort(const std::vector<GraphNode *> &nodes) {}

}  // namespace

std::vector<const GraphNode *> Graph::nodes() const {
  std::vector<const GraphNode *> res;
  for (auto &s : nodes_) res.push_back(s.get());
  return res;
}
std::vector<GraphNode *> Graph::nodes() {
  std::vector<GraphNode *> res;
  for (auto &s : nodes_) res.push_back(s.get());
  return res;
}

std::tuple<std::vector<GraphNode *>, std::vector<GraphEdge *>> Graph::topological_order() {
  return TopologicalSort(nodes());
}

std::vector<GraphNode *> Graph::dfs_order() { return std::vector<GraphNode *>(); }

std::vector<const GraphNode *> Graph::start_points() const {
  std::vector<const GraphNode *> res;
  for (auto *node : nodes()) {
    res.push_back(node);
  }
  return res;
}

void Graph::RegisterNode(size_t key, GraphNode *node) {
  registry_.emplace(key, node);
  nodes_.emplace_back(node);
}
void Graph::RegisterNode(const std::string &key, GraphNode *node) { RegisterNode(std::hash<std::string>{}(key), node); }

GraphNode *Graph::RetriveNode(size_t key) const {
  auto it = registry_.find(key);
  return it == registry_.end() ? nullptr : it->second;
}

GraphNode *Graph::RetriveNode(const std::string &key) const { return RetriveNode(std::hash<std::string>()(key)); }

}  // namespace common
}  // namespace cinn
