#include "cinn/common/graph_utils.h"

#include <glog/logging.h>

#include <deque>
#include <functional>
#include <set>
#include <stack>

#include "cinn/utils/dot_lang.h"

namespace cinn {
namespace common {

namespace {

void DFSSortUtil(const GraphNode *node, std::vector<GraphNode *> *order) {}

std::vector<GraphNode *> DFSSort(const std::vector<GraphNode *> &nodes) {
  LOG(FATAL) << "not implemented";
  return {};
}

}  // namespace

std::set<GraphNode *> Graph::dependencies(const std::vector<GraphNode *> &targets) {
  // A naive implementation.
  std::set<GraphNode *> _targets(targets.begin(), targets.end());
  std::set<GraphNode *> res;
  int targets_count = 0;
  while (targets_count != _targets.size()) {
    targets_count = _targets.size();
    for (auto *node : nodes()) {
      if (_targets.count(node)) continue;
      for (auto &edge : node->outlinks()) {
        if (_targets.count(edge->sink())) {
          res.insert(edge->sink());
          _targets.insert(edge->sink());
        }
      }
    }
  }
  return res;
}

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
  std::vector<GraphNode *> node_order;
  std::vector<GraphEdge *> edge_order;

  std::deque<GraphNode *> queue;
  // collect indegreee.
  std::map<GraphNode *, int> indegree;
  for (auto *n : nodes()) {
    indegree[n] = n->inlinks().size();
  }

  // insert start points first.
  for (auto *n : start_points()) {
    queue.push_back(n);
  }

  // start to visit
  while (!queue.empty()) {
    auto *top_node = queue.front();
    queue.pop_front();
    node_order.push_back(top_node);

    for (auto &edge : top_node->outlinks()) {
      edge_order.push_back(edge.get());
      auto *sink = edge->sink();
      if (--indegree[sink] == 0) {
        queue.push_back(sink);
      }
    }
  }

  CHECK_EQ(node_order.size(), nodes().size());

  return std::make_tuple(node_order, edge_order);
}

std::vector<GraphNode *> Graph::dfs_order() { return std::vector<GraphNode *>(); }

std::vector<const GraphNode *> Graph::start_points() const {
  std::vector<const GraphNode *> res;
  for (auto *node : nodes()) {
    if (node->inlinks().empty()) res.push_back(node);
  }
  return res;
}

std::vector<GraphNode *> Graph::start_points() {
  std::vector<GraphNode *> res;
  for (auto *node : nodes()) {
    if (node->inlinks().empty()) res.push_back(node);
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

std::string Graph::Visualize() const {
  utils::DotLang dot;

  // 1. create nodes
  for (auto &node : nodes_) {
    dot.AddNode(node->id(), {});
  }

  // 2. link each other
  for (auto &source : nodes_) {
    for (auto &sink : source->outlinks()) {
      dot.AddEdge(source->id(), sink->sink()->id(), {});
    }
  }

  return dot();
}

}  // namespace common
}  // namespace cinn
