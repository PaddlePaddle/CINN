
#include "cinnrt/common/graph_utils.h"

#include <glog/logging.h>

#include <deque>
#include <functional>
#include <set>
#include <stack>

#include "cinnrt/common/common.h"

//#include "cinnrt/cinn/dot_lang.h"

namespace cinnrt {
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

std::tuple<std::vector<GraphNode *>, std::vector<GraphEdge *>> Graph::topological_order() const {
  std::vector<GraphNode *> node_order;
  std::vector<GraphEdge *> edge_order;
  std::deque<GraphNode *> queue;

  // collect indegreee.
  std::map<std::string, int> indegree;
  for (auto *n : nodes()) {
    indegree[n->id()] = n->inlinks().size();
  }

  // insert start points first.
  for (auto *n : start_points()) {
    queue.push_back(&Reference(n));
  }

  // start to visit
  while (!queue.empty()) {
    auto *top_node = queue.front();
    node_order.push_back(top_node);

    queue.pop_front();

    for (auto &edge : top_node->outlinks()) {
      CHECK_EQ(edge->source(), top_node);
      edge_order.push_back(edge.get());
      auto *sink = edge->sink();
      if ((--indegree[sink->id()]) == 0) {
        queue.push_back(sink);
      }
    }
  }

  CHECK_EQ(node_order.size(), nodes().size()) << "circle detected in the schedule graph:\n\n" << Visualize();

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

GraphNode *Graph::RegisterNode(size_t key, GraphNode *node) {
  registry_.emplace(key, node);
  nodes_.emplace_back(node);
  return node;
}

GraphNode *Graph::RegisterNode(const std::string &key, GraphNode *node) {
  return RegisterNode(std::hash<std::string>{}(key), node);
}

GraphNode *Graph::RetrieveNode(size_t key) const {
  auto it = registry_.find(key);
  return it == registry_.end() ? nullptr : it->second;
}

GraphNode *Graph::RetrieveNode(const std::string &key) const { return RetrieveNode(std::hash<std::string>()(key)); }

// std::string Graph::Visualize() const {
//    cinnrt::cinn::DotLang dot;
//
//  // 1. create nodes
//  for (auto &node : nodes_) {
//    dot.AddNode(node->id(), {});
//  }
//
//  // 2. link each other
//  for (auto &source : nodes_) {
//    for (auto &sink : source->outlinks()) {
//      dot.AddEdge(source->id(), sink->sink()->id(), {});
//    }
//  }
//
//  return dot();
//}

const char *GraphNode::__type_info__ = "GraphNode";

bool GraphEdgeCompare::operator()(const Shared<GraphEdge> &a, const Shared<GraphEdge> &b) const {
  if (a->source()->id() == b->source()->id()) {
    return a->sink()->id() > b->sink()->id();
  }
  return a->source()->id() < b->source()->id();
}

std::set<GraphNode *> Graph::CollectNodes(std::function<bool(const common::GraphNode *)> &&teller) {
  std::set<GraphNode *> res;
  for (auto *node : nodes()) {
    if (teller(node)) res.insert(node);
  }
  return res;
}

}  // namespace common
}  // namespace cinnrt
