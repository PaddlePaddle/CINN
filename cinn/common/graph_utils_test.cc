#include <gtest/gtest.h>
#include "cinn/common/common.h"

namespace cinn {
namespace common {

struct GraphNodeWithName : public GraphNode {
  explicit GraphNodeWithName(std::string name) : name(name) {}

  std::string name;
};

TEST(Graph, basic) {
  // Create nodes: A, B, C, D, E
  Graph graph;

  auto* A = make_shared<GraphNodeWithName>("A");
  auto* B = make_shared<GraphNodeWithName>("B");
  auto* C = make_shared<GraphNodeWithName>("C");
  auto* D = make_shared<GraphNodeWithName>("D");
  auto* E = make_shared<GraphNodeWithName>("E");

  A->LinkTo(B);
  A->LinkTo(C);

  B->LinkTo(D);
  C->LinkTo(D);
  C->LinkTo(E);

  LOG(INFO) << "B: " << B->inlinks().size() << " -> " << B->outlinks().size();

  graph.RegisterNode("A", A);
  graph.RegisterNode("B", B);
  graph.RegisterNode("C", C);
  graph.RegisterNode("D", D);
  graph.RegisterNode("E", E);

  Graph::node_order_t node_order;
  Graph::edge_order_t edge_order;
  std::tie(node_order, edge_order) = graph.topological_order();

  for (auto* e : edge_order) {
    LOG(INFO) << "visit edge: " << e->source()->As<GraphNodeWithName>()->name << " -> "
              << e->sink()->As<GraphNodeWithName>()->name;
  }

  for (auto* n : node_order) {
    LOG(INFO) << "visit node: " << n->As<GraphNodeWithName>()->name;
  }
}

}  // namespace common
}  // namespace cinn
