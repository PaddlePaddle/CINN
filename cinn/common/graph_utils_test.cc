#include <gtest/gtest.h>

#include "cinn/common/common.h"

namespace cinn {
namespace common {

struct GraphNodeWithName : public GraphNode {
  explicit GraphNodeWithName(std::string name) : name(name) {}

  std::string id() const override { return name; }

  std::string name;
};

// A simple graph.
std::unique_ptr<Graph> CreateGraph0() {
  std::unique_ptr<Graph> graph(new Graph);

  auto* A = make_shared<GraphNodeWithName>("A");
  auto* B = make_shared<GraphNodeWithName>("B");
  auto* C = make_shared<GraphNodeWithName>("C");
  auto* D = make_shared<GraphNodeWithName>("D");
  auto* E = make_shared<GraphNodeWithName>("E");

  graph->RegisterNode("A", A);
  graph->RegisterNode("B", B);
  graph->RegisterNode("C", C);
  graph->RegisterNode("D", D);
  graph->RegisterNode("E", E);

  A->LinkTo(B);
  A->LinkTo(C);

  B->LinkTo(D);
  C->LinkTo(D);
  C->LinkTo(E);

  return graph;
}

std::unique_ptr<Graph> CreateGraph1() {
  std::unique_ptr<Graph> graph(new Graph);

  auto* A = make_shared<GraphNodeWithName>("A");
  auto* B = make_shared<GraphNodeWithName>("B");

  graph->RegisterNode("A", A);
  graph->RegisterNode("B", B);

  B->LinkTo(A);

  return graph;
}

TEST(Graph, basic) {
  // Create nodes: A, B, C, D, E
  auto graph = CreateGraph0();

  Graph::node_order_t node_order;
  Graph::edge_order_t edge_order;
  std::tie(node_order, edge_order) = graph->topological_order();

  std::vector<std::string> order({"A", "B", "C", "D", "E"});

  for (auto* e : edge_order) {
    LOG(INFO) << "visit edge: " << e->source()->As<GraphNodeWithName>()->name << " -> "
              << e->sink()->As<GraphNodeWithName>()->name;
  }

  for (auto* n : node_order) {
    LOG(INFO) << "visit node: " << n->As<GraphNodeWithName>()->name;
  }

  for (int i = 0; i < node_order.size(); i++) {
    EXPECT_EQ(node_order[i]->As<GraphNodeWithName>()->name, order[i]);
  }
}

TEST(Graph, Visualize) {
  auto graph = CreateGraph0();
  LOG(INFO) << "graph:\n" << graph->Visualize();
}

TEST(Graph, simple) {
  auto graph = CreateGraph1();
  Graph::node_order_t node_order;
  Graph::edge_order_t edge_order;
  std::tie(node_order, edge_order) = graph->topological_order();

  LOG(INFO) << "graph1 " << graph->Visualize();

  std::vector<GraphNode*> node_order_target({graph->RetriveNode("B"), graph->RetriveNode("A")});

  ASSERT_EQ(node_order.size(), node_order_target.size());
  for (int i = 0; i < node_order.size(); i++) {
    EXPECT_EQ(node_order[i]->id(), node_order_target[i]->id());
  }
}

}  // namespace common
}  // namespace cinn
