#include <gtest/gtest.h>
#include <any>
#include <string>
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
namespace cinn {
namespace hlir {
namespace framework {

void PrintGraphPass(Graph* src) {
  std::string res;
  auto store_node = std::get<0>(src->topological_order());
  int index       = 0;
  for (auto i : store_node) {
    if (i->id().length() < 8 || i->id().substr(0, 8) != "NodeData") {
      res += std::to_string(index) + ":";
      res += i->as<Node>()->attrs.node_name;
      res += "(" + i->id() + ")\n";
      index++;
    }
  }
  src->attrs["print_graph"] = std::make_shared<std::any>(res);
}

CINN_REGISTER_PASS(PrintGraph)
    .describe("This pass just save the visulization Graph to g.attrs[\"print_graph\"].")
    .set_change_structure(false)
    .provide_graph_attr("print_graph")
    .set_body(PrintGraphPass);

CINN_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

TEST(Operator, GetAttr) {
  Node* node0 = new Node(Operator::Get("add"), "elementwise_add", "Node_add0");
  std::shared_ptr<Node> node0_ptr(node0);
  NodeData* output0 = new NodeData(node0_ptr, 0, 0, "NodeData_add0_output0");
  Node* node1       = new Node(Operator::Get("add"), "elementwise_add", "Node_add1");
  std::shared_ptr<Node> node1_ptr(node1);
  NodeData* output1 = new NodeData(node1_ptr, 0, 0, "NodeData_add1_output0");
  node0->LinkTo(output0);
  output0->LinkTo(node1);
  node1->LinkTo(output1);
  Graph* g = new Graph;
  g->RegisterNode(0, node0);
  g->RegisterNode(1, output0);
  g->RegisterNode(2, node1);
  g->RegisterNode(3, output1);
  ApplyPass(g, "PrintGraph");
  auto s = g->GetAttr<std::string>("print_graph");
  LOG(INFO) << "0:elementwise_add(Node_add0)\n1:elementwise_add(Node_add1)\n";
  ASSERT_EQ(s, "0:elementwise_add(Node_add0)\n1:elementwise_add(Node_add1)\n");
  delete g;
  delete output1;
  delete output0;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
