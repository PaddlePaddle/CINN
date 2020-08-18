#include <gtest/gtest.h>

#include <any>
#include <string>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/ir/packed_func.h"
namespace cinn {
namespace hlir {
namespace framework {

CINN_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

void PrintGraphPass(Graph* src) {
  std::string res;
  auto store_node = std::get<0>(src->topological_order());
  int index       = 0;
  for (auto i : store_node) {
    if (i->type_info() == "hlir_framework_node") {
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

TEST(Operator, GetAttr) {
  frontend::Program prog;
  frontend::Variable a("a");
  frontend::Variable b("b");
  auto c = prog.add(a, b);
  auto d = prog.add(c, b);
  auto e = prog.add(c, d);
  ASSERT_EQ(prog.size(), 3);
  Graph* g = new Graph(prog);
  ApplyPass(g, "PrintGraph");
  auto s = g->GetAttr<std::string>("print_graph");
  LOG(INFO) << s;
  ASSERT_EQ(s, "0:add(add_0)\n1:add(add_1)\n2:add(add_2)\n");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
