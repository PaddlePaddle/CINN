#include <gtest/gtest.h>

#include <any>
#include <string>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/lang/packed_func.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace framework {

void PrintGraphPass(Graph* src) {
  std::string res;
  auto store_node = std::get<0>(src->topological_order());
  int index       = 0;
  for (auto& i : store_node) {
    if (i->is_type<Node>()) {
      res += std::to_string(index) + ":";
      res += i->safe_as<Node>()->attrs.node_name;
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

TEST(Operator, GetAttrs) {
  frontend::Program prog;
  frontend::Variable a("A");
  frontend::Variable b("B");
  Type t  = Float(32);
  a->type = t;
  b->type = t;
  auto c  = prog.add(a, b);
  auto d  = prog.add(c, b);
  auto e  = prog.add(c, d);
  ASSERT_EQ(prog.size(), 3);
  Graph* g = new Graph(prog);
  ApplyPass(g, "PrintGraph");
  auto s = g->GetAttrs<std::string>("print_graph");
  LOG(INFO) << s;
  std::string target_str = R"ROC(
0:elementwise_add(elementwise_add_0)
1:elementwise_add(elementwise_add_1)
2:elementwise_add(elementwise_add_2)
)ROC";
  ASSERT_EQ(utils::Trim(s), utils::Trim(target_str));
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
