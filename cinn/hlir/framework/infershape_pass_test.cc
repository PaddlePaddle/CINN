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

std::vector<std::vector<int>> AddInferShape(std::vector<std::vector<int>> inputs_shape) {
  CHECK(inputs_shape.size() && inputs_shape[0].size()) << "The input's shape size is 0! Please check again.";
  std::vector<std::vector<int>> res{inputs_shape[0]};
  return res;
}

CINN_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_attr<std::function<std::vector<std::vector<int>>(std::vector<std::vector<int>>)>>("infer_shape", AddInferShape)
    .set_support_level(4);

void InferShapePass(Graph* src) {
  auto res        = src->GetAttr<std::unordered_map<std::string, std::vector<int>>>("infer_shape");
  auto store_node = std::get<0>(src->topological_order());
  auto op_infershape =
      Operator::GetAttr<std::function<std::vector<std::vector<int>>(std::vector<std::vector<int>>)>>("infer_shape");
  for (auto i : store_node) {
    if (i->check_type<Node>()) {
      std::vector<std::vector<int>> inputs_shape;
      for (auto j : i->inlinks()) {
        inputs_shape.push_back(res[j->source()->safe_as<NodeData>()->id()]);
      }
      auto out_shape = op_infershape[i->safe_as<Node>()->op()](inputs_shape);
      int counter    = 0;
      CHECK_EQ(i->outlinks().size(), out_shape.size())
          << "The output number of node " << i->id() << " is " << i->outlinks().size()
          << " , which is different with the output shape size " << out_shape.size() << " . And the op type is "
          << i->safe_as<Node>()->op()->name;
      for (auto j : i->outlinks()) {
        res[j->sink()->safe_as<NodeData>()->id()] = out_shape[counter++];
      }
    }
  }
  src->attrs["infer_shape"] = std::make_shared<std::any>(res);
}

CINN_REGISTER_PASS(InferShape)
    .describe("This pass infer the shape of tensor and save to g.attrs[\"infer_shape\"].")
    .set_change_structure(false)
    .provide_graph_attr("infer_shape")
    .set_body(InferShapePass);

TEST(Operator, GetAttr) {
  frontend::Program prog;
  // TODO(Superjomn) Replace with Placeholder here.
  frontend::Variable a("a");
  frontend::Variable b("b");
  a->shape = {100, 32};
  b->shape = {100, 32};
  auto c   = prog.add(a, b);
  auto d   = prog.add(c, b);
  auto e   = prog.add(c, d);
  ASSERT_EQ(prog.size(), 3UL);
  std::unique_ptr<Graph> g(new Graph(prog));
  ApplyPass(g.get(), "InferShape");
  auto s = g->GetAttr<std::unordered_map<std::string, std::vector<int>>>("infer_shape");
  for (auto i : s) {
    LOG(INFO) << "Var id is: " << i.first << " and Var shape is: ";
    for (auto j : i.second) {
      LOG(INFO) << j << " ";
    }
    CHECK_EQ(i.second[0], 100) << "The infered shape is wrong.";
    CHECK_EQ(i.second[1], 32) << "The infered shape is wrong.";
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
