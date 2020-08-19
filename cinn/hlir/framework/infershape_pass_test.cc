#include <gtest/gtest.h>

#include <any>
#include <string>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/ir/packed_func.h"

namespace cinn {
namespace hlir {
namespace framework {

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
  std::shared_ptr<Graph> g(new Graph(prog));
  ApplyPass(g.get(), "InferShape");
  auto dict                    = g->GetAttr<std::unordered_map<std::string, std::vector<int>>>("infershape");
  std::shared_ptr<Scope> scope = std::make_shared<Scope>();

  auto get_tensor = [&](const std::string& name) {
    auto* var    = scope->Var<Tensor>(name);
    auto& tensor = std::get<Tensor>(*var);
    return tensor;
  };

  for (auto iter : dict) {
    CHECK_EQ(iter.second[0], 100) << "The infered shape is wrong.";
    CHECK_EQ(iter.second[1], 32) << "The infered shape is wrong.";
    auto* var    = scope->Var<Tensor>(iter.first);
    auto& tensor = std::get<Tensor>(*var);
    std::vector<Shape::dim_t> shape;
    int product = 1;
    for (auto shape_dim : iter.second) {
      product *= shape_dim;
      shape.push_back(Shape::dim_t(shape_dim));
    }
    tensor.Resize(Shape{shape});
    auto* data = tensor.mutable_data<float>(common::DefaultHostTarget());
    for (size_t j = 0; j < product; j++) {
      unsigned int seed = j;
      data[j]           = (rand_r(&seed) * 1.f) / RAND_MAX;  // All 1.0 data
    }
  }
  GraphCompiler gc(common::Context::Global().GetTarget(), scope, g);
  std::unique_ptr<Program> program = gc.Build();
  program->Execute();

  auto xd = get_tensor("a").data<float>();
  auto yd = get_tensor("b").data<float>();
  auto zd = get_tensor(e->id).data<float>();

  for (int i = 0; i < 100 * 32; i++) {
    LOG_FIRST_N(INFO, 3) << "data: " << 2 * xd[i] << " + " << 3 * yd[i] << " = " << zd[i];
    ASSERT_NEAR(2 * xd[i] + 3 * yd[i], zd[i], 1e-5);
  }
}
}  // namespace framework

}  // namespace hlir
}  // namespace cinn
