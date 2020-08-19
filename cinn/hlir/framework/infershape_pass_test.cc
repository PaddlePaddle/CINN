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
#include "cinn/lang/packed_func.h"

namespace cinn {
namespace hlir {
namespace framework {

Tensor GetTensor(const std::shared_ptr<Scope>& scope, const std::string& name) {
  auto* var    = scope->Var<Tensor>(name);
  auto& tensor = std::get<Tensor>(*var);
  return tensor;
}

void SetRandData(Tensor* tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    unsigned int seed = j;
    data[j]           = (rand_r(&seed) * 1.f) / RAND_MAX;  // All random data
  }
}

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
  Target target                = common::Context::Global().GetTarget();
  std::shared_ptr<Scope> scope = BuildScope(target, g);
  GraphCompiler gc(target, scope, g);
  std::unique_ptr<Program> program = gc.Build();

  auto A = GetTensor(scope, "a");
  auto B = GetTensor(scope, "b");
  SetRandData(&A, target);
  SetRandData(&B, target);

  program->Execute();

  auto A_data = A.data<float>();
  auto B_data = B.data<float>();
  auto E_data = GetTensor(scope, e->id).data<float>();
  for (int i = 0; i < 100 * 32; i++) {
    LOG_FIRST_N(INFO, 3) << "data: " << 2 * A_data[i] << " + " << 3 * B_data[i] << " = " << E_data[i];
    ASSERT_NEAR(2 * A_data[i] + 3 * B_data[i], E_data[i], 1e-5);
  }
}
}  // namespace framework

}  // namespace hlir
}  // namespace cinn
