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
#include "cinn/runtime/cpu/host_intrinsics.h"
#include "cinn/runtime/cpu/use_extern_funcs.h"

namespace cinn {
namespace hlir {
namespace framework {

Tensor GetTensor(const std::shared_ptr<Scope>& scope, const std::string& name) {
  auto* var    = scope->Var<Tensor>(name);
  auto& tensor = std::get<Tensor>(*var);
  return tensor;
}

void SetRandData(Tensor tensor, Target target) {
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    unsigned int seed = j;
    data[j]           = (rand_r(&seed) * 1.f) / RAND_MAX;  // All random data
  }
}

TEST(Operator, GetAttrs) {
  frontend::Program prog;
  // TODO(Superjomn) Replace with Placeholder here.
  frontend::Variable a("A");
  frontend::Variable b("B");
  Type t   = Float(32);
  a->shape = {1, 3, 224, 224};
  b->shape = {1, 3, 224, 224};
  a->type  = t;
  b->type  = t;
  auto c   = prog.add(a, b);
  auto d   = prog.relu(c);

  frontend::Variable e("E");
  e->shape = {32, 3, 3, 3};
  e->type  = t;

  std::unordered_map<std::string, NodeAttr::attr_t> attr1;
  attr1["stride"]   = std::vector<int>({2, 2});
  attr1["padding"]  = std::vector<int>({1, 1});
  attr1["dilation"] = static_cast<int>(1);
  attr1["epsilon"]  = 0.00001f;

  auto A = GetTensor(scope, "A");
  auto B = GetTensor(scope, "B");
  SetRandData(A, target);
  SetRandData(B, target);

  frontend::Variable g("G");
  g->shape = {4, 32};
  g->type  = t;

  auto f = prog.conv2d(d, e, attr2);
  auto h = prog.batchnorm(f, g, attr1);
  ASSERT_EQ(prog.size(), 4UL);

  auto graph = std::make_shared<Graph>(prog);
  ApplyPass(graph.get(), "InferShape");

  auto A_data = A->data<float>();
  auto B_data = B->data<float>();
  auto E_data = GetTensor(scope, e->id)->data<float>();
  for (int i = 0; i < 100 * 32; i++) {
    LOG_FIRST_N(INFO, 3) << "data: " << 2 * A_data[i] << " + " << 3 * B_data[i] << " = " << E_data[i];
    ASSERT_NEAR(2 * A_data[i] + 3 * B_data[i], E_data[i], 1e-5);
  }
}
}  // namespace framework

}  // namespace hlir
}  // namespace cinn
