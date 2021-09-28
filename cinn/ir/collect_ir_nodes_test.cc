#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace ir {

TEST(CollectIRNodes, basic0) {
  Expr C = Expr(1) + 2;

  auto exprs = CollectIRNodes(C, [](const Expr* x) { return x->As<ir::Add>(); });
  ASSERT_EQ(exprs.size(), 1UL);

  auto ints = CollectIRNodes(C, [](const Expr* x) { return x->As<ir::IntImm>(); });
  ASSERT_EQ(ints.size(), 2UL);
}

TEST(CollectIRNodes, basic) {
  Expr M(100);
  Expr N(200);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  auto fn = Lower("fn", stages, {A, B, C});

  LOG(INFO) << "fn:\n" << fn;

  auto tensors = CollectIRNodes(fn, [](const Expr* x) { return x->as_tensor(); });
  ASSERT_EQ(tensors.size(), 5UL);

  auto fn_body = fn.As<ir::_LoweredFunc_>()->body;
  LOG(INFO) << "fn.body:\n" << fn_body;
  auto tensors2 = CollectIRNodes(fn_body, [](const Expr* x) { return x->as_tensor(); });
  auto exprs    = CollectIRNodes(fn_body, [](const Expr* x) { return x; });
}

}  // namespace ir
}  // namespace cinn
