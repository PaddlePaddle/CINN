#include "cinn/optim/rfactor_rewrite.h"
#include <gtest/gtest.h>
#include "cinn/cinn.h"

namespace cinn::optim {

TEST(RFactorRewrite, basic) {
  Expr M(16), N(32), K(48);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K, "k0");
  auto C = Compute(
      {M, N}, [=](Expr i, Expr j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  auto stages = CreateStages({C});

  auto [k_outer, k_inner] = stages[C]->Split(k->name, 16);
  stages[C]->RFactor(k_inner);

  auto fn = Expr(Lower("fn", stages, {A, B, C}));

  LOG(INFO) << fn;

  // RFactorRewrite(&fn, stages);

  LOG(INFO) << fn;
}

}  // namespace cinn::optim