#include "cinn/ir/tensor.h"

#include <gtest/gtest.h>

#include "cinn/backends/codegen_c.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/cinn.h"
#include "cinn/common/test_helper.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/packed_func.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace ir {
using utils::GetStreamCnt;
using utils::Trim;

TEST(Tensor, inlined) {
  Expr M(100), N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C is inlined
  Tensor C = lang::Compute(
      {M, N}, [=](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  Tensor D = lang::Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return C(i, j) * 2.f + 1.f; }, "D");

  auto stages = CreateStages({D});
  stages[C]->ComputeInline();

  auto funcs = lang::Lower("func_C", stages, {A, B, D});
  std::cout << "output: \n" << funcs << std::endl;
  auto out = GetStreamCnt(funcs);
  EXPECT_EQ(Trim(out), Trim(R"ROC(
function func_C (_A, _B, _D)
{
  for (i, 100)
  {
    for (j, 20)
    {
      D[i, j] = (1 + ((2 * A[i, j]) + (2 * B[i, j])))
    }
  }
}
)ROC"));
}

TEST(Tensor, IsDependOnStatement) {
  Expr N(100);

  Placeholder<float> X("X", {N});
  auto t = Compute({N}, [&](Var i) -> Expr { return X(i); });

  ASSERT_TRUE(t->IsDependOnStatement("X"));
  ASSERT_FALSE(t->IsDependOnStatement("XXX"));
}

}  // namespace ir
}  // namespace cinn
