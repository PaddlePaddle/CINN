#include "cinn/lang/tensor.h"

#include <gtest/gtest.h>

#include "cinn/ir/function.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"

namespace cinn {
namespace ir {
using utils::GetStreamCnt;
using utils::Trim;

TEST(Tensor, inlined) {
  Expr M(100), N(20);

  lang::Placeholder<float> A("A", {M, N});
  lang::Placeholder<float> B("B", {M, N});

  lang::Buffer D_buf(Float(32));
  // C is inlined
  Tensor C = lang::Compute(
      {M, N}, [=](Var i, Var j) { return A(i, j) + B(i, j); }, "C");
  Tensor D = lang::Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return C(i, j) * 2.f + 1.f; }, "D");
  D->Bind(D_buf);

  auto funcs = lang::Lower("func_C", {A, B, D});
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

}  // namespace ir
}  // namespace cinn
