#include "cinn/optim/cache_read_write_replace.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

TEST(CacheReadWriteReplace, basic) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) -> Expr { return A(i, j) + B(i, j); }, "C");

  // AA cache
  auto AA = A->stage()->CacheRead("shared", {C});
  auto CC = C->stage()->CacheWrite("local");

  auto fn = Lower("fn", {A, B, C}, {}, {AA, CC});

  LOG(INFO) << "fn:\n" << Expr(fn);

  auto target = R"ROC(
function fn (_A, _B, _C)
{
  for (i, 100)
  {
    for (j, 20)
    {
      A_read_cache_3[i, j] = A[i, j]
    }
  }
  for (i, 100)
  {
    for (j, 20)
    {
      C[i, j] = (A_read_cache_3[i, j] + B[i, j])
    }
  }
  for (i, 100)
  {
    for (j, 20)
    {
      C_cache_write_out_4[i, j] = C[i, j]
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::Trim(target), utils::GetStreamCnt(fn));
}

}  // namespace optim
}  // namespace cinn
