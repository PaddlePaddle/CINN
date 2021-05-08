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

  auto stages = CreateStages({C});

  // AA cache
  auto AA = stages[A]->CacheRead("shared", {C}, stages);
  auto CC = stages[C]->CacheWrite("local", stages);

  auto fn = Lower("fn", stages, {A, B, C}, {}, {AA, CC});

  LOG(INFO) << "fn:\n" << Expr(fn);

  auto target = R"ROC(
function fn (_A, _B, _C)
{
  for (i, 0, 100)
  {
    for (j, 0, 20)
    {
      A_read_cache[i, j] = A[i, j]
    }
  }
  __syncthreads()
  for (i, 0, 100)
  {
    for (j, 0, 20)
    {
      C[i, j] = (A_read_cache[i, j] + B[i, j])
    }
  }
  for (i, 0, 100)
  {
    for (j, 0, 20)
    {
      C_cache_write_out[i, j] = C[i, j]
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::Trim(target), utils::GetStreamCnt(fn));
}

TEST(CacheReadWriteReplace, cache_write) {
  Context::Global().ResetNameId();

  Expr M(100);
  Expr N(100);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [=](Expr i, Expr j) { return A(i, j) + 1.f; }, "C");

  auto C0 = Compute(
      {M, N}, [=](Expr i, Expr j) { return C(i, j) + 1.f; }, "C0");
  auto C1 = Compute(
      {M, N}, [=](Expr i, Expr j) { return C0(i, j) + 1.f; }, "C1");

  auto stages = CreateStages({A, B, C, C0, C1});
  stages[C]->ComputeInline();
  stages[C0]->ComputeInline();

  auto Co = stages[C1]->CacheWrite("shared", stages);

  auto fn = Lower("fn", stages, {A, B, Co}, {}, {C, C0, C1});
  LOG(INFO) << "\n" << fn;

  auto target_source = R"ROC(
function fn (_A, _B, _C1_cache_write_out)
{
  for (i, 0, 100)
  {
    for (j, 0, 100)
    {
      C1[i, j] = (3 + A[i, j])
    }
  }
  for (i, 0, 100)
  {
    for (j, 0, 100)
    {
      C1_cache_write_out[i, j] = C1[i, j]
    }
  }
}
)ROC";

  ASSERT_EQ(utils::Trim(target_source), utils::GetStreamCnt(fn));
}

}  // namespace optim
}  // namespace cinn
