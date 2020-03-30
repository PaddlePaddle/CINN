#include "cinn/poly/schedule.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace poly {

TEST(CreateSchedule, compute_at) {
  lang::Placeholder<float> A("A", {100, 100});

  auto B = lang::Compute(
      {100, 100}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "B");
  lang::Buffer B_buf(B->type());
  B->Bind(B_buf);

  auto C = lang::Compute(
      {100, 100, 100}, [&](Var i, Var j, Var k) { return B(i, j) * B(j, k); }, "C");
  lang::Buffer C_buf(C->type());
  C->Bind(C_buf);

  B->stage()->ComputeAt(C->stage(), 1);

  auto funcs = lang::Lower("func", {B, C});

  std::cout << funcs->body << std::endl;

  auto target_out = R"ROC(
{
  for (i, 100)
  {
    for (j, 100)
    {
      B[i, j] = (1 + A[i, j])
      for (k, 100)
      {
        C[i, j, k] = (B[i, j] * B[j, k])
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::GetStreamCnt(funcs->body), utils::Trim(target_out));
}

TEST(CreateSchedule, buffer_bind_to_multiple_tensors_schedule) {
  lang::Placeholder<float> A("A", {100, 100});
  /*
   * We create three tensors all binded to the same buffer, but has no depend in computation.
   */

  auto B = lang::Compute(
      {100, 100}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "B");
  lang::Buffer B_buf(B->type());
  B->Bind(B_buf);

  auto C = lang::Compute(
      {100, 100}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "C");
  C->Bind(B_buf);

  auto D = lang::Compute(
      {100, 100}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "D");
  D->Bind(B_buf);

  auto funcs = lang::Lower("func", {B, C, D});

  std::cout << funcs->body << std::endl;

  auto target_out = R"ROC(
{
  for (i, 100)
  {
    for (j, 100)
    {
      B[i, j] = (1 + A[i, j])
    }
  }
  for (i, 100)
  {
    for (j, 100)
    {
      C[i, j] = (1 + A[i, j])
    }
  }
  for (i, 100)
  {
    for (j, 100)
    {
      D[i, j] = (1 + A[i, j])
    }
  }
}
)ROC";

  EXPECT_EQ(utils::GetStreamCnt(funcs->body), utils::Trim(target_out));
}

}  // namespace poly
}  // namespace cinn
