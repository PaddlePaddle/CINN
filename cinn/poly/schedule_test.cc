#include "cinn/poly/schedule.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"
#include "cinn/poly/poly_scheduler.h"

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
  CHECK_EQ(funcs.size(), 1UL);

  std::cout << funcs[0]->body << std::endl;

  auto target_out = R"ROC(
{
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j <= 99), 1)
    {
      B[((i * 100) + j)] = (A[((i * 100) + j)] + 1)
      poly_for (0, (k <= 99), 1)
      {
        C[((((i * 100) * 100) + (j * 100)) + k)] = (B[((i * 100) + j)] * B[((j * 100) + k)])
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::GetStreamCnt(funcs[0]->body), utils::Trim(target_out));
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
  CHECK_EQ(funcs.size(), 1UL);

  std::cout << funcs[0]->body << std::endl;

  auto target_out = R"ROC(
{
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j <= 99), 1)
    {
      B[((i * 100) + j)] = (A[((i * 100) + j)] + 1)
    }
  }
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j <= 99), 1)
    {
      C[((i * 100) + j)] = (A[((i * 100) + j)] + 1)
    }
  }
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j <= 99), 1)
    {
      D[((i * 100) + j)] = (A[((i * 100) + j)] + 1)
    }
  }
}
)ROC";

  EXPECT_EQ(utils::GetStreamCnt(funcs[0]->body), utils::Trim(target_out));
}

}  // namespace poly
}  // namespace cinn
