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

  lang::Lower("func", {B, C});
}

}  // namespace poly
}  // namespace cinn
