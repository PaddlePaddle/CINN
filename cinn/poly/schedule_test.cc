#include "cinn/poly/schedule.h"
#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(Schedule, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set A_set(ctx, "[]->{ A[i,j]: 0<i,j<100 }");
  Element A(A_set);
  isl::set B_set(ctx, "[]->{ B[i,j]: 0<i,j<100 }");
  Element B(B_set);
  LOG(INFO) << A.schedule();

  Scheduler scheduler;
  scheduler.RegisterElement(A);
  scheduler.RegisterElement(B);

  scheduler.After(A, B, 1);

  auto schedule = scheduler.BuildSchedule();

  for (auto item : schedule) {
    LOG(INFO) << item.first << " " << item.second;
  }
}

}  // namespace poly
}  // namespace cinn
