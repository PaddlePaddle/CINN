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

  EXPECT_EQ(utils::GetStreamCnt(schedule["A"]), "{ A[i, j] -> [t0 = 0, d0 = i, t1 = 0, d1 = j] }");
  EXPECT_EQ(utils::GetStreamCnt(schedule["B"]), "{ B[i, j] -> [t0 = 0, d0 = i, t1 = 1, d1 = j] }");

  for (auto item : schedule) {
    LOG(INFO) << item.first << " " << item.second;
  }
}

TEST(Schedule, basic_with_transform) {
  isl::ctx ctx(isl_ctx_alloc());
  Element A(isl::set(ctx, "[]->{ A[i,j]: 0<i,j<100 }"));
  Element B(isl::set(ctx, "[]->{ B[i,j]: 0<i,j<100 }"));
  auto x = A.Split("i", 4);
  LOG(INFO) << A.schedule();
  B.Split(Iterator("j"), 6);
  LOG(INFO) << B.schedule();

  Scheduler scheduler;
  scheduler.RegisterElement(A);
  scheduler.RegisterElement(B);
  scheduler.After(A, B, 1);
  auto schedule = scheduler.BuildSchedule();
  for (auto item : schedule) {
    LOG(INFO) << item.first << " " << item.second;
  }

  EXPECT_EQ(utils::GetStreamCnt(schedule["A"]),
            "{ A[i_outer, i_inner, j] -> [t0 = 0, d0 = i_outer, t1 = 0, d1 = i_inner, t2 = 0, d2 = j] }");
  EXPECT_EQ(utils::GetStreamCnt(schedule["B"]),
            "{ B[i, j_outer, j_inner] -> [t0 = 0, d0 = i, t1 = 1, d1 = j_outer, t2 = 0, d2 = j_inner] }");
}

}  // namespace poly
}  // namespace cinn
