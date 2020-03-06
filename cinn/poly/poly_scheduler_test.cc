#include "cinn/poly/poly_scheduler.h"

#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(Scheduler, basic) {
  isl::ctx ctx(Context::Global().isl_ctx());
  isl::set A_set(ctx, "[]->{ A[i,j]: 0<i,j<100 }");
  auto A = Stage::New(A_set);
  isl::set B_set(ctx, "[]->{ B[i,j]: 0<i,j<100 }");
  auto B = Stage::New(B_set);
  LOG(INFO) << A->transform();

  PolyGroupScheduler scheduler({A.get(), B.get()});
  scheduler.After(*A, *B, 1);
  scheduler.Build();

  auto schedule = scheduler.schedule_map();

  EXPECT_EQ(utils::GetStreamCnt(schedule["A"]), "{ A[i, j] -> [t0 = 0, d0 = i, t1 = 0, d1 = j] }");
  EXPECT_EQ(utils::GetStreamCnt(schedule["B"]), "{ B[i, j] -> [t0 = 0, d0 = i, t1 = 1, d1 = j] }");

  for (auto item : schedule) {
    LOG(INFO) << item.first << " " << item.second;
  }
}

TEST(Scheduler, basic_with_transform) {
  isl::ctx ctx = Context::Global().isl_ctx();
  auto A       = Stage::New(isl::set(ctx, "[]->{ A[i,j]: 0<i,j<100 }"));
  auto B       = Stage::New(isl::set(ctx, "[]->{ B[i,j]: 0<i,j<100 }"));
  auto x       = A->Split("i", 4);
  LOG(INFO) << A->transform();
  B->Split(Iterator("j"), 6);
  LOG(INFO) << B->transform();

  PolyGroupScheduler scheduler({A.get(), B.get()});
  scheduler.After(*A, *B, 1);
  scheduler.Build();
  auto schedule = scheduler.schedule_map();
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
