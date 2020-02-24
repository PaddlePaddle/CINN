#include "cinn/poly/schedule.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace poly {

TEST(Scheduler, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set A_set(ctx, "[]->{ A[i,j]: 0<i,j<100 }");
  Stage A(A_set);
  isl::set B_set(ctx, "[]->{ B[i,j]: 0<i,j<100 }");
  Stage B(B_set);
  LOG(INFO) << A.transform();

  Scheduler scheduler;
  scheduler.AddStage(A);
  scheduler.AddStage(B);

  scheduler.After(A, B, 1);

  auto schedule = scheduler.BuildSchedule();

  EXPECT_EQ(utils::GetStreamCnt(schedule["A"]), "{ A[i, j] -> [t0 = 0, d0 = i, t1 = 0, d1 = j] }");
  EXPECT_EQ(utils::GetStreamCnt(schedule["B"]), "{ B[i, j] -> [t0 = 0, d0 = i, t1 = 1, d1 = j] }");

  for (auto item : schedule) {
    LOG(INFO) << item.first << " " << item.second;
  }
}

TEST(Scheduler, basic_with_transform) {
  isl::ctx ctx(isl_ctx_alloc());
  Stage A(isl::set(ctx, "[]->{ A[i,j]: 0<i,j<100 }"));
  Stage B(isl::set(ctx, "[]->{ B[i,j]: 0<i,j<100 }"));
  auto x = A.Split("i", 4);
  LOG(INFO) << A.transform();
  B.Split(Iterator("j"), 6);
  LOG(INFO) << B.transform();

  Scheduler scheduler;
  scheduler.AddStage(A);
  scheduler.AddStage(B);
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

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, ir::Call::CallType::Halide);
  return expr;
}

TEST(CreateSchedule, without_transform) {
  // create stages
  auto ctx = Context::Global().isl_ctx();
  // create call (for tensor);
  Var i("i"), j("j"), k("k");
  std::vector<Expr> args({Expr(i), Expr(j), Expr(k)});

  Var A_arr("A"), B_arr("B"), C_arr("C");
  Expr A_call = CreateCall("A", args);
  Expr B_call = CreateCall("B", args);
  Expr C_call = CreateCall("C", args);

  // A[] = B[] + 1
  Expr A_expr = ir::Store::Make(A_arr, Expr(1.f), Expr(i));
  Expr B_expr = ir::Store::Make(B_arr, A_call + 1.f, Expr(i));
  Expr C_expr = ir::Store::Make(C_arr, B_call + A_call, Expr(i));

  // create stages
  auto* A_stage = make_shared<Stage>(isl::set(ctx, "{ A[i,j,k]: 0<=i,j,k<100 }"), A_expr);
  auto* B_stage = make_shared<Stage>(isl::set(ctx, "{ B[i,j,k]: 0<=i,j,k<100 }"), B_expr);
  auto* C_stage = make_shared<Stage>(isl::set(ctx, "{ C[i,j,k]: 0<=i,j,k<100 }"), C_expr);

  auto schedule = CreateSchedule({A_stage, B_stage, C_stage});

  // check
  std::vector<std::vector<std::string>> group_statements({{"A", "B"}, {"C"}});
  ASSERT_EQ(schedule->gened_groups().size(), 2L);

  for (int i = 0; i < schedule->gened_groups().size(); i++) {
    auto& group = schedule->gened_groups()[i];
    ASSERT_EQ(group.nodes.size(), group_statements[i].size());
    for (int j = 0; j < group.nodes.size(); j++) {
      LOG(INFO) << group_statements[i][j] << " " << group.nodes[j]->id();
      EXPECT_EQ(group_statements[i][j], group.nodes[j]->id());
    }
  }
}

}  // namespace poly
}  // namespace cinn
