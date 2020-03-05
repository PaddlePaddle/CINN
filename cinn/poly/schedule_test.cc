#include "cinn/poly/schedule.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/lang/tensor.h"
#include "cinn/poly/poly_scheduler.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, ir::Call::CallType::Halide);
  return expr;
}

/*
TEST(CreateSchedule, without_transform) {
  // create stages
  auto ctx = Context::Global().isl_ctx();
  // create call (for tensor);
  Var i("i"), j("j"), k("k");
  std::vector<Expr> args({Expr(i), Expr(j), Expr(k)});

  lang::Buffer A_arr(Float(32), "A"), B_arr(Float(32), "B"), C_arr(Float(32), "C");
  Expr A_call = CreateCall("A", args);
  Expr B_call = CreateCall("B", args);
  Expr C_call = CreateCall("C", args);

  // A[] = B[] + 1
  Expr A_expr = ir::Store::Make(Expr(A_arr.buffer()), Expr(1.f), Expr(i));
  Expr B_expr = ir::Store::Make(Expr(B_arr.buffer()), A_call + 1.f, Expr(i));
  Expr C_expr = ir::Store::Make(Expr(C_arr.buffer()), B_call + A_call, Expr(i));

  // create stages
  auto* A_stage = make_shared<Stage>(isl::set(ctx, "{ A[i,j,k]: 0<=i,j,k<100 }"), A_expr);
  auto* B_stage = make_shared<Stage>(isl::set(ctx, "{ B[i,j,k]: 0<=i,j,k<100 }"), B_expr);
  auto* C_stage = make_shared<Stage>(isl::set(ctx, "{ C[i,j,k]: 0<=i,j,k<100 }"), C_expr);

  auto schedule = CreateSchedule({A_stage, B_stage, C_stage});

  // check
  std::vector<std::vector<std::string>> group_statements({{"A", "B", "C"}});
  ASSERT_EQ(schedule->gened_groups().size(), 1L);

  for (int i = 0; i < schedule->gened_groups().size(); i++) {
    auto& group = schedule->gened_groups()[i];
    ASSERT_EQ(group.nodes.size(), group_statements[i].size());
    for (int j = 0; j < group.nodes.size(); j++) {
      LOG(INFO) << group_statements[i][j] << " " << group.nodes[j]->id();
      EXPECT_EQ(group_statements[i][j], group.nodes[j]->id());
    }
  }
}
*/

}  // namespace poly
}  // namespace cinn
