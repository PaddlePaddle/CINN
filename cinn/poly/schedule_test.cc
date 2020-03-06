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

TEST(CreateSchedule, compute_at) {
  // create stages
  auto ctx = Context::Global().isl_ctx();
  // create call (for tensor);
  Var i("i"), j("j"), k("k");
  std::vector<Expr> args({Expr(i), Expr(j), Expr(k)});

  lang::Buffer A_arr(Float(32), "A"), B_arr(Float(32), "B"), C_arr(Float(32), "C");
  Expr A_call = CreateCall("A", args);
  Expr B_call = CreateCall("B", args);

  // A[] = B[] + 1
  Expr A_expr = ir::Store::Make(Expr(A_arr.buffer()), Expr(1.f), Expr(i));
  Expr B_expr = ir::Store::Make(Expr(B_arr.buffer()), A_call + 1.f, Expr(i));

  // create stages
  auto A_stage = Stage::New(isl::set(ctx, "{ A[i,j]: 0<=i,j<100 }"), A_expr);
  auto B_stage = Stage::New(isl::set(ctx, "{ B[i,j,k]: 0<=i,j,k<100 }"), B_expr);

  A_stage->ComputeAt(B_stage.get(), 1);

  auto schedule = CreateSchedule({A_stage.get(), B_stage.get()});
  auto A_out    = utils::GetStreamCnt(schedule->schedule["A"]);
  auto B_out    = utils::GetStreamCnt(schedule->schedule["B"]);
  LOG(INFO) << "A_out" << A_out;
  LOG(INFO) << "B_out" << B_out;

  ASSERT_EQ(A_out, "{ A[i, j] -> [t0 = 0, d0 = i, t1 = 0, d1 = j, t2 = 0, d2 = 0] }");
  ASSERT_EQ(B_out, "{ B[i, j, k] -> [t0 = 0, d0 = i, t1 = 0, d1 = j, t2 = 0, d2 = k] }");
}

}  // namespace poly
}  // namespace cinn
