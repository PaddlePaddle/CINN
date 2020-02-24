#include "cinn/poly/stage.h"

#include <gtest/gtest.h>

#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, ir::Call::CallType::Halide);
  return expr;
}

TEST(Stage, input_statements) {
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

  LOG(INFO) << "A expr " << A_stage->expr();
  LOG(INFO) << "B expr " << B_stage->expr();
  LOG(INFO) << "C expr " << C_stage->expr();

  auto A_deps = A_stage->input_statements();
  auto B_deps = B_stage->input_statements();
  auto C_deps = C_stage->input_statements();

  std::set<std::string> A_target_deps;
  std::set<std::string> B_target_deps({"A"});
  std::set<std::string> C_target_deps({"A", "B"});

  EXPECT_EQ(A_deps.size(), A_target_deps.size());
  EXPECT_EQ(B_deps.size(), B_target_deps.size());
  EXPECT_EQ(C_deps.size(), C_target_deps.size());

  for (auto& x : A_deps) EXPECT_TRUE(A_target_deps.count(x));
  for (auto& x : B_deps) EXPECT_TRUE(B_target_deps.count(x));
  for (auto& x : C_deps) EXPECT_TRUE(C_target_deps.count(x));
}

TEST(Stage, split) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j]: 0<=i,j<=100 }");

  Stage ele(domain);
  Iterator outer, inner;
  std::tie(outer, inner) = ele.Split(Iterator("i"), 4);
  LOG(INFO) << ele.transform();
  EXPECT_EQ(utils::GetStreamCnt(ele.transform()),
            "{ S[i, j] -> S[i_outer, i_inner, j' = j] : (-i + i_inner) mod 4 = 0 and -3 + i <= 4i_outer <= i and 0 <= "
            "i_inner <= 3 }");

  EXPECT_EQ(outer.id, "i_outer");
  EXPECT_EQ(inner.id, "i_inner");
}

TEST(Stage, tile) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  Stage ele(domain);

  Iterator outer0, inner0, outer1, inner1;
  std::tie(outer0, inner0, outer1, inner1) = ele.Tile(Iterator("i"), Iterator("j"), 4, 6);
  LOG(INFO) << ele.transform();
  EXPECT_EQ(outer0.id, "i_outer");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(inner0.id, "i_inner");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(
      utils::GetStreamCnt(ele.transform()),
      "{ S[i, j, k] -> S[i_outer, i_inner, j_outer, j_inner, k' = k] : (-i + i_inner) mod 4 = 0 and (-j + j_inner) mod "
      "6 = 0 and -3 + i <= 4i_outer <= i and 0 <= i_inner <= 3 and -5 + j <= 6j_outer <= j and 0 <= j_inner <= 5 }");
}

TEST(Stage, reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  Stage ele(domain);
  Iterator i("i"), j("j"), k("k");
  ele.Reorder(std::vector<Iterator>{{i, k, j}});
  LOG(INFO) << ele.transform();
}

TEST(Stage, split_reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  Stage ele(domain);
  Iterator outer, inner;
  std::tie(outer, inner) = ele.Split(Iterator("i"), 4);

  Iterator i("i"), j("j"), k("k");
  ele.Reorder(std::vector<Iterator>{{outer, k, inner, j}});
  LOG(INFO) << ele.transform();
}

}  // namespace poly
}  // namespace cinn
