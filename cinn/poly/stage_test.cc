#include "cinn/poly/stage.h"

#include <gtest/gtest.h>

#include <set>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(Float(32), name, args, ir::Call::CallType::CINN);
  return expr;
}

TEST(Stage, split) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j]: 0<=i,j<=100 }");

  auto ele = Stage::New(domain);
  Iterator outer, inner;
  std::tie(outer, inner) = ele->Split(Iterator("i"), 4);
  LOG(INFO) << ele->transform();
  EXPECT_EQ(utils::GetStreamCnt(ele->transform()),
            "{ S[i, j] -> S[i_outer, i_inner, j' = j] : (-i + i_inner) mod 4 = 0 and -3 + i <= 4i_outer <= i and 0 <= "
            "i_inner <= 3 }");

  EXPECT_EQ(outer.id, "i_outer");
  EXPECT_EQ(inner.id, "i_inner");
}

TEST(Stage, tile) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);

  Iterator outer0, inner0, outer1, inner1;
  std::tie(outer0, inner0, outer1, inner1) = ele->Tile(Iterator("i"), Iterator("j"), 4, 6);
  LOG(INFO) << ele->transform();
  EXPECT_EQ(outer0.id, "i_outer");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(inner0.id, "i_inner");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(
      utils::GetStreamCnt(ele->transform()),
      "{ S[i, j, k] -> S[i_outer, i_inner, j_outer, j_inner, k' = k] : (-i + i_inner) mod 4 = 0 and (-j + j_inner) mod "
      "6 = 0 and -3 + i <= 4i_outer <= i and 0 <= i_inner <= 3 and -5 + j <= 6j_outer <= j and 0 <= j_inner <= 5 }");
}

TEST(Stage, reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  Iterator i("i"), j("j"), k("k");
  ele->Reorder(std::vector<Iterator>{{i, k, j}});
  LOG(INFO) << ele->transform();
}

TEST(Stage, split_reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  Iterator outer, inner;
  std::tie(outer, inner) = ele->Split(Iterator("i"), 4);

  Iterator i("i"), j("j"), k("k");
  ele->Reorder(std::vector<Iterator>{{outer, k, inner, j}});
  LOG(INFO) << ele->transform();
}

TEST(ComputeAtRelation, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain0(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  isl::set domain1(ctx, "{ D[a,b,c,d]: 0<=a,b,c,d<=100 }");

  auto stage0 = Stage::New(domain0);
  auto stage1 = Stage::New(domain0);

  ComputeAtRelation relation;
  relation.stage = stage1;
  relation.level = 2;
  ASSERT_TRUE(relation.IsCompatible(stage0.get()));
}

}  // namespace poly
}  // namespace cinn
