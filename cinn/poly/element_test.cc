#include "cinn/poly/element.h"

#include <gtest/gtest.h>

namespace cinn {
namespace poly {

TEST(Element, split) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j]: 0<=i,j<=100 }");

  Element ele(domain);
  Iterator outer, inner;
  std::tie(outer, inner) = ele.Split(Iterator("i"), 4);
  LOG(INFO) << ele.schedule();
  EXPECT_EQ(utils::GetStreamCnt(ele.schedule()),
            "{ S[i, j] -> S[i_outer, i_inner, j' = j] : (-i + i_inner) mod 4 = 0 and -3 + i <= 4i_outer <= i and 0 <= "
            "i_inner <= 3 }");

  EXPECT_EQ(outer.id, "i_outer");
  EXPECT_EQ(inner.id, "i_inner");
}

TEST(Element, tile) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  Element ele(domain);

  Iterator outer0, inner0, outer1, inner1;
  std::tie(outer0, inner0, outer1, inner1) = ele.Tile(Iterator("i"), Iterator("j"), 4, 6);
  LOG(INFO) << ele.schedule();
  EXPECT_EQ(outer0.id, "i_outer");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(inner0.id, "i_inner");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(
      utils::GetStreamCnt(ele.schedule()),
      "{ S[i, j, k] -> S[i_outer, i_inner, j_outer, j_inner, k' = k] : (-i + i_inner) mod 4 = 0 and (-j + j_inner) mod "
      "6 = 0 and -3 + i <= 4i_outer <= i and 0 <= i_inner <= 3 and -5 + j <= 6j_outer <= j and 0 <= j_inner <= 5 }");
}

}  // namespace poly
}  // namespace cinn
