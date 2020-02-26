#include "cinn/lang/lower.h"

#include <gtest/gtest.h>

#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace lang {

TEST(lower, basic) {
  const int M = 100;
  const int N = 15;

  Placeholder<float> A("A", {Expr(M), Expr(N)});

  auto B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  auto lower_funcs = Lower("cal_B", {A, B});

  LOG(INFO) << "lower_size " << lower_funcs.size();

#define TEST_SOUTPUT(x, out)  LOG(INFO) << "\n" << x; \
  EXPECT_EQ(utils::GetStreamCnt(x), utils::Trim(out));

  auto out = R"ROC(
{
  poly_for (0, (c1 <= 99), 1)
  {
    poly_for (0, (c3 <= 14), 1)
    {
      A(c1, c3)
      B[((c1 * 15) + c3)] = (A(c1, c3) + 1)
    }
  }
}
)ROC";
  TEST_SOUTPUT(lower_funcs.front().body, out);
}

}  // namespace lang
}  // namespace cinn
