#include "cinn/optim/ir_simplify.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {
using utils::GetStreamCnt;
using utils::Trim;

TEST(IrSimplify, basic) {
  auto A = Compute(
      {100, 20}, [&](Var i, Var j) { return Expr(1.f); }, "C");
  Buffer A_buf(A->type());
  A->Bind(A_buf);

  Var i("i"), j("j");
  i->set_type(Int(32));
  j->set_type(Int(32));

  {  // simple case
    auto B = A(i, Expr(0)) + 1.f * 0.f + 100.f + 24.5f;

    LOG(INFO) << "B " << B;
    // get (((C[(i * 20)] + 0) + 100) + 24.5)
    Simplify(&B);
    LOG(INFO) << "simplified: " << B;
    auto out = "(124.5 + C[(20 * i)])";
    EXPECT_EQ(out, utils::GetStreamCnt(B));
  }

  {
    Placeholder<float> x("X", {100, 20});
    Placeholder<float> y("y", {100, 20});

    auto B = Compute(
        {100, 20},
        [&](Expr i, Expr j) {
          return x(i + 0, j + 0) + y(i, j * 0) * 1.f + 0.f * x(i, j) + 25.f + 100.f - 0.f +
                 9.f * 10000.f * 1.f * 1.f * 0.f;
        },
        "B");
    Buffer B_buf(B->type());
    B->Bind(B_buf);

    auto func = Lower("func", {B});
    auto body = func.front()->body;

    LOG(INFO) << "original body:\n" << body;
    Simplify(&body);
    auto target_out = R"ROC(
{
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j <= 19), 1)
    {
      B[((20 * i) + j)] = (125 + (X[((20 * i) + j)] + y[(20 * i)]))
    }
  }
})ROC";
    EXPECT_EQ(Trim(GetStreamCnt(body)), Trim(target_out));
  }

  {
    Placeholder<float> x("X", {100, 20});
    Placeholder<float> y("y", {100, 20});

    auto B = Compute(
        {100, 20},
        [&](Expr i, Expr j) {
          return x(i + 0, j * 0) + y(i, j * 0) / (1.f + 2.f) + 0.f * x(i, j) + 25.f + 100.f - 0.f +
                 9.f * 10000.f * 1.f * 1.f * 0.f;
        },
        "B");
    Buffer B_buf(B->type());
    B->Bind(B_buf);

    auto func = Lower("func", {B});
    auto body = func.front()->body;

    LOG(INFO) << "original body:\n" << body;
    Simplify(&body);

    auto target_out = R"ROC(
{
  poly_for (0, (i <= 99), 1)
  {
    poly_for (0, (j <= 19), 1)
    {
      B[((20 * i) + j)] = (125 + (X[(20 * i)] + (y[(20 * i)] / 3)))
    }
  }
}
)ROC";
    EXPECT_EQ(Trim(GetStreamCnt(body)), Trim(target_out));
  }
}

}  // namespace optim
}  // namespace cinn
