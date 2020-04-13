#include "cinn/lang/lower.h"

#include <gtest/gtest.h>

#include "cinn/lang/buffer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace lang {

TEST(lower, basic) {
  auto M = Expr(100);
  auto N = Expr(15);

  Placeholder<float> A("A", {Expr(M), Expr(N)});

  Buffer B_buf(Float(32));

  auto B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  B->Bind(B_buf);

  auto lower_funcs = Lower("cal_B", {A, B});

  LOG(INFO) << "lower_size " << lower_funcs;

#define TEST_SOUTPUT(x, out)           \
  std::cout << "\n" << x << std::endl; \
  EXPECT_EQ(utils::GetStreamCnt(x), utils::Trim(out));

  auto out = R"ROC(
{
  for (i, 100)
  {
    for (j, 15)
    {
      B[i, j] = (1 + A[i, j])
    }
  }
}
)ROC";
  TEST_SOUTPUT(lower_funcs->body, out);
}

TEST(lower, more_complex) {
  Expr M(100);
  Expr N(15);
  Expr K(200);

  Placeholder<float> A("A", {Expr(M), Expr(N)});
  Placeholder<float> B("B", {Expr(N), Expr(K)});

  Buffer C_buf(Float(32));
  auto C = Compute(
      {M, N, K}, [=](Var i, Var j, Var k) -> Expr { return A(i, j) * B(j, k); }, "C");
  C->Bind(C_buf);

  auto lower_funcs = Lower("cal_C", {A, B, C});

  std::cout << "func:\n" << Expr(lower_funcs->self()) << std::endl;
}

//! To support training, the dynamic shape support is vital. We test the corresponding lower ability here.
TEST(lower, dynamic_shape) {
  Var B("B");  // B is like shape here.
  Expr N(15);
  Expr K(200);

  // Input is B * N, B is like batch.
  Placeholder<float> X("X", {Expr(B), Expr(N)});
  Placeholder<float> W("W", {Expr(N), Expr(K)});

  auto C = Compute(
      {B, N, K}, [=](Var i, Var j, Var k) -> Expr { return X(i, j) * W(j, k); }, "C");
  C->WithBuffer();

  auto lower_funcs = Lower("cal_C", {X, W, C});

  std::cout << "func:\n" << Expr(lower_funcs->self()) << std::endl;
}

}  // namespace lang
}  // namespace cinn
