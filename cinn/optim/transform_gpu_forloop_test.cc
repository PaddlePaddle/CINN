#include "cinn/optim/transform_gpu_forloop.h"
#include <gtest/gtest.h>
#include "cinn/cinn.h"
#include "cinn/optim/remove_nested_block.h"

namespace cinn {
namespace optim {

TEST(TransformGpuForloop, basic) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  C->WithBuffer();

  C->stage()->GpuBlocks({C->stage()->axis(0)});
  C->stage()->GpuThreads({C->stage()->axis(1)});

  auto func = Lower("elementwise_add", {A, B, C});
  Expr func_expr(func);

  std::cout << "\n" << func << std::endl;

  auto target_out = R"ROC(
function elementwise_add (_A, _B, _C)
{
  {
    C[blockIdx.x, threadIdx.x] = (A[blockIdx.x, threadIdx.x] * B[blockIdx.x, threadIdx.x])
  }
}
)ROC";

  EXPECT_EQ(utils::Trim(utils::GetStreamCnt(func_expr)), utils::Trim(target_out));
}

}  // namespace optim
}  // namespace cinn
