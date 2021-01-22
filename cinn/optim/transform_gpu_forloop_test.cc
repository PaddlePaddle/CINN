#include "cinn/optim/transform_gpu_forloop.h"

#include <gtest/gtest.h>

#include "cinn/cinn.h"
#include "cinn/optim/remove_nested_block.h"

namespace cinn {
namespace optim {

TEST(TransformGpuForloops, basic) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});

  auto [i_outer, i_inner] = stages[C]->Split(0, 10);  // NOLINT
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[C]->Bind(2, "threadIdx.y");

  auto func = Lower("elementwise_add", stages, {A, B, C});
  Expr func_expr(func);

  std::cout << "\n" << func << std::endl;

  ASSERT_TRUE(func->cuda_axis_info.valid());
  EXPECT_EQ(func->cuda_axis_info.grid_dim(0), 10);
  EXPECT_EQ(func->cuda_axis_info.grid_dim(1), 1);
  EXPECT_EQ(func->cuda_axis_info.grid_dim(2), 1);
  EXPECT_EQ(func->cuda_axis_info.block_dim(0), 10);
  EXPECT_EQ(func->cuda_axis_info.block_dim(1), 200);
  EXPECT_EQ(func->cuda_axis_info.block_dim(2), 1);

  auto target_out = R"ROC(
function elementwise_add (_A, _B, _C)
{
  if ((blockIdx.x < 10)) {
    if ((threadIdx.x < 10)) {
      if ((threadIdx.y < 200)) {
        C[((10 * blockIdx.x) + threadIdx.x), threadIdx.y] = (A[((10 * blockIdx.x) + threadIdx.x), threadIdx.y] * B[((10 * blockIdx.x) + threadIdx.x), threadIdx.y])
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::Trim(utils::GetStreamCnt(func_expr)), utils::Trim(target_out));
}

TEST(TransformGpuForloops, multiple_thread_axis) {
  Expr M(100);
  Expr N(200);

  Target target;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + A(i, j); }, "D");

  auto stages = CreateStages({C, D});

  auto [i_outer, i_inner] = stages[C]->Split(0, 10);  // NOLINT
  stages[C]->Bind(0, "blockIdx.x");
  stages[C]->Bind(1, "threadIdx.x");
  stages[C]->Bind(2, "threadIdx.y");

  stages[D]->Bind(0, "blockIdx.x");
  stages[D]->Bind(1, "threadIdx.x");

  auto func = Lower("elementwise_add", stages, {A, B, C, D});

  std::cout << "\n" << func << std::endl;

  auto target_source = R"ROC(
function elementwise_add (_A, _B, _C, _D)
{
  if ((blockIdx.x < 10)) {
    if ((threadIdx.x < 10)) {
      if ((threadIdx.y < 200)) {
        C[((10 * blockIdx.x) + threadIdx.x), threadIdx.y] = (A[((10 * blockIdx.x) + threadIdx.x), threadIdx.y] * B[((10 * blockIdx.x) + threadIdx.x), threadIdx.y])
      }
    }
  }
  if ((blockIdx.x < 100)) {
    if ((threadIdx.x < 200)) {
      D[blockIdx.x, threadIdx.x] = (C[blockIdx.x, threadIdx.x] + A[blockIdx.x, threadIdx.x])
    }
  }
}
)ROC";

  LOG(INFO) << "cuda axis info: " << func->cuda_axis_info;
  ASSERT_TRUE(func->cuda_axis_info.valid());
  EXPECT_EQ(func->cuda_axis_info.grid_dim(0), 100);   // x
  EXPECT_EQ(func->cuda_axis_info.grid_dim(1), 1);     // y
  EXPECT_EQ(func->cuda_axis_info.grid_dim(2), 1);     // z
  EXPECT_EQ(func->cuda_axis_info.block_dim(0), 200);  // x
  EXPECT_EQ(func->cuda_axis_info.block_dim(1), 200);  // y
  EXPECT_EQ(func->cuda_axis_info.block_dim(2), 1);    // z

  ASSERT_EQ(utils::Trim(target_source), utils::GetStreamCnt(func));
}

}  // namespace optim
}  // namespace cinn
