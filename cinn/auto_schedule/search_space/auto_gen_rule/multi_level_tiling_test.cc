// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cinn/cinn.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

TEST(MultiLevelTile, SimpleLoops) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(128);

  Placeholder<float> A("A", {M});
  Placeholder<float> B("B", {N});

  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i) + B(j); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMultiLevelTile_SimpleLoops", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  MultiLevelTiling multi_level_tiling;
  ir::ModuleExpr mod_expr_before_tile(std::vector<ir::Expr>{ast_expr});
  EXPECT_TRUE(multi_level_tiling.Init(mod_expr_before_tile));

  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  ir::ModuleExpr mod_expr_after_tile = multi_level_tiling.ApplyRandomly();
  std::vector<ir::Expr> exprs        = mod_expr_after_tile.GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  std::stringstream ss;
  ss << exprs[0];

  std::string expr_str   = ss.str();
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    for (i_0, 0, 16)
    {
      for (i_1, 0, 2)
      {
        for (j_0, 0, 32)
        {
          for (j_1, 0, 4)
          {
            ScheduleBlock(C)
            {
              i0, i1 = axis.bind(((2 * i_0) + i_1), ((4 * j_0) + j_1))
              read_buffers(_A[], _B[])
              write_buffers(_C[])
              C[i0, i1] = (A[i0] + B[i1])
            }
          }
        }
      }
    }
  }
}
)ROC";
  EXPECT_EQ(utils::Trim(target_str), utils::Trim(expr_str));
}

TEST(MulitLevelTile, MatrixMultiply) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(32);
  Expr K(32);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMultiLevelTile_MatrixMultiply", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  MultiLevelTiling multi_level_tiling;
  ir::ModuleExpr mod_expr_before_tile(std::vector<ir::Expr>{ast_expr});
  EXPECT_TRUE(multi_level_tiling.Init(mod_expr_before_tile));

  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);

  ir::ModuleExpr mod_expr_after_tile = multi_level_tiling.ApplyRandomly();
  std::vector<ir::Expr> exprs        = mod_expr_after_tile.GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  std::stringstream ss;
  ss << exprs[0];

  std::string expr_str   = ss.str();
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    for (i_0, 0, 2)
    {
      for (i_1, 0, 16)
      {
        for (j_0, 0, 16)
        {
          for (j_1, 0, 2)
          {
            ScheduleBlock(C__reduce_init)
            {
              i0, i1 = axis.bind(((16 * i_0) + i_1), ((2 * j_0) + j_1))
              write_buffers(_C[])
              C__reduce_init[i0, i1] = 0
            }
            for (reduce_axis_k_0, 0, 16)
            {
              for (reduce_axis_k_1, 0, 2)
              {
                ScheduleBlock(C)
                {
                  i0, i1, i2 = axis.bind(((16 * i_0) + i_1), ((2 * j_0) + j_1), ((2 * reduce_axis_k_0) + reduce_axis_k_1))
                  read_buffers(_A[], _B[], _C[])
                  write_buffers(_C[])
                  C[i0, i1] = (C[i0, i1] + (A[i0, i2] * B[i2, i1]))
                }
              }
            }
          }
        }
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::Trim(target_str), utils::Trim(expr_str));
}

}  // namespace auto_schedule
}  // namespace cinn
