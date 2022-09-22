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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
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

TEST(MultiLevelTile, SampleSplitTwo) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  MultiLevelTiling multi_level_tiling(target);

  for (int i = 0; i < 100; ++i) {
    size_t number_to_split    = rand() % 65535 + 2;  // random number in [2, 2^16]
    std::vector<size_t> split = multi_level_tiling.SampleSplitTwo<size_t>(number_to_split);
    EXPECT_EQ(split.size(), 2UL);
    EXPECT_EQ(split[0] * split[1], number_to_split);
  }
}

TEST(MultiLevelTile, SampleTileSplit) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  MultiLevelTiling multi_level_tiling(target);

  for (int i = 0; i < 100; ++i) {
    int number_to_split    = rand() % 65535 + 2;  // random number in [2, 2^16]
    int split_size         = rand() % 5 + 1;      // random in [1, 5]
    std::vector<int> split = multi_level_tiling.SampleTileSplit<int>(number_to_split, split_size);
    EXPECT_EQ(split.size(), static_cast<size_t>(split_size));
    int product = 1;
    for (int num : split) {
      product *= num;
    }
    EXPECT_EQ(product, number_to_split);
  }
}

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

  MultiLevelTiling multi_level_tiling(target);
  ir::IRSchedule init_schedule(ir::ModuleExpr({ast_expr}));
  EXPECT_EQ(multi_level_tiling.Init(init_schedule), RuleApplyType::kApplyAndSkipThisRule);

  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  ir::IRSchedule applied_schedule = multi_level_tiling.ApplyRandomly();
  std::vector<ir::Expr> exprs     = applied_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  std::stringstream ss;
  ss << exprs[0];

  std::string expr_str = ss.str();
  VLOG(6) << expr_str;
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

  MultiLevelTiling multi_level_tiling(target);
  ir::IRSchedule init_schedule(ir::ModuleExpr({ast_expr}));
  EXPECT_EQ(multi_level_tiling.Init(init_schedule), RuleApplyType::kApplyAndSkipThisRule);

  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);

  ir::IRSchedule appied_schedule = multi_level_tiling.ApplyRandomly();
  std::vector<ir::Expr> exprs    = appied_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  std::stringstream ss;
  ss << exprs[0];

  std::string expr_str = ss.str();
  VLOG(6) << expr_str;
}

}  // namespace auto_schedule
}  // namespace cinn
