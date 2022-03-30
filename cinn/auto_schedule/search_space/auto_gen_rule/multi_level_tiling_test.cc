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

namespace cinn {
namespace auto_schedule {

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
  VLOG(6) << "Expr after MultiLevelTiling: ";
  VLOG(6) << exprs[0];
}

}  // namespace auto_schedule
}  // namespace cinn
