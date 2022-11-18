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

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_write.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/backends/compiler.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/module.h"
#include "cinn/ir/tensor.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

TEST(AddCacheWrite, Init) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(32);
  ir::Expr K(32);

  // matmul case
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestAddCacheWrite_InitTrue", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr matmul_expr = funcs[0]->body;
  VLOG(6) << "Matmul Expr before AddCacheWrite: ";
  VLOG(6) << matmul_expr;

  ir::IRSchedule ir_schedule_matmul(ir::ModuleExpr({matmul_expr}));

  AddCacheWrite add_cache_write(target);
  auto apply_type = add_cache_write.Init(&ir_schedule_matmul);
#ifdef CINN_WITH_CUDA
  EXPECT_EQ(apply_type, RuleApplyType::kApplyAndSkipAllRules);
#else
  EXPECT_EQ(apply_type, RuleApplyType::kApplyAndSkipThisRule);
#endif
  EXPECT_EQ(add_cache_write.NumberApplicable(), 1);

  add_cache_write.ApplyRandomly();
  std::vector<ir::Expr> exprs = ir_schedule_matmul.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Matmul Expr after AddCacheWrite: " << exprs[0];

  // add case
  Placeholder<float> D("D", {M, K});
  Placeholder<float> E("E", {K, N});
  ir::Tensor F = Compute(
      {M, N}, [&](Var i, Var j) { return D(i, j) * E(i, j); }, "F");

  poly::StageMap stages_add = CreateStages({F});
  std::vector<ir::LoweredFunc> funcs_add =
      lang::LowerVec("TestAddCacheWrite_InitFalse", stages_add, {F}, {}, {}, nullptr, target, true);

  ir::Expr add_expr = funcs_add[0]->body;
  VLOG(6) << "Mat Add Expr before AddCacheWrite: ";
  VLOG(6) << add_expr;

  ir::IRSchedule ir_schedule_add(ir::ModuleExpr({add_expr}));

  AddCacheWrite add_cache_write2(target);
  EXPECT_EQ(add_cache_write2.Init(&ir_schedule_add), RuleApplyType::kCannotApply);
  EXPECT_EQ(add_cache_write2.NumberApplicable(), 0);
}

TEST(AddCacheWrite, MatrixMultiply) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(32);
  ir::Expr K(32);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestAddCacheWrite_MatrixMultiply", stages, {A, B, C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));

  // Apply MultiLevelTiling before AddCacheWrite
  MultiLevelTiling multi_level_tiling(target);
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule), RuleApplyType::kApplyAndSkipThisRule);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);

  multi_level_tiling.ApplyRandomly();
  std::vector<ir::Expr> exprs = ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after MultiLevelTiling: " << exprs[0];

  // Apply AddCacheWrite
  AddCacheWrite add_cache_write(target);
  auto apply_type = add_cache_write.Init(&ir_schedule);
#ifdef CINN_WITH_CUDA
  EXPECT_EQ(apply_type, RuleApplyType::kApplyAndSkipAllRules);
#else
  EXPECT_EQ(apply_type, RuleApplyType::kApplyAndSkipThisRule);
#endif
  EXPECT_EQ(add_cache_write.NumberApplicable(), 1);

  add_cache_write.ApplyRandomly();
  exprs = ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after AddCacheWrite: " << exprs[0];

  auto temp_buffers = lang::GetTempBuffers({A, B, C}, stages, exprs[0]);
  auto func         = ir::_LoweredFunc_::Make(funcs[0]->name, funcs[0]->args, exprs[0], temp_buffers);

  ir::Module::Builder builder("test_bulder", target);
  builder.AddFunction(func);
  auto build_module = builder.Build();

  auto compiler = backends::Compiler::Create(target);
  compiler->Build(build_module);
}

}  // namespace auto_schedule
}  // namespace cinn
