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
#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/auto_schedule/tests/test_op_builder.h"
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
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, "C"), RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, "C");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    std::string expr_str = ss.str();
    VLOG(6) << expr_str;
  };

  test_func(&ir_schedule);
  test_func(&new_states[0]->ir_schedule);
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
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, "C"), RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, "C");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    std::string expr_str = ss.str();
    VLOG(6) << expr_str;
  };

  test_func(&ir_schedule);
  test_func(&new_states[0]->ir_schedule);
}

class TestMultiLevelTiling : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names;
  std::vector<std::string> default_output_names;
};

TEST_F(TestMultiLevelTiling, Matmul) {
  default_input_names            = {"X", "Y"};
  default_output_names           = {"temp_matmul_out"};
  std::vector<int32_t> X_shape   = {32, 32};
  std::vector<int32_t> Y_shape   = {32, 32};
  std::vector<int32_t> out_shape = {32, 32};

  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(MatmulOpBuilder(X_shape, Y_shape)());
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(target_);
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, default_output_names[0]);
  VLOG(6) << "After MultiLevelTiling, state:\n" << new_states[0]->DebugString();

  // build ir::Module and debug source code
  auto ir_module   = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  // execute and check precision
  CheckResult(GenExecutableKernel(ir_module),
              GenExecutableKernel(
                  BuildIRModule(MakeIRSchedule(MatmulOpBuilder(X_shape, Y_shape)(), /* apply_manual_schedule*/ true))),
              default_input_names,
              default_output_names,
              {X_shape, Y_shape},
              {out_shape},
              target_);
}

TEST_F(TestMultiLevelTiling, ReduceSum) {
  default_input_names             = {"X"};
  default_output_names            = {"var_0_tmp"};
  std::vector<int32_t> X_shape    = {1, 16, 32};
  std::vector<int32_t> out_shape  = {1, 16, 1};
  std::vector<int32_t> reduce_dim = {2};

  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(ReduceSumOpBuilder(X_shape, reduce_dim)());
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(target_);
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]), RuleApplyType::kCannotApply);
}

}  // namespace auto_schedule
}  // namespace cinn
