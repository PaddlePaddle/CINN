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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/cinn.h"
#include "cinn/lang/lower.h"
#include "tests/program_builder.h"
#include "tests/subgraph_program_builder.h"

namespace cinn {
namespace auto_schedule {

TEST(AutoUnroll, Init) {
  using namespace ir;

  Expr M(100);
  Expr N(4);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});
  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  auto stages = CreateStages({C});
  auto funcs  = cinn::lang::LowerVec("test_init", stages, {A, B, C}, {}, {}, nullptr, target, true);

  auto ast_expr = funcs[0]->body;
  ir::IRSchedule init_schedule(ir::ModuleExpr({ast_expr}));
  AutoUnroll test_rule(target);
  // not meet specific condition
  ASSERT_EQ(test_rule.Init(&init_schedule), RuleApplyType::kCannotApply);
}

TEST(AutoUnroll, UnrollableApply) {
  using namespace ir;

  Expr M(100);
  Expr N(4);
  Expr K(32);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});
  Var k(K.as_int32(), "k0");
  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  auto stages = CreateStages({C});
  auto funcs  = cinn::lang::LowerVec("test_unrollable", stages, {A, B, C}, {}, {}, nullptr, target, true);

  auto ast_expr             = funcs[0]->body;
  auto* init_block_realize  = ast_expr.As<ir::Block>()->stmts.front().As<ir::ScheduleBlockRealize>();
  auto* init_schedule_block = init_block_realize->schedule_block.As<ir::ScheduleBlock>();
  ASSERT_NE(init_schedule_block, nullptr);
  ASSERT_TRUE(init_schedule_block->attrs.empty());
  VLOG(-6) << "Before auto-unroll:\n" << ast_expr;

  AutoUnroll test_rule(target);
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  ASSERT_EQ(test_rule.Init(&ir_schedule), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(test_rule.NumberApplicable(), 1);
  test_rule.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(test_rule.AnalyseApplyType(state, "C"), RuleApplyType::kApplyAndPruneOtherRules);
  std::vector<cinn::auto_schedule::SearchState> states = test_rule.ApplyOnBlock(state, "C");

  auto test_func = [](IRSchedule* ir_sch) {
    Expr applied_expr            = ir_sch->GetModule().GetExprs().front();
    auto* applied_block_realize  = applied_expr.As<ir::Block>()->stmts.front().As<ir::ScheduleBlockRealize>();
    auto* applied_schedule_block = applied_block_realize->schedule_block.As<ir::ScheduleBlock>();
    ASSERT_FALSE(applied_schedule_block->attrs.empty());
    EXPECT_EQ(applied_schedule_block->attrs.count(ir::attr::auto_unroll_max_step), 1);
    const auto& attr_value = applied_schedule_block->attrs.at(ir::attr::auto_unroll_max_step);
    const int* max_step    = absl::get_if<int>(&attr_value);
    EXPECT_NE(max_step, nullptr);
    EXPECT_LE(*max_step, 128);
    VLOG(-6) << "After auto-unroll:max_step=" << *max_step << ", Ast:\n" << ir_sch->GetModule().GetExprs().front();
  };

  test_func(&ir_schedule);
  test_func(&states[0]->ir_schedule);
}

#ifdef CINN_WITH_CUDA
class TestAutoUnroll : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names  = {"X", "Y"};
  std::vector<std::string> default_output_names = {"temp_matmul_out"};
};
TEST_F(TestAutoUnroll, ApplyOnMatmulWithTiling) {
  frontend::Program matmul_op = tests::OpBuilder("matmul").Build({{"X", {32, 4}}, {"Y", {4, 32}}});
  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule       = MakeIRSchedule(matmul_op);
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  AutoUnroll auto_unroll(target_);
  SearchState state(ir_schedule, 0, {});
  const std::string& applied_block_name = default_output_names.back();
  EXPECT_EQ(auto_unroll.AnalyseApplyType(state, applied_block_name), RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states             = auto_unroll.ApplyOnBlock(state, applied_block_name);
  std::vector<ir::Expr> exprs = new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  // Check if the block has an 'auto_unroll_max_step' attribute
  auto* applied_block_realize  = exprs.front().As<ir::Block>()->stmts.front().As<ir::ScheduleBlockRealize>();
  auto* applied_schedule_block = applied_block_realize->schedule_block.As<ir::ScheduleBlock>();
  ASSERT_FALSE(applied_schedule_block->attrs.empty());
  EXPECT_EQ(applied_schedule_block->attrs.count(ir::attr::auto_unroll_max_step), 1);
  const auto& attr_value = applied_schedule_block->attrs.at(ir::attr::auto_unroll_max_step);
  const int* max_step    = absl::get_if<int>(&attr_value);
  EXPECT_NE(max_step, nullptr);
  EXPECT_LE(*max_step, 128);
  VLOG(6) << "Expr after AutoUnroll applied on block:max_step=" << *max_step << ", Ast:\n" << exprs.front();

  // build ir::Module and debug source code
  auto build_module = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code  = GenSourceCode(build_module);
  VLOG(6) << " auto-schedule source code:\n" << source_code;
  // execute and check precision
  CheckResult(GenExecutableKernel(build_module),
              GenExecutableKernel(BuildIRModule(MakeIRSchedule(matmul_op, /* apply_manual_schedule */ true))),
              default_input_names,
              default_output_names,
              {{4, 4}, {4, 4}},
              {{4, 4}},
              target_);
}

TEST_F(TestAutoUnroll, PureSpatial) {
  Target target = common::DefaultNVGPUTarget();
  Initialize(target);
  std::vector<std::string> input_names  = {"x", "y"};
  std::vector<std::string> output_names = {
      "var_6", "var_4", "constant_idx_last", "constant_idx_first", "var_2", "var_5"};
  std::vector<int32_t> input_shape{256, 256};
  std::vector<tests::VariableInfo> inputs_varinfo({{"x", input_shape}, {"y", input_shape}});

  Context::Global().ResetNameId();
  ir::IRSchedule ir_schedule = MakeIRSchedule(tests::GatherAddSubSubGraphBuilder().Build(inputs_varinfo));
  SearchState state(ir_schedule, 0, {});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  AutoUnroll auto_unroll(target_);
  for (const auto& applied_block_name : output_names) {
    EXPECT_EQ(auto_unroll.AnalyseApplyType(state, applied_block_name), RuleApplyType::kCannotApply);
  }
}
#endif

}  // namespace auto_schedule
}  // namespace cinn
