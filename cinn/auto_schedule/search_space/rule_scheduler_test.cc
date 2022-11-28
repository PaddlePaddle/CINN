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

#include "cinn/auto_schedule/search_space/rule_scheduler.h"

#include <gtest/gtest.h>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir.h"

namespace cinn {
namespace auto_schedule {

#ifdef CINN_WITH_CUDA
Target target = common::DefaultNVGPUTarget();
#else
Target target = common::DefaultHostTarget();
#endif

std::vector<AutoGenRule*> GenerateTestRules() { return {new AutoUnroll(target), new SkipRule(target)}; }

SearchState GenerateTestState() {
  ir::Expr M(32);
  ir::Expr N(32);
  ir::Expr K(32);

  // matmul case
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages              = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec("test_func", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr matmul_expr = funcs[0]->body;
  ir::IRSchedule ir_schedule(ir::ModuleExpr({matmul_expr}));

  return SearchState(ir_schedule, 0);
}

TEST(RuleScheduler, Make) {
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto traversal_block_scheduler  = RuleScheduler::Make(rules, "traversal");
  ASSERT_STREQ(traversal_block_scheduler->Name(), "traversal");
  auto probabilistic_block_scheduler = RuleScheduler::Make(rules, "probabilistic");
  ASSERT_STREQ(probabilistic_block_scheduler->Name(), "probabilistic");
}

TEST(TraversalRuleScheduler, NextRule) {
  SearchState state                 = GenerateTestState();
  std::vector<ir::Expr> block_exprs = state->ir_schedule.GetAllBlocks();
  // for (auto b : block_exprs) {
  //   ir::ScheduleBlockRealize* br = b.As<ir::ScheduleBlockRealize>();
  //   ir::ScheduleBlock* bl = br->schedule_block.As<ir::ScheduleBlock>();
  //   VLOG(6) << "xb_debug block name = " << bl->name;
  // }
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto traversal_rule_scheduler   = RuleScheduler::Make(rules, "traversal");
  AutoGenRule* rule               = traversal_rule_scheduler->NextRule(state, "C");
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
  rule = traversal_rule_scheduler->NextRule(state, "C");
  ASSERT_EQ("SkipRule", rule->GetRuleName());
  rule = traversal_rule_scheduler->NextRule(state, "C");
  ASSERT_EQ("SkipRule", rule->GetRuleName());
  traversal_rule_scheduler->Reset();
  rule = traversal_rule_scheduler->NextRule(state, "C");
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
}

TEST(ProbabilisticRuleScheduler, NextRule) {
  SearchState state                 = GenerateTestState();
  std::vector<AutoGenRule*> rules   = GenerateTestRules();
  auto probabilistic_rule_scheduler = RuleScheduler::Make(rules, "probabilistic", {4, 1});
  AutoGenRule* rule;
  for (int i = 0; i < 20; ++i) {
    rule = probabilistic_rule_scheduler->NextRule(state, "C");
    VLOG(6) << "next rule name: " << rule->GetRuleName();
  }
}

}  // namespace auto_schedule
}  // namespace cinn
