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

namespace cinn {
namespace auto_schedule {

#ifdef CINN_WITH_CUDA
Target target = common::DefaultNVGPUTarget();
#else
Target target = common::DefaultHostTarget();
#endif

std::vector<AutoGenRule*> GenerateTestRules() { return {new AutoUnroll(target), new SkipRule(target)}; }

TEST(RuleScheduler, Make) {
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto traversal_block_scheduler  = RuleScheduler::Make(rules, "traversal");
  ASSERT_STREQ(traversal_block_scheduler->Name(), "traversal");
  auto probabilistic_block_scheduler = RuleScheduler::Make(rules, "probabilistic");
  ASSERT_STREQ(probabilistic_block_scheduler->Name(), "probabilistic");
}

TEST(TraversalRuleScheduler, NextRule) {
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto traversal_rule_scheduler   = RuleScheduler::Make(rules, "traversal");
  AutoGenRule* rule               = traversal_rule_scheduler->NextRule();
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
  rule = traversal_rule_scheduler->NextRule();
  ASSERT_EQ("SkipRule", rule->GetRuleName());
  traversal_rule_scheduler->Reset();
  rule = traversal_rule_scheduler->NextRule();
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
}

TEST(ProbabilisticRuleScheduler, NextRule) {
  std::vector<AutoGenRule*> rules   = GenerateTestRules();
  auto probabilistic_rule_scheduler = RuleScheduler::Make(rules, "probabilistic", {4, 1});
  AutoGenRule* rule;
  for (int i = 0; i < 20; ++i) {
    rule = probabilistic_rule_scheduler->NextRule();
    VLOG(6) << "next rule name: " << rule->GetRuleName();
  }
}

}  // namespace auto_schedule
}  // namespace cinn
