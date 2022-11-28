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

#pragma once

#include <memory>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/ir/ir_base.h"

namespace cinn {
namespace auto_schedule {

class SearchState;

class RuleScheduler {
 public:
  static std::unique_ptr<RuleScheduler> Make(const std::vector<AutoGenRule*>& potential_rules,
                                             const std::string& strategy     = "traversal",
                                             const std::vector<int>& weights = {});

  virtual const char* Name() const = 0;

  virtual void Reset() = 0;

  virtual AutoGenRule* NextRule(SearchState state, const std::string& block_name) = 0;

 protected:
  RuleScheduler(const std::vector<AutoGenRule*>& potential_rules) : potential_rules_(&potential_rules) {}

  const std::vector<AutoGenRule*>* potential_rules_;
};

class TraversalRuleScheduler : public RuleScheduler {
 public:
  TraversalRuleScheduler(const std::vector<AutoGenRule*>& potential_rules)
      : RuleScheduler(potential_rules), cur_idx_(0) {}

  const char* Name() const override { return "traversal"; }

  void Reset() override { cur_idx_ = 0; }

  AutoGenRule* NextRule(SearchState state, const std::string& block_name) override;

 private:
  int cur_idx_;
};

class ProbabilisticRuleScheduler : public RuleScheduler {
 public:
  ProbabilisticRuleScheduler(const std::vector<AutoGenRule*>& potential_rules, const std::vector<int>& weights = {});

  const char* Name() const override { return "probabilistic"; }

  void Reset() override {}

  AutoGenRule* NextRule(SearchState state, const std::string& block_name) override;

 private:
  std::vector<int> weights_;
};

}  // namespace auto_schedule
}  // namespace cinn
