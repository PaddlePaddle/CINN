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

#include <algorithm>
#include <random>

namespace cinn {
namespace auto_schedule {

std::unique_ptr<RuleScheduler> RuleScheduler::Make(const std::vector<AutoGenRule*>& potential_rules,
                                                   const std::string& strategy,
                                                   const std::vector<int>& weights) {
  CHECK_GT(potential_rules.size(), 0) << "Empty rule list";
  if (strategy == "traversal") {
    return std::make_unique<TraversalRuleScheduler>(potential_rules);
  } else if (strategy == "probabilistic") {
    return std::make_unique<ProbabilisticRuleScheduler>(potential_rules, weights);
  }

  LOG(FATAL) << "Unimplementd strategy:" << strategy;
  return nullptr;
}

AutoGenRule* TraversalRuleScheduler::NextRule() {
  if (cur_idx_ < potential_rules_->size()) {
    return potential_rules_->at(cur_idx_++);
  }

  return nullptr;
}

ProbabilisticRuleScheduler::ProbabilisticRuleScheduler(const std::vector<AutoGenRule*>& potential_rules,
                                                       const std::vector<int>& weights)
    : RuleScheduler(potential_rules), weights_(weights), gen_(rd_()) {
  if (weights.empty()) {
    weights_.resize(potential_rules.size(), 1);
  } else {
    CHECK_EQ(potential_rules.size(), weights_.size());
  }

  distribution_ = std::discrete_distribution<>(weights_.begin(), weights_.end());
}

AutoGenRule* ProbabilisticRuleScheduler::NextRule() { return potential_rules_->at(distribution_(gen_)); }

}  // namespace auto_schedule
}  // namespace cinn