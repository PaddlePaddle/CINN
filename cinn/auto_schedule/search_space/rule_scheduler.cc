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

#include "cinn/ir/ir.h"

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

AutoGenRule* TraversalRuleScheduler::NextRule(SearchState state, const std::string& block_name) {
  while (cur_idx_ < potential_rules_->size()) {
    AutoGenRule* rule = potential_rules_->at(cur_idx_);
    VLOG(6) << "xb_debug rule = " << rule->GetRuleName();
    RuleApplyType apply_type = rule->AnalyseApplyType(state, block_name);
    if (apply_type == RuleApplyType::kApply) {
      return rule;
    } else if (apply_type == RuleApplyType::kApplyAndSkipThisRule) {
      ++cur_idx_;
      return rule;
    } else if (apply_type == RuleApplyType::kApplyAndSkipAllRules) {
      cur_idx_ = potential_rules_->size();
      return rule;
    } else if (apply_type == RuleApplyType::kCannotApply) {
      ++cur_idx_;
    }
  }

  return nullptr;
}

ProbabilisticRuleScheduler::ProbabilisticRuleScheduler(const std::vector<AutoGenRule*>& potential_rules,
                                                       const std::vector<int>& weights)
    : RuleScheduler(potential_rules), weights_(weights) {
  if (weights_.empty()) {
    weights_.resize(potential_rules.size(), 1);
  } else {
    CHECK_EQ(potential_rules.size(), weights_.size());
  }
}

AutoGenRule* ProbabilisticRuleScheduler::NextRule(SearchState state, const std::string& block_name) {
  std::vector<int> weights_applicable;
  std::vector<std::pair<int, RuleApplyType>> index2type;
  for (int i = 0; i < potential_rules_->size(); ++i) {
    AutoGenRule* rule        = potential_rules_->at(i);
    RuleApplyType apply_type = rule->AnalyseApplyType(state, block_name);
    if (weights_.at(i) != 0 && apply_type != RuleApplyType::kCannotApply) {
      weights_applicable.push_back(weights_.at(i));
      index2type.push_back(std::make_pair(i, apply_type));
    }
  }
  if (weights_applicable.empty()) {
    return nullptr;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dis(weights_applicable.begin(), weights_applicable.end());
  std::pair<int, cinn::auto_schedule::RuleApplyType> rule_index_type = index2type.at(dis(gen));
  if (rule_index_type.second == RuleApplyType::kApplyAndSkipThisRule) {
    weights_[rule_index_type.first] = 0;
  } else if (rule_index_type.second == RuleApplyType::kApplyAndSkipAllRules) {
    weights_.resize(potential_rules_->size(), 0);
  }

  return potential_rules_->at(rule_index_type.first);
}

}  // namespace auto_schedule
}  // namespace cinn