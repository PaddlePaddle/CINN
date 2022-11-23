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

#include "cinn/auto_schedule/search_space/search_space.h"

#include <glog/logging.h>

#include <cstdlib>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_read.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_write.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/runtime/flags.h"

DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

SearchSpace::SearchSpace(const TuneTask& tune_task) : tune_task_(tune_task) {
  const auto& target = tune_task_.target;
  // initialize a set of rules and they are commonly used by all states
  // TODO(zhhsplendid): pass correct output names to AutoInline
  // sketch_rules_.emplace_back(new AutoInline(target, tune_task_.output_names));
  sketch_rules_.emplace_back(new MultiLevelTiling(target));
  sketch_rules_.emplace_back(new AddCacheRead(target));
  // sketch_rules_.emplace_back(new AddCacheWrite(target));
  sketch_rules_.emplace_back(new AutoUnroll(target));
  sketch_rules_.emplace_back(new SkipRule(target));
}

std::vector<SearchState> SearchSpace::GetRandomInitialSketch(int num) {
  VLOG(4) << "Start SearchSpace::GetRandomInitialSketch with num:" << num;
  ir::IRSchedule init_schedule(ir::ModuleExpr(tune_task_.GetLoweredFuncBodyExprs()));
  std::vector<AutoGenRule*> init_rules;
  std::transform(sketch_rules_.begin(), sketch_rules_.end(), std::back_inserter(init_rules), [](const auto& rule) {
    return rule.get();
  });

  std::vector<SearchState> result;
  while (result.size() < num) {
    SearchState state(init_schedule, SearchState::NOT_INIT_COST, init_rules);
    for (int i = 0; i < init_sketch_random_depth_; ++i) {
      VLOG(5) << "Generating random sketch at depth: " << i;
      state = RandomScheduleMutate(state);
      if (state->applicable_rules.empty()) {
        break;
      }
    }
    // TODO:(zhhsplendid): De-duplication on the result after we have Expr/ModuleExpr hash;
    auto debug_str = state->DebugString();
    VLOG(5) << utils::StringFormat("Sketch-%lu generated, SearchState hash:%lu, DebugString:%s",
                                   result.size(),
                                   std::hash<std::string>()(debug_str),
                                   debug_str.c_str());
    result.emplace_back(std::move(state));
  }

  return result;
}

SearchState SearchSpace::GetScheduleMutate(const SearchState& state, const ExprCostModel& cost_model) {
  VLOG(5) << "Start SearchSpace::GetScheduleMutate in state:" << std::hash<std::string>()(state->DebugString());
  bool has_manual_schedule = false;
  if (has_manual_schedule) {
    SearchState ret = ManualScheduleMutate(state);
    return ret;
  }
  SearchState ret = RandomScheduleMutate(state);
  if (FLAGS_auto_schedule_use_cost_model) {
    ret->predicted_cost = cost_model.Predict(ret->ir_schedule.GetModule(), tune_task_.target);
  }
  return ret;
}

SearchState SearchSpace::ManualScheduleMutate(const SearchState& state) {
  // TODO(zhhsplendid): Add manual schedule mutate
  return state;
}

SearchState SearchSpace::RandomScheduleMutate(const SearchState& state) {
  VLOG(5) << "Start SearchSpace::RandomScheduleMutate";

  // 1. Found the schedules which can apply on this Expr
  // 2. Make a distribution on those schedules
  std::map<int, int> weight_to_rule_index;
  int cur_weight = 0;
  SearchState ret(state);
  std::vector<RuleApplyType> apply_types(ret->applicable_rules.size());
  for (int idx = 0; idx != ret->applicable_rules.size(); ++idx) {
    AutoGenRule* rule        = ret->applicable_rules.at(idx);
    RuleApplyType apply_type = rule->Init(&ret->ir_schedule);
    VLOG(6) << "Evaluate rule:" << rule->GetRuleName() << "=" << static_cast<int>(apply_type);
    apply_types[idx] = apply_type;
    if (apply_type != RuleApplyType::kCannotApply) {
      weight_to_rule_index[cur_weight] = idx;
      cur_weight += rule->NumberApplicable();
    }
  }

  if (weight_to_rule_index.empty()) {
    // No applicable rule, return the input mod_expr
    VLOG(6) << "No applicable rule";
    return ret;
  }

  // 3. Sample a schedule on the distribution
  int sample_weighted_index = rand() % cur_weight;

  auto iter = weight_to_rule_index.lower_bound(sample_weighted_index);
  if (iter->first > sample_weighted_index) {
    // weight_to_rule must contain key 0, and sample_index >= 0, so --iter won't exceed the beginning.
    --iter;
  }
  int sample_rule_index = iter->second;
  CHECK_LT(sample_rule_index, ret->applicable_rules.size());
  AutoGenRule* sample_rule = ret->applicable_rules.at(sample_rule_index);
  VLOG(6) << "Apply rule: " << sample_rule->GetRuleName() << " with index=" << sample_weighted_index - iter->first;
  // 4. Apply the schedule change
  sample_rule->Apply(sample_weighted_index - iter->first);

  // 5. Remove the rule after applying it
  if (apply_types.at(sample_rule_index) == RuleApplyType::kApplyAndSkipThisRule) {
    ret->applicable_rules.erase(ret->applicable_rules.begin() + sample_rule_index);
  } else if (apply_types.at(sample_rule_index) == RuleApplyType::kApplyAndSkipAllRules) {
    ret->applicable_rules.clear();
  }

  return ret;
}

}  // namespace auto_schedule
}  // namespace cinn
