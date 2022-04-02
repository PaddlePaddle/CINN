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

#include "cinn/auto_schedule/cost_model/cost_model.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

SearchSpace::SearchSpace(const TuneTask& tune_task) : tune_task_(tune_task) {}

std::vector<ir::ModuleExpr> SearchSpace::GetRandomInitialSketch(int num) {
  std::vector<ir::ModuleExpr> result;
  while (result.size() < num) {
    std::vector<std::shared_ptr<AutoGenRule>> candidate_rules = auto_gen_rules_;
    ir::ModuleExpr mod_expr                                   = tune_task_.tune_context().module;
    for (int i = 0; i < init_sketch_random_depth_; ++i) {
      mod_expr = RandomScheduleMutate(mod_expr, &candidate_rules);
      if (candidate_rules.empty()) {
        break;
      }
    }
    // TODO:(zhhsplendid): De-duplication on the result after we have Expr/ModuleExpr hash;
    result.emplace_back(std::move(mod_expr));
  }
  return result;
}

std::pair<ir::ModuleExpr, float> SearchSpace::GetScheduleMutate(const CostModel& cost_model,
                                                                const ir::ModuleExpr& mod_expr) {
  // TODO(zhhsplendid): cost model predict
  bool has_manual_schedule = false;
  if (has_manual_schedule) {
    ir::ModuleExpr manual_expr = ManualScheduleMutate(mod_expr);
    return std::make_pair<ir::ModuleExpr, float>(std::move(manual_expr), 0.0f);
  }

  std::vector<std::shared_ptr<AutoGenRule>> candidate_rules = auto_gen_rules_;
  ir::ModuleExpr random_expr                                = RandomScheduleMutate(mod_expr, &candidate_rules);
  return std::make_pair<ir::ModuleExpr, float>(std::move(random_expr), 0.0f);
}

ir::ModuleExpr SearchSpace::ManualScheduleMutate(const ir::ModuleExpr& mod_expr) {
  // TODO(zhhsplendid): Add manual schedule mutate
  return ir::ModuleExpr(mod_expr);
}

ir::ModuleExpr SearchSpace::RandomScheduleMutate(const ir::ModuleExpr& mod_expr,
                                                 std::vector<std::shared_ptr<AutoGenRule>>* candidate_rules) {
  // 1. Found the schedules which can apply on this Expr
  // 2. Make a distribution on those schedules
  std::map<int, std::shared_ptr<AutoGenRule>> weight_to_rule;
  int cur_weight = 0;
  for (auto iter = candidate_rules->begin(); iter != candidate_rules->end(); ++iter) {
    std::shared_ptr<AutoGenRule> rule = *iter;
    RuleApplyType apply_type          = rule->Init(mod_expr);
    if (apply_type != RuleApplyType::kCannotApply) {
      weight_to_rule[cur_weight] = rule;
      cur_weight += rule->NumberApplicable();
      if (apply_type == RuleApplyType::kApplyAndSkipThisRule) {
        candidate_rules->erase(iter);
      } else if (apply_type == RuleApplyType::kApplyAndSkipAllRules) {
        candidate_rules->clear();
      }
    }
  }

  if (weight_to_rule.empty()) {
    // No applicable rule, return the input mod_expr
    return mod_expr;
  }

  // 3. Sample a schedule on the distribution
  int sample_index                         = rand() % cur_weight;
  auto iter                                = weight_to_rule.lower_bound(sample_index);
  std::shared_ptr<AutoGenRule> sample_rule = iter->second;

  // 4. Apply the schedule change
  return sample_rule->Apply(sample_index - iter->first);
}

}  // namespace auto_schedule
}  // namespace cinn
