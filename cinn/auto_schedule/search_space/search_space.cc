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
#include "cinn/auto_schedule/task/tune_context.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace auto_schedule {

SearchSpace::SearchSpace(const TuneContext& tune_context) : tune_context_(tune_context) {}

std::vector<SearchState> SearchSpace::GetRandomInitialSketch(int num) {
  VLOG(4) << "Start SearchSpace::GetRandomInitialSketch";
  std::vector<SearchState> result;
  while (result.size() < num) {
    std::vector<ir::Expr> body_exprs = tune_context_.GetLoweredFuncBodyExprs();
    std::vector<ir::Expr> copy_exprs;
    for (const ir::Expr& e : body_exprs) {
      copy_exprs.push_back(optim::IRCopy(e));
    }
    SearchState state(std::move(ir::ModuleExpr(copy_exprs)));
    state.InitAutoGenRules(tune_context_.target, tune_context_.output_names);
    for (int i = 0; i < init_sketch_random_depth_; ++i) {
      VLOG(5) << "Generating random sketch at depth: " << i;
      state = RandomScheduleMutate(state);
      if (state.applicable_rules.empty()) {
        break;
      }
    }
    // TODO:(zhhsplendid): De-duplication on the result after we have Expr/ModuleExpr hash;
    result.emplace_back(std::move(state));
  }
  return result;
}

SearchState SearchSpace::GetScheduleMutate(const SearchState& state, const CostModel& cost_model) {
  VLOG(4) << "Start SearchSpace::GetScheduleMutate";
  // TODO(zhhsplendid): cost model predict
  bool has_manual_schedule = false;
  if (has_manual_schedule) {
    SearchState ret = ManualScheduleMutate(state);
    return ret;
  }
  SearchState ret = RandomScheduleMutate(state);
  return ret;
}

SearchState SearchSpace::ManualScheduleMutate(const SearchState& state) {
  // TODO(zhhsplendid): Add manual schedule mutate
  return state;
}

SearchState SearchSpace::RandomScheduleMutate(const SearchState& state) {
  VLOG(4) << "Start SearchSpace::RandomScheduleMutate";

  // 1. Found the schedules which can apply on this Expr
  // 2. Make a distribution on those schedules
  std::map<int, std::shared_ptr<AutoGenRule>> weight_to_rule;
  int cur_weight = 0;
  SearchState ret(state);
  for (auto iter = ret.applicable_rules.begin(); iter != ret.applicable_rules.end();) {
    std::shared_ptr<AutoGenRule> rule = *iter;
    VLOG(6) << "Rule name = " << rule->GetRuleName();
    RuleApplyType apply_type = rule->Init(ret.mod_expr);
    if (apply_type != RuleApplyType::kCannotApply) {
      weight_to_rule[cur_weight] = rule;
      cur_weight += rule->NumberApplicable();
      if (apply_type == RuleApplyType::kApplyAndSkipThisRule) {
        iter = ret.applicable_rules.erase(iter);
        continue;
      } else if (apply_type == RuleApplyType::kApplyAndSkipAllRules) {
        ret.applicable_rules.clear();
        break;
      }
    }
    ++iter;
  }

  if (weight_to_rule.empty()) {
    // No applicable rule, return the input mod_expr
    VLOG(6) << "No applicable rule";
    return ret;
  }

  // 3. Sample a schedule on the distribution
  int sample_index                         = rand() % cur_weight;
  auto iter                                = weight_to_rule.lower_bound(sample_index);
  std::shared_ptr<AutoGenRule> sample_rule = iter->second;
  VLOG(6) << "Sample AutoGenRule " << sample_rule->GetRuleName();

  // 4. Apply the schedule change
  ret.mod_expr = sample_rule->Apply(sample_index - iter->first);
  return ret;
}

}  // namespace auto_schedule
}  // namespace cinn
