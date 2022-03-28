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

std::vector<ir::ModuleExpr> SearchSpace::GetRandomInitialSketch(int num) { return std::vector<ir::ModuleExpr>(); }

std::pair<ir::ModuleExpr, float> SearchSpace::GetScheduleMutate(const CostModel& cost_model,
                                                                const ir::ModuleExpr& mod_expr) {
  // TODO(zhhsplendid): finish the logic that adds manual schedule
  bool has_manual_schedule = false;
  if (has_manual_schedule) {
    ir::ModuleExpr manual_expr = ManualScheduleMutate(mod_expr);
    // TODO(zhhsplendid): cost model predict on the manual mutate
    return std::make_pair<ir::ModuleExpr, float>(std::move(manual_expr), 0.0f);
  }

  return std::make_pair<ir::ModuleExpr, float>(ir::ModuleExpr(), 0.0f);
}

ir::ModuleExpr SearchSpace::ManualScheduleMutate(const ir::ModuleExpr& mod_expr) { return ir::ModuleExpr(mod_expr); }

ir::ModuleExpr SearchSpace::RandomScheduleMutate(const ir::ModuleExpr& mod_expr) {
  // 1. Found the schedules which can apply on this Expr
  // 2. Make a distribution on those schedules
  std::map<int, AutoGenRule*> weight_to_rule;
  int cur_weight = 0;
  for (AutoGenRule& rule : auto_gen_rules) {
    if (rule.Init(mod_expr)) {
      weight_to_rule[cur_weight] = &rule;
      cur_weight += rule.NumberApplicable();
    }
  }

  // 3. Sample a schedule on the distribution
  int sample_index         = rand() % cur_weight;
  auto iter                = weight_to_rule.lower_bound(sample_index);
  AutoGenRule* sample_rule = iter->second;

  // 4. Apply the schedule change
  return sample_rule->Apply(sample_index - iter->first);
}

}  // namespace auto_schedule

}  // namespace cinn
