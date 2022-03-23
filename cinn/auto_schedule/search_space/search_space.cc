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

#include <utility>
#include <vector>

#include "cinn/auto_schedule/cost_model/cost_model.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

std::vector<ir::ModuleExpr> SearchSpace::GetRandomInitialSketch(int num) { return std::vector<ir::ModuleExpr>(); };

std::pair<ir::ModuleExpr, float> SearchSpace::GetScheduleMutate(const CostModel& cost_model,
                                                                const ir::ModuleExpr& mod_expr) {
  return std::make_pair<ir::ModuleExpr, float>(ir::ModuleExpr(), 0.0f);
};
};  // namespace auto_schedule

}  // namespace cinn
}  // namespace cinn
