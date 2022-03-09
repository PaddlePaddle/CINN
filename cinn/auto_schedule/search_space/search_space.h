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

#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/cost_model/cost_model.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * This class is an abstraction of the transformations can be applied to
 * ir::Expr during auto-tuning. The transformation can be:
 *
 * 1. Manual defined schedule
 * 2. Schedule generated by AutoGenRule
 *
 * TODO(zhhsplendid): Finish the detail implementation of this class after
 * we have low-level IR.
 */
class SearchSpace {
 public:
  SearchSpace(const TuneTask& tune_task){

  };

  // Generate sketch as initial population of evolutionary search
  std::vector<ir::ModuleExpr> GetRandomInitialSketch(int num){};

  // Evolutionary search mutate, returns the mutated ModuleExpr and estimited cost
  std::pair<ir::ModuleExpr, float> GetScheduleMutate(const CostModel& cost_model, const ir::ModuleExpr& mod_expr){};
};

}  // namespace auto_schedule
}  // namespace cinn
