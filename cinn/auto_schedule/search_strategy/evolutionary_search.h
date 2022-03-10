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

#include "cinn/auto_schedule/search_space/search_space.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class EvolutionarySearch {
 public:
  EvolutionarySearch() = default;
  EvolutionarySearch(TuneTask* tune_task);
  ~EvolutionarySearch();

  void SetTuneTask(TuneTask* tune_task);
  ir::ModuleExpr GetAutoTuneModuleExpr();
  std::vector<ir::ModuleExpr> GetAutoTuneModuleExprBests();
  std::vector<ir::ModuleExpr> GetAutoTuneEpsGreedy();

#ifdef CINN_WITH_TEST
  void SetSearchSpace(SearchSpace* search_space) { search_space_.reset(search_space); }
#endif

 private:
  std::vector<ir::ModuleExpr> GetTopKCandidatesFromDatabase(int topk);

  std::vector<ir::ModuleExpr> RandomInitSketch(int num);

  ir::ModuleExpr CrossOver(const ir::ModuleExpr& mod_expr1, const ir::ModuleExpr& mod_expr2);

  std::vector<ir::ModuleExpr> Evolve(const std::vector<ir::ModuleExpr>& population, int num);

  std::vector<ir::ModuleExpr> PickNextGeneration(const std::vector<ir::ModuleExpr>& population,
                                                 const std::vector<ir::ModuleExpr>& random_init,
                                                 int num,
                                                 float eps_greedy);

  int database_topk_       = 8;
  int init_population_num_ = 10;
  int cross_over_num_      = 10;
  int sample_num_          = 10;

  float eps_greedy_ = 0.0f;

  std::unique_ptr<SearchSpace> search_space_;

  TuneTask* tune_task_;

  CostModel* cost_model_;  // not owned
};

}  // namespace auto_schedule
}  // namespace cinn
