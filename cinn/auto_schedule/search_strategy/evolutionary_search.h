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
#include "cinn/auto_schedule/task/tune_context.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * Class implement the evolutionary search on ModuleExpr search space.
 */
class EvolutionarySearch {
 public:
  /**
   * Default constructor. Note that this class doesn't set TuneContext, you should
   * call EvolutionarySearch::SetTuneContext to bind a TuneContext.
   */
  EvolutionarySearch() = default;

  /**
   * constutor with TuneContext.
   *
   * @param tune_context: the TuneContext this class works on. This class doesn't
   *     take ownership of the pointer.
   */
  EvolutionarySearch(TuneContext* tune_context);

  /**
   * Destructor
   */
  ~EvolutionarySearch();

  /**
   * Set the TuneContext this class works on. This class doesn't take ownership
   * of the pointer.
   *
   * @param tune_context: the TuneContext this class works on.
   */
  void SetTuneContext(TuneContext* tune_context);

  /**
   * Run the evolutionary search for one iteration.
   *
   * @return the best ir::ModuleExpr searched in this iteration
   */
  ir::ModuleExpr SearchModuleExpr();

  /**
   * Run the evolutionary search for one iteration.
   *
   * @return those best ir::ModuleExpr's searched in this iteration
   */
  std::vector<ir::ModuleExpr> SearchModuleExprBests();

  /**
   * Run the evolutionary search for one iteration, but since evolutionary
   * search with cost model may not be accurate, this method picks
   * "eps * total_return_size" random samples along with those best
   * ir::ModuleExpr's searched in this iteration.
   *
   * @return those best ir::ModuleExpr's searched in this iteration and
   *     some random samples. There are "eps * total_return_size" random
   *     samples and "(1 - eps) * total_return_size" best searched samples.
   */
  std::vector<ir::ModuleExpr> SearchModuleExprEpsGreedy();

#ifdef CINN_WITH_TEST
  /**
   * Method only be called during testing. It is used to set mock search
   * space.
   *
   * @param search_space: the mock search space, note that EvolutionarySearch
   *     takes the ownership.
   */
  void SetSearchSpace(SearchSpace* search_space) { search_space_.reset(search_space); }
#endif

 private:
  std::vector<ir::ModuleExpr> GetTopKCandidatesFromDatabase(int topk);

  std::vector<ir::ModuleExpr> RandomInitSketch(int num);

  ir::ModuleExpr CrossOver(const ir::ModuleExpr& mod_expr1, const ir::ModuleExpr& mod_expr2);

  std::vector<ir::ModuleExpr> Evolve(const std::vector<ir::ModuleExpr>& population, int num);

  std::vector<ir::ModuleExpr> PickNextGenerationEpsGreedy(const std::vector<ir::ModuleExpr>& population,
                                                          const std::vector<ir::ModuleExpr>& random_init,
                                                          int num,
                                                          float eps_greedy);

  int database_topk_       = 8;
  int init_population_num_ = 10;
  int cross_over_num_      = 10;
  int sample_num_          = 10;

  float eps_greedy_ = 0.0f;

  std::unique_ptr<SearchSpace> search_space_;

  TuneContext* tune_context_;

  CostModel* cost_model_;  // not owned
};

}  // namespace auto_schedule
}  // namespace cinn
