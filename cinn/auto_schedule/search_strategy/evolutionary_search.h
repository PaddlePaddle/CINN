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

#include "cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "cinn/auto_schedule/database/database.h"
#include "cinn/auto_schedule/search_space/search_space.h"
#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/tuning.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * Class implement the evolutionary search on ModuleExpr search space.
 */
class EvolutionarySearch {
 public:
  /**
   * constutor with TuneTask.
   *
   * @param tune_task: the TuneTask this class works on. This class doesn't
   *     take ownership of the pointer.
   */
  EvolutionarySearch(const TuneTask& tune_task, const ExprCostModel& cost_model, Database* database);

  /**
   * Destructor
   */
  ~EvolutionarySearch();

  /**
   * Run the evolutionary search for one iteration.
   *
   * @return SearchState containing the best ir::ModuleExpr searched in this iteration
   */
  SearchState SearchModuleExpr(const TuningOptions& options);

  /**
   * Run the evolutionary search for one iteration.
   *
   * @return SearchState(s) containing best ir::ModuleExpr(s) searched in this iteration
   */
  std::vector<SearchState> SearchModuleExprBests(const TuningOptions& options);

  /**
   * Run the evolutionary search for one iteration, but since evolutionary
   * search with cost model may not be accurate, this method picks
   * "eps * total_return_size" random samples along with those best
   * ir::ModuleExpr's searched in this iteration.
   *
   * @return SearchSpace containing those best ir::ModuleExpr's searched
   *     in this iteration and some random samples. There are
   *     "eps * total_return_size" random samples and
   *     "(1 - eps) * total_return_size" best searched samples.
   */
  std::vector<SearchState> SearchModuleExprEpsGreedy(const TuningOptions& options);

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
  std::vector<SearchState> GetTopKCandidatesFromDatabase(int topk);

  /**
   * \brief Generate sketch as initial population of evolutionary search.
   * @param num The number of sketches to generate.
   * @param strategy The strategy to generate sketchs,
   *        Current optional strategies are "rule_prune" or "random_prune" or "random".
   * - "rule_prune": will use rules to prune and generate sketches as efficiently as possible.
   * - "random_prune": will use the new interface ApplySketchRules() to simulate the random generation of sketches,
   *    and supports the function of a rule returning multiple SearchStates and random pruning by probability.
   * - "random": will randomly select a block and a rule to apply and repeat this step several times,
   *    however, each rule can only be used on one SearchState at most once.
   * @return  Generated sketchs.
   */
  std::vector<SearchState> InitSketch(int num, const std::string& strategy);

  SearchState CrossOver(const SearchState& state1, const SearchState& state2);

  std::vector<SearchState> Evolve(const std::vector<SearchState>& population, int cross_over_num, int ret_num);

  std::vector<SearchState> PickNextGenerationEpsGreedy(const std::vector<SearchState>& population,
                                                       const std::vector<SearchState>& random_init,
                                                       int num,
                                                       float eps_greedy);

  std::unique_ptr<SearchSpace> search_space_;

  const TuneTask& tune_task_;

  const ExprCostModel& cost_model_;  // not owned

  Database* database_;  // not owned
};

}  // namespace auto_schedule
}  // namespace cinn
