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

#include "cinn/auto_schedule/search_strategy/evolutionary_search.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>

#include "cinn/auto_schedule/search_space/search_space.h"
#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/tuning.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/utils/sized_multi_set.h"

namespace cinn {
namespace auto_schedule {

EvolutionarySearch::EvolutionarySearch(const TuneTask& tune_task) : tune_task_(tune_task) {
  search_space_ = std::make_unique<SearchSpace>(tune_task);
}

EvolutionarySearch::~EvolutionarySearch() {}

SearchState EvolutionarySearch::SearchModuleExpr(const TuningOptions& options) {
  return SearchModuleExprBests(options)[0];
}

std::vector<SearchState> EvolutionarySearch::SearchModuleExprBests(const TuningOptions& options) {
  std::vector<SearchState> init_population;
  std::vector<SearchState> topk_from_database = GetTopKCandidatesFromDatabase(options.evolution_pick_database_topk);
  VLOG(5) << "EvolutionarySearch got " << topk_from_database.size() << " as topk from database";
  int random_num = options.evolution_init_population_num - topk_from_database.size();

  VLOG(5) << "EvolutionarySearch will fetch " << random_num << " sketches";
  std::vector<SearchState> random_sketch = RandomInitSketch(random_num);

  init_population.insert(init_population.end(), topk_from_database.begin(), topk_from_database.end());
  init_population.insert(init_population.end(), random_sketch.begin(), random_sketch.end());

  VLOG(5) << "EvolutionarySearch got generation size " << init_population.size();
  std::vector<SearchState> picked_bests =
      Evolve(init_population, options.evolution_cross_over_num, options.num_samples_per_iteration);

  VLOG(5) << "EvolutionarySearch got evolved generation size " << picked_bests.size();
  return picked_bests;
}

std::vector<SearchState> EvolutionarySearch::SearchModuleExprEpsGreedy(const TuningOptions& options) {
  std::vector<SearchState> picked_bests = SearchModuleExprBests(options);
  int random_num                        = options.evolution_init_population_num - options.evolution_pick_database_topk;
  return PickNextGenerationEpsGreedy(
      picked_bests, RandomInitSketch(random_num), options.num_samples_per_iteration, options.evolution_eps_greedy);
}

std::vector<SearchState> EvolutionarySearch::GetTopKCandidatesFromDatabase(int topk) {
  // TODO(zhhsplendid): implement it after we have the database
  return std::vector<SearchState>();
}

std::vector<SearchState> EvolutionarySearch::RandomInitSketch(int num) {
  return search_space_->GetRandomInitialSketch(num);
}

SearchState EvolutionarySearch::CrossOver(const SearchState& state1, const SearchState& state2) {
  std::vector<ir::Expr> cross_over_exprs;
  std::vector<ir::Expr> father_exprs = state1.mod_expr.GetExprs();
  std::vector<ir::Expr> mother_exprs = state2.mod_expr.GetExprs();

  CHECK_EQ(father_exprs.size(), mother_exprs.size())
      << "CrossOver ModuleExpr in EvolutionarySearch must have same number of AST";

  for (size_t i = 0; i < father_exprs.size(); ++i) {
    if (rand() % 2 == 0) {
      cross_over_exprs.push_back(optim::IRCopy(father_exprs[i]));
    } else {
      cross_over_exprs.push_back(optim::IRCopy(mother_exprs[i]));
    }
  }
  return SearchState(ir::ModuleExpr(cross_over_exprs));
}

std::vector<SearchState> EvolutionarySearch::Evolve(const std::vector<SearchState>& population,
                                                    int cross_over_num,
                                                    int ret_num) {
  int generation_num = population.size();
  if (generation_num == 0) {
    return std::vector<SearchState>();
  }
  std::vector<SearchState> evolution(population);

  for (int i = 0; i < cross_over_num; ++i) {
    int first_rand_idx  = rand() % generation_num;
    int second_rand_idx = rand() % generation_num;
    while (first_rand_idx == second_rand_idx) {
      second_rand_idx = rand() % generation_num;
    }
    evolution.push_back(CrossOver(population[first_rand_idx], population[second_rand_idx]));
  }

  utils::SizedMultiSet<SearchState> evolution_with_cost(ret_num);
  for (size_t i = 0; i < evolution.size(); ++i) {
    evolution_with_cost.Push(search_space_->GetScheduleMutate(evolution[i], *cost_model_));
  }

  return evolution_with_cost.ReturnAsContainer<std::vector<SearchState>>();
}

std::vector<SearchState> EvolutionarySearch::PickNextGenerationEpsGreedy(const std::vector<SearchState>& picked_bests,
                                                                         const std::vector<SearchState>& random_init,
                                                                         int num,
                                                                         float eps_greedy) {
  int num_rands = num * eps_greedy;
  int num_bests = num - num_rands;

  std::vector<SearchState> result;
  int best_idx = 0;
  int rand_idx = 0;
  for (int i = 0; i < num; ++i) {
    if (i < num_bests && best_idx < picked_bests.size()) {
      result.push_back(picked_bests[best_idx]);
      ++best_idx;
    } else if (rand_idx < random_init.size()) {
      result.push_back(random_init[rand_idx]);
      ++rand_idx;
    } else if (best_idx < picked_bests.size()) {
      result.push_back(picked_bests[best_idx]);
      ++best_idx;
    } else {
      break;
    }
  }
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn
