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
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "cinn/auto_schedule/search_space/search_space.h"
#include "cinn/auto_schedule/search_space/search_state.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/auto_schedule/tuning.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/utils/sized_multi_set.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

EvolutionarySearch::EvolutionarySearch(const TuneTask& tune_task, const ExprCostModel& cost_model)
    : tune_task_(tune_task), cost_model_(cost_model) {
  search_space_ = std::make_unique<SearchSpace>(tune_task);
}

EvolutionarySearch::~EvolutionarySearch() {}

void PrintStates(const int vlog_level, const std::string& phase_name, const std::vector<SearchState>& states) {
  if (!VLOG_IS_ON(vlog_level)) {
    return;
  }
  VLOG(vlog_level) << "EvolutionarySearch-" << phase_name << " states size:" << states.size();
  for (auto i = 0; i < states.size(); ++i) {
    auto debug_str = states[i].DebugString();
    VLOG(vlog_level) << "State-" << i << " hash:" << std::hash<std::string>()(debug_str);
    VLOG(vlog_level + 1) << "****** State-" << i << " Detail ******\n" << debug_str;
    VLOG(vlog_level + 1) << "****** SearchState End ******";
  }
}

SearchState EvolutionarySearch::SearchModuleExpr(const TuningOptions& options) {
  return SearchModuleExprBests(options)[0];
}

std::vector<SearchState> EvolutionarySearch::SearchModuleExprBests(const TuningOptions& options) {
  std::vector<SearchState> init_population;
  std::vector<SearchState> topk_from_database = GetTopKCandidatesFromDatabase(options.evolution_pick_database_topk);
  PrintStates(4, "GetTopKCandidatesFromDatabase", topk_from_database);
  int random_num = options.evolution_init_population_num - topk_from_database.size();

  std::vector<SearchState> random_sketch = RandomInitSketch(random_num);
  PrintStates(4, "RandomInitSketch", random_sketch);

  init_population.insert(init_population.end(), topk_from_database.begin(), topk_from_database.end());
  init_population.insert(init_population.end(), random_sketch.begin(), random_sketch.end());

  VLOG(4) << "EvolutionarySearch got init generation size " << init_population.size();
  std::vector<SearchState> picked_bests =
      Evolve(init_population, options.evolution_cross_over_num, options.num_samples_per_iteration);
  PrintStates(4, "Evolve", picked_bests);

  return picked_bests;
}

std::vector<SearchState> EvolutionarySearch::SearchModuleExprEpsGreedy(const TuningOptions& options) {
  std::vector<SearchState> picked_bests = SearchModuleExprBests(options);
  int random_num                        = options.evolution_init_population_num - options.evolution_pick_database_topk;
  auto results                          = PickNextGenerationEpsGreedy(
      picked_bests, RandomInitSketch(random_num), options.num_samples_per_iteration, options.evolution_eps_greedy);
  PrintStates(4, "PickNextGenerationEpsGreedy", results);
  return results;
}

std::vector<SearchState> EvolutionarySearch::GetTopKCandidatesFromDatabase(int topk) {
  // TODO(zhhsplendid): implement it after we have the database
  VLOG(4) << "GetTopKCandidatesFromDatabase topk:" << topk;
  return std::vector<SearchState>();
}

std::vector<SearchState> EvolutionarySearch::RandomInitSketch(int num) {
  VLOG(4) << "RandomInitSketch is fetching " << num << " sketches";
  return search_space_->GetRandomInitialSketch(num);
}

SearchState EvolutionarySearch::CrossOver(const SearchState& state1, const SearchState& state2) {
  PrintStates(5, "CrossOver", {state1, state2});
  // TODO(CtfGo): tracing CrossOver with IRSchedule
  std::vector<ir::Expr> cross_over_exprs;
  std::vector<ir::Expr> father_exprs = state1.ir_schedule.GetModule().GetExprs();
  std::vector<ir::Expr> mother_exprs = state2.ir_schedule.GetModule().GetExprs();

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
  VLOG(4) << utils::StringFormat(
      "Evolve with population size=%lu,cross_over_num:%lu,ret_num:%lu", population.size(), cross_over_num, ret_num);
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
    evolution_with_cost.Push(search_space_->GetScheduleMutate(evolution[i], cost_model_));
  }

  return evolution_with_cost.ReturnAsContainer<std::vector<SearchState>>();
}

std::vector<SearchState> EvolutionarySearch::PickNextGenerationEpsGreedy(const std::vector<SearchState>& picked_bests,
                                                                         const std::vector<SearchState>& random_init,
                                                                         int num,
                                                                         float eps_greedy) {
  VLOG(4) << utils::StringFormat(
      "PickNextGenerationEpsGreedy with picked_bests size=%lu,random_init size=%lu,num:%lu,eps_greedy:%f",
      picked_bests.size(),
      random_init.size(),
      num,
      eps_greedy);
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
