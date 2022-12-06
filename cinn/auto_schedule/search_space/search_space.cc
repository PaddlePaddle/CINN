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

#include "cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"
#include "cinn/auto_schedule/search_space/block_sampler.h"
#include "cinn/auto_schedule/search_space/rule_sampler.h"
#include "cinn/auto_schedule/task/tune_task.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/runtime/flags.h"

DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

SearchSpace::SearchSpace(const TuneTask& tune_task) : tune_task_(tune_task) {
  const auto& target = tune_task_.target;
  // initialize a set of rules and they are commonly used by all states
  // TODO(zhhsplendid): pass correct output names to AutoInline
  sketch_rules_.emplace_back(new AutoInline(target, tune_task_.output_names));
  sketch_rules_.emplace_back(new MultiLevelTiling(target));
  sketch_rules_.emplace_back(new AutoUnroll(target));
  sketch_rules_.emplace_back(new SkipRule(target));
}

SearchState SearchSpace::GetScheduleMutate(const SearchState& state, const ExprCostModel& cost_model) {
  VLOG(5) << "Start SearchSpace::GetScheduleMutate in state:" << std::hash<std::string>()(state->DebugString());
  bool has_manual_schedule = false;
  if (has_manual_schedule) {
    SearchState ret = ManualScheduleMutate(state);
    return ret;
  }
  SearchState ret = RandomScheduleMutate(state);
  if (FLAGS_auto_schedule_use_cost_model) {
    ret->predicted_cost = cost_model.Predict(ret->ir_schedule.GetModule(), tune_task_.target);
  }
  return ret;
}

SearchState SearchSpace::ManualScheduleMutate(const SearchState& state) {
  // TODO(zhhsplendid): Add manual schedule mutate
  return state;
}

SearchState SearchSpace::RandomScheduleMutate(const SearchState& state) {
  VLOG(5) << "Start SearchSpace::RandomScheduleMutate";

  auto all_blocks        = state->ir_schedule.GetAllBlocks();
  auto block_sampler     = BlockSampler::Make(all_blocks, "probabilistic");
  std::string block_name = block_sampler->NextBlock(false);
  // TODO(BiynXu): Add rules after we have mutate rules, now only simulate with skip rule.
  std::vector<AutoGenRule*> mutate_rules = {sketch_rules_.back().get()};
  auto rule_sampler                      = RuleSampler::Make(mutate_rules, "probabilistic");
  auto new_states                        = CollectStateTransfer(state, block_name, rule_sampler.get(), 1, false, 1);
  return new_states.at(0);
}

std::vector<SearchState> SearchSpace::GetRandomPrunedInitialSketch() {
  VLOG(6) << "Start generating random pruned sketch...";
  ir::IRSchedule init_schedule(ir::ModuleExpr(tune_task_.GetLoweredFuncBodyExprs()));
  auto all_blocks    = init_schedule.GetAllBlocks();
  auto block_sampler = BlockSampler::Make(all_blocks, "probabilistic");

  std::vector<AutoGenRule*> init_rules;
  std::transform(sketch_rules_.begin(), sketch_rules_.end() - 1, std::back_inserter(init_rules), [](const auto& rule) {
    return rule.get();
  });
  CHECK(init_rules.size() > 0) << "number of init rules cannot be 0";

  SearchState init_state(init_schedule, SearchState::NOT_INIT_COST, {});
  std::vector<SearchState> states_buf1{init_state}, states_buf2;
  std::vector<SearchState>* p_states_cur  = &states_buf1;
  std::vector<SearchState>* p_states_next = &states_buf2;
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<> distribution(0, init_rules.size());
  int total_steps = 0, steps;
  std::string block_name;
  while ("" != (block_name = block_sampler->NextBlock()) && total_steps < init_sketch_random_depth_) {
    steps = distribution(rng);
    if (total_steps + steps > init_sketch_random_depth_) {
      steps = init_sketch_random_depth_ - total_steps;
    }
    total_steps += steps;
    p_states_next->clear();
    for (const auto& state : *p_states_cur) {
      auto rule_sampler = RuleSampler::Make(init_rules, "probabilistic");
      auto new_states   = CollectStateTransfer(state, block_name, rule_sampler.get(), steps, false, 1);
      p_states_next->insert(p_states_next->end(), new_states.begin(), new_states.end());
    }
    std::swap(p_states_cur, p_states_next);
  }
  VLOG(6) << "End generating random pruned sketch with new states num: " << p_states_next->size();
  return *p_states_next;
}

std::vector<SearchState> SearchSpace::GetRulePrunedInitialSketch() {
  VLOG(6) << "Start generating rule pruned sketch...";
  ir::IRSchedule init_schedule(ir::ModuleExpr(tune_task_.GetLoweredFuncBodyExprs()));
  auto all_blocks = init_schedule.GetAllBlocks();
  std::reverse(all_blocks.begin(), all_blocks.end());
  auto block_sampler = BlockSampler::Make(all_blocks, "traversal");

  std::vector<AutoGenRule*> init_rules;
  std::transform(sketch_rules_.begin(), sketch_rules_.end() - 1, std::back_inserter(init_rules), [](const auto& rule) {
    return rule.get();
  });
  CHECK(init_rules.size() > 0) << "number of init rules cannot be 0";

  SearchState init_state(init_schedule, SearchState::NOT_INIT_COST, {});
  std::vector<SearchState> states_buf1{init_state}, states_buf2;
  std::vector<SearchState>* p_states_cur  = &states_buf1;
  std::vector<SearchState>* p_states_next = &states_buf2;
  std::string block_name;
  while ("" != (block_name = block_sampler->NextBlock())) {
    p_states_next->clear();
    for (const auto& state : *p_states_cur) {
      auto rule_sampler = RuleSampler::Make(init_rules, "traversal");
      auto new_states   = CollectStateTransfer(state, block_name, rule_sampler.get(), 0, true);
      p_states_next->insert(p_states_next->end(), new_states.begin(), new_states.end());
    }
    std::swap(p_states_cur, p_states_next);
  }
  VLOG(6) << "End generating rule pruned sketch with new states num: " << p_states_next->size();
  return *p_states_next;
}

std::vector<SearchState> SearchSpace::GetInitialSketch(int num, const std::string& strategy) {
  VLOG(4) << "Start SearchSpace::GetInitialSketch with num:" << num;

  std::vector<SearchState> result;
  while (result.size() < num) {
    std::vector<SearchState> sketchs;
    if (strategy == "rule_prune") {
      sketchs = GetRulePrunedInitialSketch();
    } else if (strategy == "random_prune") {
      sketchs = GetRandomPrunedInitialSketch();
    } else {
      LOG(FATAL) << "Unimplemented init sketch strategy";
    }
    VLOG(6) << "generate sketch size: " << sketchs.size();
    if (VLOG_IS_ON(6)) {
      for (int i = 0; i < sketchs.size(); ++i) {
        VLOG(6) << "sketch-" << i << " :\n" << sketchs[i]->DebugString();
      }
    }
    // the more rules are applied, the greater the possibility of good results,
    // the more rules are applied, the more they are saved behind the queue,
    // so we give priority to the results in the rear
    for (auto iter = sketchs.rbegin(); iter != sketchs.rend(); ++iter) {
      result.push_back(*iter);
      if (result.size() == num) {
        break;
      }
    }
  }

  return result;
}

// Check whether the block named block_name exists in a SearchState
bool CheckBlockExist(const SearchState& state, std::string block_name) {
  auto block_exprs = state->ir_schedule.GetAllBlocks();
  for (auto block_expr : block_exprs) {
    const ir::ScheduleBlockRealize* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
    const ir::ScheduleBlock* block                = block_realize->schedule_block.As<ir::ScheduleBlock>();
    if (block->name == block_name) {
      return true;
    }
  }
  return false;
}

std::vector<SearchState> SearchSpace::CollectStateTransfer(const SearchState& state,
                                                           const std::string& block_name,
                                                           RuleSampler* rule_sampler,
                                                           int steps,
                                                           bool prune_by_rule,
                                                           double prune_probability) {
  std::list<SearchState> layer{state};
  int step = 0;
  AutoGenRule* rule;
  // After determining a SearchState and a block, each rule has two possibilities: apply and not apply.
  // In all transfer spaces, select a rule at each step, and collect all possible new states arrived by apply and not
  // apply. This forms a tree, and we can use rule pruning or random pruning to reduce the number of sketches.
  VLOG(6) << "Collect the states of all transfers within steps: " << steps;
  while ((step++ < steps || steps == 0) && (rule = rule_sampler->NextRule())) {
    VLOG(6) << "step = " << step << ", rule: " << rule->GetRuleName();
    std::list<SearchState> new_states;
    int id = 0;
    for (std::list<SearchState>::iterator iter = layer.begin(); iter != layer.end();) {
      // Some rules will reduce the number of blocks, such as AutoInline,
      // so we need to check whether the SearchState still has the block.
      if (!CheckBlockExist(*iter, block_name)) {
        ++iter;
        continue;
      }
      auto type = rule->AnalyseApplyType(*iter, block_name);
      VLOG(6) << "At SearchState " << ++id
              << ", apply type = " << static_cast<typename std::underlying_type<RuleApplyType>::type>(type);
      // if cannot apply the rule, skip it
      if (type == RuleApplyType::kCannotApply) {
        ++iter;
        continue;
      }
      // if can apply the rule, apply it and determine whether to prune the branche that do not apply
      std::vector<SearchState> tmp_states = rule->ApplyOnBlock(*iter, block_name);
      new_states.insert(new_states.end(), tmp_states.begin(), tmp_states.end());
      bool need_prune = false;
      if (prune_by_rule) {
        need_prune = (type == RuleApplyType::kApplyAndSkipAllRules);
      } else {
        std::mt19937 rng;
        rng.seed(std::random_device()());
        std::uniform_real_distribution<double> distribution(0, 1);
        need_prune = (distribution(rng) < prune_probability);
      }
      if (need_prune) {
        iter = layer.erase(iter);
      } else {
        ++iter;
      }
    }
    VLOG(6) << "apply on block: " << block_name << ", generate " << new_states.size() << " new states at step " << step;
    layer.splice(layer.end(), std::move(new_states));
  }
  VLOG(6) << "apply on block: " << block_name << ", generate " << layer.size() - 1 << " more states at all";
  return std::vector<SearchState>(layer.begin(), layer.end());
}

}  // namespace auto_schedule
}  // namespace cinn
