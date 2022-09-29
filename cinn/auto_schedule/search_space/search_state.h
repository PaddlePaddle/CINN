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

#include <functional>
#include <limits>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * Class to store immediate states during search
 */
class SearchState {
 public:
  // The IRSchedule, which contains ir::ModuleExpr and trace scheduling process
  ir::IRSchedule ir_schedule;

  // The rules that can be applied to this ModuleExpr at this state.
  // Initialized by list of all AutoGenRule
  std::vector<std::shared_ptr<AutoGenRule>> applicable_rules;

  // Cost model predicted cost
  float predicted_cost = NOT_INIT_COST;

  // Constant standing for a cost not being initialized
  static constexpr float NOT_INIT_COST = std::numeric_limits<float>::max();

  // return detail string of a SearchState for debug;
  std::string DebugString() const;

  SearchState() = default;

  SearchState(const ir::ModuleExpr& mod_expr);

  SearchState(ir::IRSchedule&& ir_sch);

  SearchState(const SearchState& state);

  SearchState(SearchState&& state) = default;

  SearchState& operator=(const SearchState& src);

  friend bool operator<(const SearchState& left, const SearchState& right);

  // Not all ModuleExpr has to be mutated AutoGenRule. For those states which
  // have ModuleExpr to random mutated by AutoGenRule, initialize it.
  void InitAutoGenRules(const common::Target& target, const std::unordered_set<std::string>& output_names);
};

}  // namespace auto_schedule
}  // namespace cinn
