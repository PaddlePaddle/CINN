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
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

struct _SearchState_;
//! Shared Wrapper for _SearchState_
class SearchState : public common::Shared<_SearchState_> {
 public:
  SearchState() = default;
  // create a new SearchState
  SearchState(ir::IRSchedule ir_sch, float cost = NOT_INIT_COST, const std::vector<AutoGenRule*>& rules = {});

  // Constant standing for a cost not being initialized
  static constexpr float NOT_INIT_COST = std::numeric_limits<float>::max();
  // compare function for two states
  friend bool operator<(const SearchState& left, const SearchState& right);
};

//! Class to store immediate states during search
struct _SearchState_ : public common::Object {
  // IRSchedule contains ir::ModuleExpr and trace scheduling process
  ir::IRSchedule ir_schedule;
  // Cost model predicted cost
  float predicted_cost;
  // The rules that can be applied to the IRSchedule at this state.
  std::vector<AutoGenRule*> applicable_rules;

  // return detail string of content for debug;
  std::string DebugString() const;

  const char* type_info() const override { return __type_info__; }
  static constexpr char* __type_info__ = "auto_schedule_state";
};

}  // namespace auto_schedule
}  // namespace cinn
