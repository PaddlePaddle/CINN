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

#include "cinn/auto_schedule/search_space/search_state.h"

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

SearchState::SearchState(ir::IRSchedule ir_sch, float cost, const std::vector<AutoGenRule*>& rules)
    : common::Shared<_SearchState_>(common::make_shared<_SearchState_>()) {
  auto* state             = get();
  state->ir_schedule      = std::move(ir_sch);
  state->applicable_rules = rules;
  state->predicted_cost   = cost;
}

std::string _SearchState_::DebugString() const {
  const auto& exprs = ir_schedule.GetModule().GetExprs();
  std::stringstream module_stream;
  for (auto i = 0; i < exprs.size(); ++i) {
    module_stream << "Expr " << i << " {\n" << exprs.at(i) << "\n}  // end Expr";
  }

  const char* fmt_str = R"ROC(
ModuleExpr {
%s
} // end ModuleExpr
ScheduleDesc {
%s
} // end ScheduleDesc
predicted_cost: %f)ROC";

  return utils::StringFormat(
      fmt_str, module_stream.str().c_str(), ir_schedule.GetTraceDesc().DebugString().c_str(), predicted_cost);
}

bool operator<(const SearchState& left, const SearchState& right) {
  return left->predicted_cost < right->predicted_cost;
}

}  // namespace auto_schedule
}  // namespace cinn
