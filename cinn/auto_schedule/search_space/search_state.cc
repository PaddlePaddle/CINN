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
#include <unordered_set>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

SearchState::SearchState(const ir::ModuleExpr& mod_expr) : ir_schedule(mod_expr) {}

SearchState::SearchState(ir::IRSchedule&& ir_sch) : ir_schedule(std::move(ir_sch)) {}

SearchState::SearchState(const SearchState& state) : ir_schedule(state.ir_schedule.GetModule()) {
  predicted_cost = state.predicted_cost;
  for (const std::shared_ptr<AutoGenRule>& rule : state.applicable_rules) {
    applicable_rules.emplace_back(std::shared_ptr<AutoGenRule>(rule->NewPointer()));
  }
}

SearchState& SearchState::operator=(const SearchState& src) {
  this->ir_schedule.SetExprs(src.ir_schedule.GetModule().GetExprs());
  this->predicted_cost = src.predicted_cost;
  this->applicable_rules.clear();
  for (const std::shared_ptr<AutoGenRule>& rule : src.applicable_rules) {
    this->applicable_rules.emplace_back(std::shared_ptr<AutoGenRule>(rule->NewPointer()));
  }
  return *this;
}

std::string SearchState::DebugString() const {
  const auto& exprs = ir_schedule.GetModule().GetExprs();
  std::stringstream module_stream;
  for (auto i = 0; i < exprs.size(); ++i) {
    module_stream << "Expr " << i << " {\n" << exprs.at(i) << "\n      }  // end Expr";
  }

  const char* fmt_str = R"ROC(
SearchState {
  IRSchedule {
    ModuleExpr {
      %s
    }   // end ModuleExpr
    ScheduleDesc {
      %s
    }
  }
  predicted_cost: %f
})ROC";

  return utils::StringFormat(
      fmt_str, module_stream.str().c_str(), ir_schedule.GetTraceDesc().DebugString().c_str(), predicted_cost);
}

bool operator<(const SearchState& left, const SearchState& right) { return left.predicted_cost < right.predicted_cost; }

void SearchState::InitAutoGenRules(const common::Target& target, const std::unordered_set<std::string>& output_names) {
  // TODO(zhhsplendid): pass correct output names to AutoInline
  applicable_rules = {std::shared_ptr<AutoGenRule>(new AutoInline(target, output_names)),
                      std::shared_ptr<AutoGenRule>(new MultiLevelTiling(target)),
                      std::shared_ptr<AutoGenRule>(new AutoUnroll(target)),
                      std::shared_ptr<AutoGenRule>(new SkipRule(target))};
}

}  // namespace auto_schedule
}  // namespace cinn
