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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"

#include <cstdlib>

#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

static std::vector<int> auto_unroll_options = {0, 8, 32, 128};

bool HasReduceIter(const ir::ScheduleBlock* schedule_block) { return true; }

RuleApplyType AutoUnroll::Init(const ir::ModuleExpr& mod_expr) {
  ir_schedule_ = std::make_unique<ir::IRSchedule>(mod_expr);
  auto exprs   = mod_expr.GetExprs();
  std::vector<ir::ScheduleBlock*> root_schedule_blocks_;
  for (auto&& it_expr : exprs) {
    auto* block = it_expr.As<ir::Block>();
    CHECK(block) << "block is null";
    for (auto&& stmt : block->stmts) {
      ir::ScheduleBlockRealize* block_realize = stmt.As<ir::ScheduleBlockRealize>();
      CHECK(block_realize) << "stmt is not a ScheduleBlockRealize";
      ir::ScheduleBlock* schedule_block = block_realize->schedule_block.As<ir::ScheduleBlock>();
      CHECK(schedule_block) << "schedule_block field is not a ScheduleBlock";
      root_schedule_blocks_.emplace_back(schedule_block);
    }
  }

  applicable_schedule_blocks_.clear();
  for (size_t i = 0; i < root_schedule_blocks_.size(); ++i) {
    ir::ScheduleBlock* schedule_block = root_schedule_blocks_.at(i);
    if (HasReduceIter(schedule_block)) {
      applicable_schedule_blocks_.emplace_back(schedule_block);
    }
  }
  num_applicable_ = applicable_schedule_blocks_.size();

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndSkipThisRule : RuleApplyType::kCannotApply;
}

ir::ModuleExpr AutoUnroll::Apply(int index) {
  CHECK_LT(index, applicable_schedule_blocks_.size()) << "invalid apply index:" << index;
  auto* schedule_block = applicable_schedule_blocks_.at(index);
  int max_step         = auto_unroll_options[std::rand() % auto_unroll_options.size()];
  return ir_schedule_->GetModule();
}

}  // namespace auto_schedule
}  // namespace cinn
