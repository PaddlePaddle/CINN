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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class AddCacheRead : public AutoGenRule {
 public:
  AddCacheRead(const common::Target& target) : AutoGenRule(target) {
    // Select a cache memory type
    cache_memory_type_ = kMemoryTypes.at(target_->arch);
  }
  ~AddCacheRead() = default;

  // initailize the AddCacheRead rule, it must be called before further actions.
  RuleApplyType Init(ir::IRSchedule* init_schedule) override;

  // Applies rule on the ir::ModuleExpr for a schedule block specified by index
  // between 0 (inclusive) and NumberApplicable() (exclusive)
  void Apply(int index) override;

  // Returns the name of the rule, used for debug.
  std::string GetRuleName() const override { return "AddCacheRead"; }

  RuleApplyType AnalyseApplyType(SearchState state, const std::string& block_name) const override;

  std::vector<SearchState> ApplyOnBlock(SearchState state, const std::string& block_name) override;

 private:
  // Returns true if the schedule block expr is applicable by AddCacheRead
  bool MeetCondition(const ir::Expr& block_expr) const;

  // Get the out most reduce loop to set cache block in.
  ir::Expr GetOutermostReduceLoop(const ir::Expr& block_expr, ir::IRSchedule* ir_sch) const;

 private:
  std::vector<ir::Expr> applicable_schedule_blocks_;
  std::string cache_memory_type_;

  static const std::unordered_map<common::Target::Arch, std::string> kMemoryTypes;
};

}  // namespace auto_schedule
}  // namespace cinn
