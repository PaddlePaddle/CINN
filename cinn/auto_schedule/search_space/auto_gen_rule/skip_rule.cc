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

#include "cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"

#include <string>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/optim/ir_copy.h"

namespace cinn {
namespace auto_schedule {

SkipRule::SkipRule(const common::Target& target) : AutoGenRule(target) {}

RuleApplyType SkipRule::Init(const ir::IRSchedule& init_schedule) {
  ir_schedule_    = std::make_unique<ir::IRSchedule>(optim::IRCopy(init_schedule));
  num_applicable_ = 1;
  return RuleApplyType::kApply;
}

ir::IRSchedule SkipRule::Apply(int index) { return optim::IRCopy(*ir_schedule_); }

std::string SkipRule::GetRuleName() const { return "SikpRule"; }

AutoGenRule* SkipRule::NewPointer() const { return new SkipRule(*target_); }

}  // namespace auto_schedule
}  // namespace cinn
