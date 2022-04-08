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

#include <string>

#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {
/**
 * Enum class representing how this rule can be applied to a ModuleExpr.
 */
enum class RuleApplyType : int {
  // This rule cannot be applied to ModuleExpr
  kCannotApply = 0,
  // This rule can be applied to ModuleExpr
  kApply = 1,
  // This rule can be applied, but after applying, we should skip this rule
  // to apply on the module again.
  kApplyAndSkipThisRule = 2,
  // This rule can be applied, but after applying, we should skip all rules
  kApplyAndSkipAllRules = 3
};

/**
 * Base class for rules of auto-generating schedule (like Ansor's sketch generation)
 *
 */
class AutoGenRule {
 public:
  AutoGenRule()  = default;
  ~AutoGenRule() = default;

  // Initailize the AutoGenRule, it must be called before further actions.
  // Returns false if the rule cannot be applied on the mod_expr, true otherwise.
  virtual RuleApplyType Init(const ir::ModuleExpr& mod_expr) = 0;

  // CINN IRSchedule can contain many ScheduleBlock(s) and Loop(s), so
  // a auto gen rule may be suitable to different number of
  // Schedule Blocks. This method returns the number of ScheduleBlock
  // that can be applied by this auto gen rule
  virtual int NumberApplicable() const;

  // Applies rule on the ir::ModuleExpr for a schedule block randomly
  virtual ir::ModuleExpr ApplyRandomly();

  // Applies rule on the ir::ModuleExpr for a schedule block specified by index
  // between 0 (inclusive) and NumberApplicable() (exclusive)
  virtual ir::ModuleExpr Apply(int index) = 0;

  // Returns the name of the rule, used for debug.
  virtual std::string GetRuleName() const = 0;

 protected:
  // number of ScheduleBlock that can apply this auto gen rule
  int num_applicable_ = -1;
};

}  // namespace auto_schedule
}  // namespace cinn
