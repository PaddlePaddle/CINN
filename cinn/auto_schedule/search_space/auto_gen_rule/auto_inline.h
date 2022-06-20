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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * The types of the AutoInline
 */
enum class AutoInlineType : int {
  // The block cannot be inlined
  kCannotInline = 0,
  // Inline this block into the consumer
  kInlineIntoConsumer,
  // Inline this block into the producer
  kInlineIntoProducer,
};

class AutoInline : public AutoGenRule {
 public:
  AutoInline(const common::Target& target);
  ~AutoInline() = default;

  RuleApplyType Init(const ir::ModuleExpr& mod_expr) override;

  ir::ModuleExpr Apply(int index) override;

  std::string GetRuleName() const override;

  AutoGenRule* NewPointer() const override;

  AutoInlineType AnalyzeInlineType(const ir::ScheduleBlockRealize& sche_block_realize) const;

  bool CanInlineIntoConsumer(const ir::ScheduleBlockRealize& sche_block_realize) const;

 private:
  std::unique_ptr<ir::IRSchedule> ir_schedule_;
  std::vector<ir::Expr> all_block_realizes_;
  std::vector<std::pair<int, AutoInlineType>> apply_indices_and_type_;
};

}  // namespace auto_schedule
}  // namespace cinn
