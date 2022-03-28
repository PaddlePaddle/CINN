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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class AutoInline : public AutoGenRule {
 public:
  AutoInline()  = default;
  ~AutoInline() = default;

  bool Init(const ir::ModuleExpr& mod_expr);

  ir::ModuleExpr Apply(int index);

  std::string GetRuleName() const;
};

}  // namespace auto_schedule
}  // namespace cinn
