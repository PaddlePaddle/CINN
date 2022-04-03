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

#include "cinn/auto_schedule/task/tune_context.h"

#include <glog/logging.h>

#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/lowered_func.h"

namespace cinn {
namespace auto_schedule {

std::vector<ir::Expr> TuneContext::GetLoweredFuncBodyExprs() const {
  std::vector<ir::Expr> result;
  for (const ir::LoweredFunc& func : lowered_funcs) {
    result.push_back(func->body);
  }
  return result;
}

void TuneContext::SetLoweredFuncBodyExprs(const std::vector<ir::Expr>& exprs) {
  size_t exprs_size = exprs.size();
  CHECK_EQ(exprs_size, lowered_funcs.size())
      << "SetLoweredFuncBodyExprs must have same number of Expr(s) and LoweredFunc(s)";
  for (size_t i = 0; i < exprs_size; ++i) {
    lowered_funcs[i]->body = exprs[i];
  }
}

}  // namespace auto_schedule
}  // namespace cinn
