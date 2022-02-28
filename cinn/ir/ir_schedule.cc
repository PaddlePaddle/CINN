// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/ir/ir_schedule.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

IRSchedule::IRSchedule(const ModuleExpr &module_expr, bool debug_flag) {
  ScheduleHelper sch_helper(module_expr, debug_flag);
  helper_ = sch_helper;
}

std::vector<Expr> ScheduleHelper::GetLoops() const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  for (auto &it_expr : exprs) {
    auto loop_nodes = ir::CollectIRNodes(it_expr, [&](const Expr *x) { return x->As<ir::For>(); });
    for (auto &it_for : loop_nodes) result.push_back(it_for);
  }
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });

  for (auto &it_for : result) VLOG(3) << "Get Loops : \n" << it_for;
  return result;
}

}  // namespace ir
}  // namespace cinn
