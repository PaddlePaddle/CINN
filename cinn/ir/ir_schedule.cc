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

#include "cinn/common/cas.h"
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

void ReplaceExpr(Expr* source, const std::vector<Var>& replaced, const std::vector<Expr>& candidates) {
  CHECK_EQ(replaced.size(), candidates.size())
      << "In ReplaceExpr, the size of Vars to be replaced must be equal to the size of cadidate Exprs! Please check.";
  if (replaced.empty()) return;
  for (int i = 0; i < replaced.size(); i++) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i]) continue;
    optim::ReplaceVarWithExpr(source, replaced[i], candidates[i]);
  }
  return;
}

std::vector<int> ValidateFactors(const std::vector<int>& factors, int total_extent) {
  CHECK(!factors.empty()) << "The factors param of Split should not be empty! Please check.";
  bool has_minus_one = false;
  int product        = 1;
  for (auto& i : factors) {
    CHECK(i != 0) << "The params in factors of Split should not be 0! Please check.";
    CHECK(i >= -1) << "The params in factors of Split should not be less than -1! Please check.";
    if (i == -1) {
      CHECK(!has_minus_one) << "The params in factors of Split should not have more than one -1! Please check.";
      has_minus_one = true;
    } else {
      product *= i;
    }
  }
  std::vector<int> validated_factors = factors;
  if (!has_minus_one) {
    CHECK_EQ(product, total_extent)
        << "In Split, the factors' product should be equal to original loop's extent! Please check.";
    return validated_factors;
  } else {
    CHECK_LE(product, total_extent) << "In Split, when there is -1 in factors, the other factors' product should be <= "
                                       "original loop's extent! Please check.";
    int minus_one_candidate = (int)ceil((double)total_extent / (double)product);
    for (int i = 0; i < validated_factors.size(); i++) {
      if (validated_factors[i] == -1) {
        validated_factors[i] = minus_one_candidate;
      }
    }
    return validated_factors;
  }
}

std::vector<Expr> IRSchedule::Split(Expr& loop, const std::vector<int>& factors) {
  CHECK(loop.As<ir::For>()) << "Expr param of Split must be For node! Please check.";
  ir::For* for_node = loop.As<ir::For>();
  CHECK(common::is_zero(for_node->min)) << "The For node must start with 0! Please check.";
  CHECK(for_node->extent.is_constant()) << "The For node's extent must be constant! Please check.";
  int tot_extent         = for_node->extent.get_constant();
  auto processed_factors = ValidateFactors(factors, tot_extent);
  int prod_size = std::accumulate(processed_factors.begin(), processed_factors.end(), 1, std::multiplies<int>());
  std::vector<Var> new_loop_vars;
  Expr substitute_value(0);
  for (int i = 0; i < processed_factors.size(); i++) {
    Var temp_var(for_node->loop_var->name + "_" + std::to_string(i));
    substitute_value = Expr(temp_var) + substitute_value * Expr(processed_factors[i]);
    new_loop_vars.push_back(std::move(temp_var));
  }
  substitute_value = common::AutoSimplify(substitute_value);
  Expr new_node    = for_node->body;
  ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
  std::vector<Expr> splited_loops;
  splited_loops.reserve(processed_factors.size());

  if (tot_extent < prod_size) {
    new_node = IfThenElse::Make(LT::Make(substitute_value, for_node->extent), new_node);
  }

  for (int i = processed_factors.size() - 1; i >= 0; i--) {
    if (!new_node.As<ir::Block>()) new_node = Block::Make({new_node});
    new_node = For::Make(
        new_loop_vars[i], Expr(0), Expr(processed_factors[i]), for_node->for_type(), for_node->device_api, new_node);
    splited_loops[i] = new_node;
  }
  helper_.Replace(loop, new_node);
  return splited_loops;
}

Expr IRSchedule::Fuse(std::vector<Expr>& loops) {
  std::vector<ir::For*> for_nodes;
  std::vector<Var> loop_vars;
  CHECK(!loops.empty()) << "The loops param of Fuse should not be empty! Please check.";

  for (Expr& it_loop : loops) {
    CHECK(it_loop.As<ir::For>()) << "Expr param of Fuse must be For node! Please check.";
    if (!for_nodes.empty()) {
      CHECK(for_nodes.back()->body.As<ir::Block>()) << "The body of for node is not Block!";
      CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts[0], it_loop)
          << "The For nodes in loops param of Fuse must be adjacent! Please check.";
    }
    for_nodes.push_back(it_loop.As<ir::For>());
    loop_vars.push_back(it_loop.As<ir::For>()->loop_var);
  }
  std::string suffix;
  suffix           = for_nodes[0]->loop_var->name;
  int loops_number = for_nodes.size();
  for (int i = 1; i < loops_number; i++) {
    suffix += "_" + for_nodes[i]->loop_var->name;
  }
  suffix += "_fused";
  Var fused_var(suffix);
  std::vector<Expr> substitute_value;
  substitute_value.resize(loops_number);
  Expr fused_expr(fused_var);
  for (int i = loops_number - 1; i > 0; i--) {
    substitute_value[i] = Mod::Make(fused_expr, for_nodes[i]->extent);
    fused_expr          = Div::Make(fused_expr, for_nodes[i]->extent);
  }
  substitute_value[0] = fused_expr;

  Expr fused_body = for_nodes.back()->body;
  ReplaceExpr(&fused_body, loop_vars, substitute_value);
  Expr fused_extent(1);
  for (int i = 0; i < loops_number; i++) {
    fused_extent = fused_extent * for_nodes[i]->extent;
  }
  fused_extent = common::AutoSimplify(fused_extent);
  /*   fused_var->lower_bound = Expr(0);
    fused_var->upper_bound = fused_extent; */
  if (!fused_body.As<ir::Block>()) fused_body = Block::Make({fused_body});
  Expr new_stmt =
      For::Make(fused_var, Expr(0), fused_extent, for_nodes[0]->for_type(), for_nodes[0]->device_api, fused_body);
  helper_.Replace(loops[0], new_stmt);
  return loops[0];
}

IRSchedule::IRSchedule(const ModuleExpr& module_expr, bool debug_flag) {
  ScheduleHelper sch_helper(module_expr, debug_flag);
  helper_ = sch_helper;
}

void ScheduleHelper::Replace(Expr& src_sref, const Expr& tgt_stmt) {
  CHECK(src_sref.As<ir::For>() && tgt_stmt.As<ir::For>());
  if (src_sref == tgt_stmt) {
    LOG(INFO) << "two exprs are the same, no need to replace";
    return;
  }
  src_sref.As<ir::For>()->loop_var   = tgt_stmt.As<ir::For>()->loop_var;
  src_sref.As<ir::For>()->min        = tgt_stmt.As<ir::For>()->min;
  src_sref.As<ir::For>()->extent     = tgt_stmt.As<ir::For>()->extent;
  src_sref.As<ir::For>()->body       = tgt_stmt.As<ir::For>()->body;
  src_sref.As<ir::For>()->device_api = tgt_stmt.As<ir::For>()->device_api;
}

std::vector<Expr> ScheduleHelper::GetLoops() const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  for (auto& it_expr : exprs) {
    auto loop_nodes = ir::CollectIRNodes(it_expr, [&](const Expr* x) { return x->As<ir::For>(); });
    for (auto& it_for : loop_nodes) result.push_back(it_for);
  }
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });

  for (auto& it_for : result) VLOG(3) << "Get Loops : \n" << it_for;
  return result;
}

}  // namespace ir
}  // namespace cinn
