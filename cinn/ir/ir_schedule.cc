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
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

// Self-defined operator to support std::set<Expr>
struct CompExpr {
  bool operator()(const Expr& left, const Expr& right) const {
    return utils::GetStreamCnt(left) < utils::GetStreamCnt(right);
  }
};

/**
 * \brief Given a For loop, return the next For loop in its body.
 * @param for_loop The given For loop.
 * @return The next For loop.
 */
Expr GetNextForLoop(const Expr& for_loop) {
  Expr result;
  CHECK(for_loop.As<ir::For>()) << "The input of GetNextForLoop should be ir::For!";
  Expr for_body = for_loop.As<ir::For>()->body;
  CHECK(for_body.As<ir::Block>()) << "The for_loop's body shoule be Block!";
  if (for_body.As<ir::Block>()->stmts.size() != 1U) return result;
  Expr block_body = for_body.As<ir::Block>()->stmts[0];
  if (block_body.As<IfThenElse>()) {
    CHECK(block_body.As<IfThenElse>()->true_case.As<ir::Block>());
    Expr true_case = block_body.As<IfThenElse>()->true_case;
    if (true_case.As<ir::Block>()->stmts.size() != 1U || !true_case.As<ir::Block>()->stmts[0].As<ir::For>())
      return result;
    result = true_case.As<ir::Block>()->stmts[0];
    return result;
  } else if (block_body.As<ir::For>()) {
    return block_body;
  } else {
    return result;
  }
}

/**
 * \brief Given two For loops, return all ir::IfThenElse nodes between them.
 * @param top The given top For loop.
 * @param bottom The given bottom For loop.
 * @return All ir::IfThenElse nodes between them.
 */
std::vector<Expr> GetIfThenElseInRange(const Expr& top, const Expr& bottom) {
  std::vector<Expr> if_nodes;
  CHECK(top.As<ir::For>());
  CHECK(bottom.As<ir::For>());
  for (auto loop_iter = top; loop_iter != bottom;) {
    CHECK(loop_iter.As<ir::For>());
    CHECK(loop_iter.As<ir::For>()->body.As<ir::Block>()) << "For node's body should be Block!";
    auto block = loop_iter.As<ir::For>()->body.As<ir::Block>();
    if (block->stmts.size() != 1) LOG(FATAL) << "Between For top and For bottom, there is a block's size not = 1!";
    Expr tmp = block->stmts[0];
    if (tmp.As<IfThenElse>()) {
      if_nodes.push_back(tmp);
      CHECK(tmp.As<IfThenElse>()->true_case.As<ir::Block>());
      Expr true_case = tmp.As<IfThenElse>()->true_case;
      CHECK(true_case.As<ir::Block>()->stmts.size() == 1U && true_case.As<ir::Block>()->stmts[0].As<ir::For>());
      tmp = true_case.As<ir::Block>()->stmts[0];
    }
    if (tmp.As<ir::For>())
      loop_iter = tmp;
    else
      LOG(FATAL) << "Between For top and For bottom, Block stmt:\n " << tmp << " is neither IfThenElse nor For!";
  }
  return if_nodes;
}

/**
 * Replace Vars in replaced to Exprs in candidates in source. Vars -> Exprs is one-to-one correspondence.
 * @param source The Expr we will implement the change.
 * @param replaced The Vars to be replaced.
 * @param candidates The Exprs to replace Vars in replaced.
 */
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

/**
 * Validate the factors param of Split. We will check if factors are validate and change -1 to positive integer.
 * @param factors The original factors.
 * @param total_extent The extent of the loop to be splitted.
 * @param return The valiated factors.
 */
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

std::vector<Expr> IRSchedule::Split(const Expr& loop, const std::vector<int>& factors) {
  CHECK(loop.As<ir::For>()) << "Expr param of Split must be For node! Please check.";
  auto* for_node = loop.As<ir::For>();
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
    new_loop_vars.push_back(temp_var);
  }
  substitute_value = common::AutoSimplify(substitute_value);
  Expr new_node    = optim::IRCopy(for_node->body);
  ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
  std::vector<Expr> splited_loops;
  splited_loops.resize(processed_factors.size());
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

std::vector<Expr> IRSchedule::Split(const std::string& block_name, int loop_index, const std::vector<int>& factors) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  Expr loop_expr;
  CHECK_LT(loop_index, (int)all_loops.size()) << "The loop index in Split should be less than total loop's number.";
  loop_expr = all_loops[loop_index];
  return this->Split(loop_expr, factors);
}

Expr IRSchedule::Fuse(const std::vector<Expr>& loops) {
  std::vector<const ir::For*> for_nodes;
  std::vector<Var> loop_vars;
  CHECK(!loops.empty()) << "The loops param of Fuse should not be empty! Please check.";

  for (const Expr& it_loop : loops) {
    CHECK(it_loop.As<ir::For>()) << "Expr param of Fuse must be For node! Please check.";
    if (!for_nodes.empty()) {
      CHECK(for_nodes.back()->body.As<ir::Block>()) << "The body of for node is not Block!";
      CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts.size(), 1U) << "The Block'size of for node is not 1!";
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

  Expr fused_body = optim::IRCopy(for_nodes.back()->body);
  ReplaceExpr(&fused_body, loop_vars, substitute_value);
  optim::Simplify(&fused_body);
  Expr fused_extent(1);
  for (int i = 0; i < loops_number; i++) {
    fused_extent = fused_extent * for_nodes[i]->extent;
  }
  fused_extent = common::AutoSimplify(fused_extent);

  if (!fused_body.As<ir::Block>()) fused_body = Block::Make({fused_body});
  Expr new_stmt =
      For::Make(fused_var, Expr(0), fused_extent, for_nodes[0]->for_type(), for_nodes[0]->device_api, fused_body);
  helper_.Replace(loops[0], new_stmt);
  return new_stmt;
}

Expr IRSchedule::Fuse(const std::string& block_name, const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); i++) {
    if (i > 0) CHECK_EQ(loops_index[i - 1] + 1, loops_index[i]) << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size()) << "The loop index in Reorder should be less than total loop's number.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

IRSchedule::IRSchedule(const ModuleExpr& module_expr, bool debug_flag) {
  ScheduleHelper sch_helper(module_expr, debug_flag);
  helper_ = sch_helper;
}

/**
 * Replace a For node to another For node.
 * @param src_sref The For node to be changed.
 * @param tgt_stmt The For node we want.
 */
void ScheduleHelper::Replace(const Expr& src_sref, const Expr& tgt_stmt) {
  CHECK(src_sref.As<ir::For>() && tgt_stmt.As<ir::For>());
  if (src_sref == tgt_stmt) {
    VLOG(3) << "two exprs are the same, no need to replace";
    return;
  }
  struct ForLoopMutator : public ir::IRMutator<> {
    ForLoopMutator(const Expr& source, const Expr& target) : source_(source), target_(target) {}

    void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::For* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }
    const Expr& source_;
    const Expr& target_;
  };
  auto exprs = module_expr_.GetExprs();
  ForLoopMutator mutator(src_sref, tgt_stmt);
  for (auto& i : exprs) {
    mutator(&i);
  }
}

/**
 * \brief Given a vector of For loops, return a set of them.
 * @param loops The given vector of For loops.
 * @return A set containing all the For loops in loops.
 */
const std::set<Expr, CompExpr> CollectLoopsToSet(const std::vector<Expr>& loops) {
  std::set<Expr, CompExpr> for_loops;
  for (auto& i : loops) {
    CHECK(i.As<ir::For>()) << "loops should be For node! Please check.";
    auto inserted = for_loops.insert(i);
    if (!inserted.second) {
      LOG(FATAL) << "There should be no duplicate elements in loops! Please check.";
    }
  }
  return for_loops;
}

/**
 * \brief Given a set of For loops, return the boundary among them.
 * @param loop_set The given set of For loops.
 * @return A pair of the boundary among For loops.(The top For and bottom For)
 */
std::pair<Expr, Expr> GetBoundaryOfReorderRange(const std::set<Expr, CompExpr>& loop_set) {
  Expr top = *loop_set.begin();
  Expr bottom;
  std::set<Expr, CompExpr> visited;
  bool first_traversal = true;
  for (Expr loop_i : loop_set) {
    if (visited.count(loop_i)) continue;
    for (auto v_for = loop_i;;) {
      if (visited.count(v_for)) {
        if (v_for != top) {
          LOG(FATAL) << "Loops in GetBoundaryOfReorderRange is not a chain! Please check.";
        }
        top = loop_i;
        break;
      }
      visited.insert(v_for);
      if (first_traversal && loop_set.count(v_for)) {
        bottom = v_for;
      }
      CHECK(v_for.As<ir::For>());
      auto tmp = GetNextForLoop(v_for);
      if (!tmp.defined()) break;
      v_for = tmp;
    }
    first_traversal = false;
  }
  CHECK(top.As<ir::For>());
  CHECK(bottom.defined());
  CHECK(bottom.As<ir::For>());
  return std::make_pair(top, bottom);
}

/**
 * \brief Given two For loops, return all loops between them.
 * @param top The top For loop.
 * @param bottom The bottom For loop.
 * @return A vector containing all For loops between the boundary, stored in ascending order.
 */
std::vector<Expr> GetLoopsInRange(const Expr& top, const Expr& bottom) {
  std::vector<Expr> chain;
  CHECK(top.As<ir::For>());
  CHECK(bottom.As<ir::For>());
  for (auto loop_iter = top; loop_iter != bottom;) {
    Expr tmp = GetNextForLoop(loop_iter);
    if (!tmp.defined()) LOG(FATAL) << "Loops in GetLoopsInReorderRange is not a chain! Please check.";
    chain.push_back(loop_iter);
    loop_iter = tmp;
  }
  chain.push_back(bottom);
  return chain;
}

/**
 * \brief Given params, construct a new loop.
 */
Expr ConstructNewLoopChain(const std::vector<Expr>& chain,
                           const std::vector<Expr>& ordered_loops,
                           const std::set<Expr, CompExpr>& loop_set,
                           std::vector<Expr>& if_nodes) {
  std::vector<std::set<std::string>> condition_vars;
  // In each IfThenElse node, find the vars its condition depends on.
  for (auto& if_expr : if_nodes) {
    CHECK(if_expr.As<IfThenElse>());
    auto var_set = ir::CollectIRNodes(if_expr.As<IfThenElse>()->condition, [&](const Expr* x) { return x->as_var(); });
    std::set<std::string> var_name_set;
    for (auto& i : var_set) var_name_set.insert(i.as_var()->name);
    condition_vars.push_back(var_name_set);
  }
  Expr new_loop;
  int index = static_cast<int>(ordered_loops.size()) - 1;
  // Construct the loop from bottom to top.
  for (int i = static_cast<int>(chain.size()) - 1; i >= 0; i--) {
    auto& loop_in_chain = chain[i];
    CHECK(loop_in_chain.As<ir::For>());
    Expr temp;
    if (loop_set.count(loop_in_chain)) {
      CHECK_GE(index, 0);
      temp = optim::IRCopy(ordered_loops[index]);
      --index;
    } else {
      temp = optim::IRCopy(loop_in_chain);
    }
    CHECK(temp.defined());
    CHECK(temp.As<ir::For>());
    if (new_loop.defined()) {
      temp.As<ir::For>()->body = Block::Make({new_loop});
    } else {
      temp.As<ir::For>()->body = loop_in_chain.As<ir::For>()->body;
    }
    Expr original_temp = temp;
    // Here we handle the IfThenElse nodes.
    for (int i = 0; i < static_cast<int>(if_nodes.size()); i++) {
      if (condition_vars[i].count(original_temp.As<ir::For>()->loop_var->name)) {
        Expr temp_body = temp.As<ir::For>()->body;
        if (temp_body.As<Block>() && temp_body.As<Block>()->stmts.size() == 1U)
          temp_body = temp_body.As<Block>()->stmts[0];
        temp.As<ir::For>()->body = IfThenElse::Make(
            if_nodes[i].As<IfThenElse>()->condition, temp_body, if_nodes[i].As<IfThenElse>()->false_case);
        temp.As<ir::For>()->body = Block::Make({temp.As<ir::For>()->body});
        if_nodes.erase(if_nodes.begin() + i);
        condition_vars.erase(condition_vars.begin() + i);
        i--;
      }
    }
    new_loop = temp;
  }
  CHECK(new_loop.defined());
  return new_loop;
}

void IRSchedule::Reorder(const std::vector<Expr>& loops) {
  if (loops.size() <= 1) return;
  std::set<Expr, CompExpr> loop_set = CollectLoopsToSet(loops);
  auto boundary                     = GetBoundaryOfReorderRange(loop_set);
  Expr top                          = boundary.first;
  Expr bottom                       = boundary.second;
  std::vector<Expr> chain           = GetLoopsInRange(top, bottom);
  std::vector<Expr> if_nodes        = GetIfThenElseInRange(top, bottom);

  Expr new_loop = ConstructNewLoopChain(chain, loops, loop_set, if_nodes);
  helper_.Replace(top, new_loop);
}

void IRSchedule::Reorder(const std::string& block_name, const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size()) << "The loop index in Reorder should be less than total loop's number.";
    loops_expr.emplace_back(all_loops[i]);
  }
  this->Reorder(loops_expr);
}

std::vector<Expr> ScheduleHelper::GetLoops(const Expr& block) const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
  for (auto& it_expr : exprs) {
    auto find_block = ir::CollectIRNodes(it_expr, [&](const Expr* x) {
      return x->As<ir::ScheduleBlockRealize>() &&
             x->As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>() &&
             x->As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name == block_name;
    });
    if (!find_block.empty()) {
      if (!result.empty()) LOG(FATAL) << "Find block with name: \n" << block_name << " appeared in more than one AST!";
      std::set<std::string> loops_name;
      for (auto& iter_val : block.As<ir::ScheduleBlockRealize>()->iter_values) {
        auto vars = ir::CollectIRNodes(iter_val, [&](const Expr* x) { return x->is_var(); });
        for (auto& iter_var : vars) loops_name.insert(iter_var.as_var_ref()->name);
      }
      auto loop_nodes = ir::CollectIRNodes(it_expr, [&](const Expr* x) {
        return x->As<ir::For>() && loops_name.count(x->As<ir::For>()->loop_var->name) != 0;
      });
      for (auto& it_for : loop_nodes) result.push_back(it_for);
    }
  }
  if (result.empty()) LOG(FATAL) << "Didn't find block with name: \n" << block_name << " in ModuleExepr.";
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });
  for (auto& it_for : result) VLOG(3) << "Get Loops :\n" << it_for;
  return result;
}

std::vector<Expr> ScheduleHelper::GetLoops(const std::string& block_name) const {
  Expr block               = this->GetBlock(block_name);
  std::vector<Expr> result = this->GetLoops(block);
  return result;
}

std::vector<Expr> ScheduleHelper::GetAllBlocks() const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  for (auto& it_expr : exprs) {
    auto block_nodes = ir::CollectIRNodes(it_expr, [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); });
    for (auto& it_block : block_nodes) result.push_back(it_block);
  }
  CHECK(!result.empty());
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    CHECK(i.As<ir::ScheduleBlockRealize>());
    CHECK(j.As<ir::ScheduleBlockRealize>());
    CHECK(i.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
    CHECK(j.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
    std::string& i_name = i.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
    std::string& j_name = j.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
    return (i_name < j_name);
  });
  return result;
}

Expr ScheduleHelper::GetBlock(const std::string& block_name) const {
  Expr result;
  std::vector<Expr> all_blocks = this->GetAllBlocks();
  for (auto& it_block : all_blocks) {
    if (it_block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name == block_name)
      result = it_block;
  }
  if (!result.defined()) LOG(FATAL) << "Didn't find a block with name " << block_name << " in this ModuleExpr!";
  return result;
}

}  // namespace ir
}  // namespace cinn
