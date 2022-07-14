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
#include "cinn/lang/compute.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace ir {

Tensor GetTensor(const Expr& block) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::CollectIRNodesWithoutTensor(block, [&](const Expr* x) { return x->As<ir::Store>(); });
  CHECK(!find_tensor.empty()) << "Didn't find Store in block!";
  CHECK_EQ(find_tensor.size(), 1U) << "One block should only have one Store node!(except for root block)";
  CHECK((*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor());
  Tensor tensor = (*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor_ref();
  return tensor;
}

Tensor GetReadTensor(const Expr& block, int index) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::CollectIRNodesWithoutTensor(block, [&](const Expr* x) { return x->As<ir::Store>(); });
  CHECK(!find_tensor.empty()) << "Didn't find Store in block!";
  CHECK_EQ(find_tensor.size(), 1U) << "One block should only have one Store node!(except for root block)";
  std::vector<Tensor> res;
  auto find_read_tensor = ir::CollectIRNodesWithoutTensor(block, [&](const Expr* x) {
    if (x->As<ir::Load>()) res.push_back(x->As<ir::Load>()->tensor.as_tensor_ref());
    return x->As<ir::Load>();
  });
  CHECK_EQ(find_read_tensor.size(), res.size());
  CHECK(!find_read_tensor.empty()) << "Didn't find Load tensor in block!";
  CHECK_LT(index, (int)find_read_tensor.size()) << "Index is not < read tensor's size!";
  return res[index];
}

int GetLoopExtent(const Expr& loop) {
  CHECK(loop.As<ir::For>());
  CHECK(common::is_zero(loop.As<ir::For>()->min));
  CHECK(loop.As<ir::For>()->extent.is_constant());
  return (int)loop.As<ir::For>()->extent.get_constant();
}

// Self-defined operator to support std::set<Expr>
struct CompExpr {
  bool operator()(const Expr& left, const Expr& right) const {
    return utils::GetStreamCnt(left) < utils::GetStreamCnt(right);
  }
};

// Self-defined operator to support std::set<Var>
struct CompVar {
  bool operator()(const Var& left, const Var& right) const { return left->name < right->name; }
};

/*!
 * \brief Check if a Expr node contains a ScheduleBlockRealize node.
 * \param container The container Expr node.
 * \param expr The node we want to find.
 * \return If the container contains the expr.
 */
bool Contains(const Expr& container, const Expr& expr) {
  auto find_expr = ir::CollectIRNodesWithoutTensor(container, [&](const Expr* x) { return *x == expr; });
  return (!find_expr.empty());
}

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

struct MappingVarToExprMutator : public ir::IRMutator<> {
  MappingVarToExprMutator(const std::map<Var, Expr, CompVar>& replacing_map) : replacing_map_(replacing_map) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (replacing_map_.count(op->as_var_ref())) {
      *op = replacing_map_.at(op->as_var_ref());
    }
  }

 private:
  const std::map<Var, Expr, CompVar>& replacing_map_;
};

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
  std::map<Var, Expr, CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i]) continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  MappingVarToExprMutator mapper(replacing_map);
  mapper(source);
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
    for (int i = 0; i < validated_factors.size(); ++i) {
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
  for (int i = 0; i < processed_factors.size(); ++i) {
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
  CHECK_GE(loop_index, 0) << "The loop index in Split should be >= 0.";
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
  for (int i = 1; i < loops_number; ++i) {
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
  for (int i = 0; i < loops_number; ++i) {
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
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0) CHECK_EQ(loops_index[i - 1] + 1, loops_index[i]) << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size()) << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

Expr IRSchedule::Fuse(const Expr& block, const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0) CHECK_EQ(loops_index[i - 1] + 1, loops_index[i]) << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size()) << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

void IRSchedule::MutateForType(const Expr& loop, ForType for_type, int factor) {
  auto* for_node = loop.As<ir::For>();
  CHECK(for_node) << "loop param must be For node! Please check.";
  CHECK(for_node->is_serial()) << "loop is not serial, current forloop type is "
                               << static_cast<int>(for_node->for_type());
  auto loop_copy     = optim::IRCopy(loop);
  auto* new_for_node = loop_copy.As<ir::For>();
  CHECK(new_for_node);
  new_for_node->set_for_type(for_type);
  if (new_for_node->is_vectorized()) {
    VectorizeInfo vec_info(0, factor);
    new_for_node->set_vectorize_info(vec_info);
  } else if (new_for_node->is_binded()) {
    BindInfo bind_info(for_type, factor, DeviceAPI::GPU);
    new_for_node->set_bind_info(bind_info);
  }
  helper_.Replace(loop, loop_copy);
}

void IRSchedule::Parallel(const Expr& loop) { MutateForType(loop, ForType::Parallel); }

void IRSchedule::Vectorize(const Expr& loop, int factor) {
  CHECK_GT(factor, 0) << "vectorize factor should be more than 0";
  MutateForType(loop, ForType::Vectorized, factor);
}

void IRSchedule::Unroll(const Expr& loop) { MutateForType(loop, ForType::Unrolled); }

void IRSchedule::Bind(const Expr& loop, const std::string& thread_axis) {
  static std::set<std::string> thread_axes = {
      "blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"};
  CHECK(thread_axes.count(thread_axis)) << "thread_axis " << thread_axis << " is not supported";
  int offset = thread_axis.back() - 'x';
  if (thread_axis[0] == 'b') {
    MutateForType(loop, ForType::GPUBlock, offset);
  } else {
    MutateForType(loop, ForType::GPUThread, offset);
  }
}

// check whether or not have the var with the same name
bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name) {
  for (auto& expr : exprs) {
    auto find_expr = ir::CollectIRNodesWithoutTensor(
        expr, [&](const Expr* x) { return x->As<_Var_>() && x->As<_Var_>()->name == var_name; });
    if (!find_expr.empty()) return true;
  }
  return false;
}

// The struct used to mutate new rfactor forloop and its' schedule block.
struct RfMutator : public ir::IRMutator<> {
 public:
  RfMutator(const Expr& rf_loop, const int& rf_axis) : rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) {
    auto* rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    old_rf_loop_var_ = rf_for->loop_var;
    new_rf_loop_var_ = Var("rf_" + old_rf_loop_var_->name);
    IRMutator::Visit(expr, expr);
  }

  Tensor GetNewRfTensor() { return new_rf_tensor_; }

  void Visit(const ScheduleBlockRealize* op, Expr* expr) override {
    // modify iter_vars and iter_values
    auto* node = expr->As<ScheduleBlockRealize>();
    CHECK(node);
    auto* schedule_block = node->schedule_block.As<ScheduleBlock>();
    CHECK(schedule_block);
    old_output_name_  = schedule_block->name;
    find_tensor_      = false;
    auto& block_vars  = schedule_block->iter_vars;
    auto& iter_values = node->iter_values;
    CHECK(old_rf_loop_var_.defined());
    CHECK(new_rf_loop_var_.defined());
    CHECK_EQ(iter_values.size(), block_vars.size());
    int rf_index = -1;
    for (int i = 0; i < iter_values.size(); ++i) {
      // substitute the old rfactor loop var to new rfactor loop var
      if (ContainVar({iter_values[i]}, old_rf_loop_var_->name)) {
        CHECK_EQ(rf_index, -1) << "only one block var can bind the rfactor loop var";
        CHECK(iter_values[i].As<_Var_>()) << "rfactor loop var not support composite bindings";
        rf_index = i;
        optim::ReplaceVarWithExpr(&iter_values[i], old_rf_loop_var_, new_rf_loop_var_);
        new_rf_itervar_ = block_vars[i];
      }
    }
    // create new rfactor block var if not exist
    if (rf_index == -1) {
      new_rf_itervar_ = Var("i" + std::to_string(block_vars.size()));
      iter_values.push_back(new_rf_loop_var_);
      block_vars.push_back(new_rf_itervar_);
    }
    IRMutator::Visit(&node->schedule_block, &node->schedule_block);
    CHECK(find_tensor_) << "not find the store tensor with the schedule block name " << old_output_name_;
    schedule_block->name = "rf_" + old_output_name_;
  }

  void Visit(const Load* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Load>();
    CHECK(node);
    auto* tensor = node->tensor.As<_Tensor_>();
    CHECK(tensor);
    if (tensor->name == "rf_" + old_output_name_) {
      int size = node->indices.size();
      CHECK_LE(rf_axis_, size) << "rf_axis should not be greater than indice size " << size;
      CHECK(new_rf_itervar_.defined());
      CHECK(!ContainVar(node->indices, new_rf_itervar_->name))
          << "original output tensor " << old_output_name_ << " should not have the new rfactor index "
          << new_rf_itervar_;
      node->indices.insert(node->indices.begin() + rf_axis_, new_rf_itervar_);
    }
  }

  void Visit(const Store* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Store>();
    CHECK(node);
    auto* tensor = node->tensor.As<_Tensor_>();
    CHECK(tensor);
    if (tensor->name == old_output_name_) {
      find_tensor_ = true;
      tensor->name = "rf_" + tensor->name;
      int size     = node->indices.size();
      CHECK_LE(rf_axis_, size) << "rf_axis should not be greater than indice size " << size;
      CHECK(!ContainVar(node->indices, new_rf_itervar_->name))
          << "original output tensor " << old_output_name_ << " should not have the new rfactor index "
          << new_rf_itervar_;
      node->indices.insert(node->indices.begin() + rf_axis_, new_rf_itervar_);
      auto* rf_for = rf_loop_.As<For>();
      CHECK(rf_for);
      CHECK(is_zero(rf_for->min)) << "rfactor loop's min should be zero";
      auto extent  = common::AutoSimplify(rf_for->extent);
      auto& shape  = tensor->shape;
      auto& domain = tensor->domain;
      CHECK_LE(rf_axis_, shape.size()) << "rf_axis should not be greater than tensor shape size " << shape.size();
      CHECK_LE(rf_axis_, domain.size()) << "rf_axis should not be greater than tensor domain size " << domain.size();
      shape.insert(shape.begin() + rf_axis_, extent);
      domain.insert(domain.begin() + rf_axis_, extent);
      if (tensor->buffer.defined()) {
        if (tensor->buffer->name.find_first_of("rf") == std::string::npos) {
          tensor->buffer->name  = "rf_" + tensor->buffer->name;
          tensor->buffer->shape = shape;
        }
      }
      new_rf_tensor_ = Tensor(tensor);
    }
    IRMutator::Visit(&node->value, &node->value);
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    CHECK(node);
    depth++;
    auto* rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    // erase the original rfactor forloop
    if (node->loop_var->name == old_rf_loop_var_->name) {
      auto body = node->body.As<Block>();
      if (body && body->stmts.size() == 1) {
        *expr = body->stmts[0];
      } else {
        *expr = node->body;
      }
      IRMutator::Visit(expr, expr);
    } else {
      IRMutator::Visit(&node->body, &node->body);
    }
    if (rf_axis_ == 0 && depth == rf_axis_) {
      // insert new rfactor forloop in the rf_axis as serial loop
      *expr = For::Make(
          new_rf_loop_var_, rf_for->min, rf_for->extent, ForType::Serial, rf_for->device_api, Block::Make({*expr}));
    } else if (depth == rf_axis_ - 1) {
      // insert new rfactor forloop in the rf_axis as serial loop
      node->body = Block::Make(
          {For::Make(new_rf_loop_var_, rf_for->min, rf_for->extent, ForType::Serial, rf_for->device_api, node->body)});
    }
    depth--;
  }

 private:
  Expr rf_loop_;
  Var old_rf_loop_var_;
  Var new_rf_loop_var_;
  int rf_axis_;
  int depth         = -1;
  bool find_tensor_ = false;
  std::string old_output_name_;
  Var new_rf_itervar_;
  Tensor new_rf_tensor_;
};

// The struct used to mutate final write-back forloop and schedule block.
struct FinalMutator : public ir::IRMutator<> {
 public:
  FinalMutator(const Expr& rf_loop, const int& rf_axis, const Tensor& new_rf_tensor)
      : rf_loop_(rf_loop), rf_axis_(rf_axis), new_rf_tensor_(new_rf_tensor) {}
  void operator()(Expr* expr) {
    auto* rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    old_rf_loop_var_ = rf_for->loop_var;
    IRMutator::Visit(expr, expr);
  }

  void Visit(const ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ScheduleBlockRealize>();
    CHECK(node);
    auto* schedule_block = node->schedule_block.As<ScheduleBlock>();
    CHECK(schedule_block);
    auto& iter_vars   = schedule_block->iter_vars;
    auto& iter_values = node->iter_values;
    output_name_      = schedule_block->name;
    visit_init_block_ = output_name_.rfind("_init") != std::string::npos;
    if (!visit_init_block_) {
      for (int i = 0; i < iter_values.size(); ++i) {
        if (ContainVar({iter_values[i]}, old_rf_loop_var_->name)) {
          // record the rfactor loop var's block var
          CHECK(iter_values[i].As<_Var_>()) << "not support complex reduce bindings: " << iter_values[i];
          old_rf_iter_var_ = iter_vars[i];
          break;
        }
      }
    }
    IRMutator::Visit(&node->schedule_block, &node->schedule_block);
    // modify iter_vars and iter_values, erase other reduce block vars and values
    for (auto it = iter_values.begin(); it != iter_values.end(); ++it) {
      for (auto erase_var : erase_reduce_loopvars_) {
        if (ContainVar({*it}, erase_var)) {
          CHECK((*it).As<_Var_>()) << "not support complex reduce bindings: " << *it;
          iter_vars.erase(it - iter_values.begin() + iter_vars.begin());
          iter_values.erase(it);
          --it;
          break;
        }
      }
    }
  }

  // currently only support reduce_sum, reduce_mul, reduce_min and reduce_max
  void Visit(const Add* op, Expr* expr) override {
    auto* node = expr->As<Add>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b       = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Mul* op, Expr* expr) override {
    auto* node = expr->As<Mul>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b       = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Min* op, Expr* expr) override {
    auto* node = expr->As<Min>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b       = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Max* op, Expr* expr) override {
    auto* node = expr->As<Max>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b       = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Store* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Store>();
    CHECK(node);
    auto* tensor = node->tensor.As<_Tensor_>();
    CHECK(tensor);
    CHECK_EQ(tensor->name, output_name_) << "store name should be same with the schedule block name";
    if (!visit_init_block_) {
      new_rf_indice_ = node->indices;
      CHECK_LE(rf_axis_, new_rf_indice_.size())
          << "rf_axis_ should not be greater than tensor indice size " << new_rf_indice_.size();
      CHECK(old_rf_iter_var_.defined());
      new_rf_indice_.insert(new_rf_indice_.begin() + rf_axis_, old_rf_iter_var_);
      IRMutator::Visit(&node->value, &node->value);
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    CHECK(node);
    auto* rf_for = rf_loop_.As<For>();
    // erase the reduce forloops after the init block except the rfactor loop
    if (visit_init_block_ && node->loop_var->name != old_rf_loop_var_->name) {
      erase_reduce_loopvars_.insert(node->loop_var->name);
      auto body = node->body.As<Block>();
      if (body && body->stmts.size() == 1) {
        *expr = body->stmts[0];
      } else {
        *expr = node->body;
      }
      IRMutator::Visit(expr, expr);
    } else {
      IRMutator::Visit(&node->body, &node->body);
    }
  }

 private:
  Expr rf_loop_;
  int rf_axis_;
  Var old_rf_loop_var_;
  Var old_rf_iter_var_;
  std::string output_name_;
  // collect reduce loop vars except rfactor loop var
  std::set<std::string> erase_reduce_loopvars_;
  bool visit_init_block_ = false;
  Tensor new_rf_tensor_;
  std::vector<Expr> new_rf_indice_;
};

// The struct used to create all stmts after rfactor transformation.
struct RfCreater : public ir::IRMutator<> {
 public:
  RfCreater(const Expr& root, const Expr& rf_loop, const int& rf_axis)
      : root_(root), rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  Expr CreateRfAllStmts() {
    auto root_realize = root_.As<ScheduleBlockRealize>();
    CHECK(root_realize);
    auto root_block = root_realize->schedule_block.As<ScheduleBlock>();
    CHECK(root_block);
    Expr root_loop = optim::IRCopy(root_block->body);
    if (auto block = root_loop.As<Block>()) {
      CHECK_EQ(block->stmts.size(), 1U) << "rfactor root should only have one block stmt";
      root_loop = block->stmts[0];
    }
    auto* root_for = root_loop.As<For>();
    CHECK(root_for);
    auto rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    // create new rfactor forloops
    Expr new_rf_forloop = optim::IRCopy(root_loop);
    RfMutator rf_mutator(rf_loop_, rf_axis_);
    rf_mutator(&new_rf_forloop);
    VLOG(3) << "After RfMutator, new rf_forloop is\n" << new_rf_forloop;
    auto new_rf_tensor = rf_mutator.GetNewRfTensor();
    // create final write-back forloops
    Expr final_forloop = optim::IRCopy(root_loop);
    FinalMutator final_mutator(rf_loop_, rf_axis_, new_rf_tensor);
    final_mutator(&final_forloop);
    VLOG(3) << "After FinalMuator, final write-back forloop is\n" << final_forloop;
    // combine the new created rfactor forloops with the final write-back forloops and replace
    root_block->body = Block::Make({new_rf_forloop, final_forloop});
    return new_rf_tensor;
  }

  Expr root_;
  Expr rf_loop_;
  int rf_axis_;
};

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis) {
  auto* rf_for = rf_loop.As<ir::For>();
  CHECK(rf_for) << "Expr param of Rfactor must be For node! Please check.";
  // check the rf_loop only has one schedule block
  auto block_nodes = ir::CollectIRNodes(rf_loop, [&](const Expr* x) { return x->As<ScheduleBlockRealize>(); });
  CHECK_EQ(block_nodes.size(), 1U) << "Rfactor Loop should only have one schedule block";
  auto find_store = ir::CollectIRNodes(rf_loop, [&](const Expr* x) { return x->As<Store>(); });
  CHECK_EQ(find_store.size(), 1U);
  auto indice = find_store.begin()->As<Store>()->indices;
  // check rf_axis
  CHECK_LE(rf_axis, indice.size()) << "rf_axis should not be greater than store's domain size";
  // check rfactor loop is reduce
  auto* sch_block_realize = block_nodes.begin()->As<ScheduleBlockRealize>();
  auto* sch_block         = sch_block_realize->schedule_block.As<ScheduleBlock>();
  CHECK(sch_block);
  auto& iter_values = sch_block_realize->iter_values;
  auto& iter_vars   = sch_block->iter_vars;
  CHECK_EQ(iter_values.size(), iter_vars.size());
  auto rf_loop_var = rf_for->loop_var;
  Var rf_block_var;
  for (int i = 0; i < iter_values.size(); ++i) {
    if (ContainVar({iter_values[i]}, rf_loop_var->name)) {
      CHECK(!rf_block_var.defined()) << "rfactor loop var can only be binded to one block var";
      auto iter_value = iter_values[i].As<_Var_>();
      CHECK(iter_value) << "not support complex reduce bindings";
      rf_block_var = iter_vars[i];
      auto it      = std::find_if(indice.begin(), indice.end(), [&](const Expr& x) {
        return x.As<_Var_>() && x.As<_Var_>()->name == rf_block_var->name;
      });
      CHECK(it == indice.end()) << "rfactor loop var is not reduce, please check!";
    }
  }
}

Expr IRSchedule::Rfactor(const Expr& rf_loop, int rf_axis) {
  CHECKRfactorValidation(rf_loop, rf_axis);
  // get root ScheduleBlockRealize
  Expr root = GetRootBlock(rf_loop);
  // create all stmts after rfactor transformation
  RfCreater rf_create(root, rf_loop, rf_axis);
  // return new created rfactor tensor
  return rf_create.CreateRfAllStmts();
}

/**
 * Return loops that contain the expr.
 * @param expr The expr.
 * @param root The root of the whole AST.
 * @param return Loops in AST that contain the expr.
 */
std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root) {
  auto loop_nodes = ir::CollectIRNodes(root, [&](const Expr* x) { return x->As<ir::For>() && Contains(*x, expr); });
  std::vector<Expr> result(loop_nodes.begin(), loop_nodes.end());
  if (result.empty()) LOG(FATAL) << "Didn't find expr's loops in root.";
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });
  return result;
}

/**
 * Given an Expr and all vars' range, return the Expr's range(min and max).
 * @param index The Expr we want to calculate its range.
 * @param iter_vars The vars in expr.
 * @param iter_range Each var's range.
 * @param i The index indicating we are replacing i-th var to its range.
 * @param return The <min, max> of index after replacing i-th var to its range. If the range is not constant, return
 * <-1, -1>.
 */
std::pair<Expr, Expr> GetRange(Expr index,
                               const std::vector<Var>& iter_vars,
                               const std::vector<std::pair<Expr, Expr>>& iter_range,
                               int i) {
  if (index.is_constant()) {
    if (index.get_constant() == 0.f) return std::make_pair(index, common::AutoSimplify(index + Expr(1)));
    return std::make_pair(index, index);
  } else if (i >= (int)iter_vars.size()) {
    return std::make_pair(Expr(-1), Expr(-1));
  } else {
    Expr index2 = index;
    ReplaceExpr(&index, {iter_vars[i]}, {iter_range[i].first});
    ReplaceExpr(&index2, {iter_vars[i]}, {iter_range[i].second});
    index       = common::AutoSimplify(index);
    index2      = common::AutoSimplify(index2);
    auto range1 = GetRange(index, iter_vars, iter_range, i + 1);
    auto range2 = GetRange(index2, iter_vars, iter_range, i + 1);
    CHECK(range1.first.is_constant());
    CHECK(range1.second.is_constant());
    CHECK(range2.first.is_constant());
    CHECK(range2.second.is_constant());
    return std::make_pair(range1.first.get_constant() > range2.first.get_constant() ? range2.first : range1.first,
                          range1.second.get_constant() > range2.second.get_constant() ? range1.second : range2.second);
  }
}

/**
 * Given a vector of Expr and all vars' range, return the vector of Expr's ranges(min and max).
 * @param tensor_indices The vector of Expr. We want to calculate each Expr's range.
 * @param iter_vars The vars in expr.
 * @param iter_range Each var's range.
 * @param tensor The tensor. tensor_indices is its index.
 * @param return The <min, max> of tensor_indices. If it is not constant, return corresponding tensor's shape.
 */
std::vector<std::pair<Expr, Expr>> GetRange(const std::vector<Expr>& tensor_indices,
                                            const std::vector<Var>& iter_vars,
                                            const std::vector<std::pair<Expr, Expr>>& iter_range,
                                            const Tensor& tensor) {
  CHECK_EQ(iter_vars.size(), iter_range.size());
  std::vector<std::pair<Expr, Expr>> result;
  for (int i = 0; i < (int)tensor_indices.size(); ++i) {
    auto range = GetRange(tensor_indices[i], iter_vars, iter_range, 0);
    CHECK(range.first.is_constant());
    if ((int)range.first.get_constant() == (-1)) {
      if (tensor->buffer.defined()) {
        CHECK_GT((int)tensor->buffer->shape.size(), i);
        result.push_back(std::make_pair(Expr(0), tensor->buffer->shape[i]));
      } else {
        CHECK_GT((int)tensor->shape.size(), i);
        result.push_back(std::make_pair(Expr(0), tensor->shape[i]));
      }
    } else {
      CHECK(range.second.is_constant());
      result.push_back(range);
    }
  }
  return result;
}

/**
 * Given a ScheduleBlockRealize, an AST root, a tensor and its tensor_indices, return the accessed buffer region of the
 * tensor in block.
 * @param block The ScheduleBlockRealize.
 * @param tensor_indices The tensor's indices.
 * @param tensor The tensor.
 * @param root The root of whole AST.
 * @param return The accessed buffer region of the tensor in block.
 */
std::vector<std::pair<Expr, Expr>> CalculateTensorRegions(const Expr& block,
                                                          const std::vector<Expr>& tensor_indices,
                                                          const Tensor& tensor,
                                                          const Expr& root) {
  CHECK(block.As<ScheduleBlockRealize>());
  auto iter_var   = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
  auto iter_value = block.As<ir::ScheduleBlockRealize>()->iter_values;

  std::vector<Var> iter_vars;
  std::vector<std::pair<Expr, Expr>> iter_range;

  auto outer_loops = GetLoopsOfExpr(block, root);
  for (auto& loop : outer_loops) {
    CHECK(loop.As<For>());
    iter_vars.push_back(loop.As<For>()->loop_var);
    iter_range.push_back(
        std::make_pair(loop.As<For>()->min, common::AutoSimplify(loop.As<For>()->min + loop.As<For>()->extent)));
  }

  std::vector<Expr> replaced_indices;

  for (auto& index : tensor_indices) {
    auto temp = optim::IRCopy(index);
    ReplaceExpr(&temp, iter_var, iter_value);
    replaced_indices.push_back(temp);
  }
  auto result = GetRange(replaced_indices, iter_vars, iter_range, tensor);

  return result;
}
struct CacheBlockInfo {
  /*! \brief The tensor to be read. */
  Tensor read_tensor;
  /*! \brief The tensor to be written. */
  Tensor write_tensor;
  /*! \brief The tensor allocation to be inserted into the block signature. */
  Tensor alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  Expr loc_block;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  int loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Expr cache_block;
};

/**
 * Return n-th access tensor in block
 * @param block The ScheduleBlockRealize.
 * @param index The index indicating which tensor we want to get.
 * @param is_write We want to get write tensor or read tensor.
 * @param return The n-th access tensor in block. Should be ir::Store(is_write) or ir::Load(!is_write).
 */
Expr GetNthAccessExpr(const Expr& block, int index, bool is_write) {
  CHECK(block.As<ScheduleBlockRealize>());
  auto compute_body = block.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body;
  if (is_write) {
    std::vector<Expr> find_store_vec;
    auto find_store = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) {
      if (x->As<ir::Store>()) find_store_vec.push_back(*x);
      return x->As<ir::Store>();
    });
    CHECK_EQ(find_store.size(), find_store_vec.size());
    CHECK_LT(index, (int)find_store.size());
    Expr store_index = find_store_vec[index];
    return store_index;
  } else {
    std::vector<Expr> find_load_vec;
    auto find_load = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) {
      if (x->As<ir::Load>()) find_load_vec.push_back(*x);
      return x->As<ir::Load>();
    });
    CHECK_EQ(find_load.size(), find_load_vec.size());
    CHECK_LT(index, (int)find_load.size());
    Expr load_index = find_load_vec[index];
    return load_index;
  }
}

/**
 * Make a tensor's cache tensor.
 * @param tensor The original tensor.
 * @param memory_type The memory type of the cache tensor.
 * @param return The tensor's cache tensor.
 */
Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type) {
  auto cache_tensor = lang::Compute(
      tensor->shape, [=](const std::vector<Expr>& dims) { return tensor(dims); }, tensor->name + "_" + memory_type);
  cache_tensor->WithBuffer(memory_type);
  return cache_tensor;
}

/**
 * Make a the cache tensor's block.
 * @param buffer_region The accessed region of cache tensor.
 * @param info The information of cache block.
 * @param memory_type The memory type of cache tensor.
 * @param device_api The device api of this Expr.
 * @param return ScheduleBlockRealize of the cache tensor.
 */
Expr MakeCacheBlock(const std::vector<std::pair<Expr, Expr>>& buffer_region,
                    CacheBlockInfo* info,
                    const std::string& memory_type,
                    DeviceAPI device_api) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<Expr> iter_values;
  // Create loop vars and block vars' binding_value
  for (auto& axis_range : buffer_region) {
    Var loop_var("ax" + std::to_string(loop_vars.size()));
    loop_vars.push_back(loop_var);
    iter_values.push_back(common::AutoSimplify(axis_range.first + loop_var));
  }
  // block variables
  std::vector<Var> block_vars;
  Tensor new_tensor = info->alloc;
  // Create block vars, block's accessed region and accessing indices
  CHECK(new_tensor->buffer.defined());
  for (auto& dim : new_tensor->buffer->shape) {
    Var var(Expr(0), dim, "v" + std::to_string(block_vars.size()), false);
    block_vars.push_back(var);
  }
  auto body                  = new_tensor->tensor_store_expanded_body();
  std::vector<Var> axis_vars = common::GenDefaultAxis(new_tensor->domain.size());
  axis_vars.insert(axis_vars.end(), new_tensor->reduce_axis.begin(), new_tensor->reduce_axis.end());
  for (int i = 0; i < axis_vars.size(); ++i) {
    optim::ReplaceVarWithExpr(&body, axis_vars[i], block_vars[i]);
  }
  Expr block = ir::ScheduleBlockRealize::Make(
      iter_values,
      ir::ScheduleBlock::Make(block_vars, {}, {}, common::UniqName(new_tensor->name), Block::Make({body})));
  Expr new_body = block;
  for (int i = (int)loop_vars.size() - 1; i >= 0; i--) {
    new_body = For::Make(loop_vars[i],
                         Expr(0),
                         common::AutoSimplify(buffer_region[i].second - buffer_region[i].first),
                         ir::ForType::Serial,
                         device_api,
                         ir::Block::Make({new_body}));
  }
  info->cache_block = std::move(new_body);
  return block;
}

struct CacheReadRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheReadRewriter rewriter(root, info);
    Expr new_root = optim::IRCopy(root);
    rewriter(&new_root);
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheReadRewriter(const Expr& root, CacheBlockInfo* info) : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    if (expr->tensor == Expr(info_->read_tensor)) {
      IRMutator::Visit(expr, op);
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
};

struct CacheWriteRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheWriteRewriter rewriter(root, info);
    Expr new_root               = optim::IRCopy(root);
    rewriter.mutate_cache_block = true;
    rewriter(&info->cache_block);
    rewriter.mutate_cache_block = false;
    rewriter(&new_root);
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheWriteRewriter(const Expr& root, CacheBlockInfo* info) : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (op->As<ScheduleBlock>()->name == info_->write_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->read_tensor->name;
    } else if (op->As<ScheduleBlock>()->name == info_->read_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->write_tensor->name;
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Load>()->tensor == Expr(info_->write_tensor) && mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Load>()->tensor == Expr(info_->read_tensor) && mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    }
  }

  void Visit(const ir::Store* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Store>()->tensor == Expr(info_->write_tensor)) {
      op->As<Store>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Store>()->tensor == Expr(info_->read_tensor) && mutate_cache_block) {
      op->As<Store>()->tensor = Expr(info_->write_tensor);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
  /*! \brief Are we mutating the cache tensor's block */
  bool mutate_cache_block{true};
};

//! Visit all ScheduleBlock and change its body to ir::Block if it is not.
struct ChangeBodyToBlock : public ir::IRMutator<> {
 public:
  static void Change(Expr* expr) {
    ChangeBodyToBlock mutator;
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (!op->As<ScheduleBlock>()->body.As<Block>()) {
      op->As<ScheduleBlock>()->body = Block::Make({op->As<ScheduleBlock>()->body});
    }
    IRMutator::Visit(expr, op);
  }
};

/**
 * Fidn cache tensor block's insertion point in the whole AST(root).
 * @param root The whole AST.
 * @param info The information of cache block.
 * @param is_write Are we inserting a write cache tensor or a read cache tensor.
 */
void FindInsertionPoint(Expr& root, CacheBlockInfo* info, bool is_write) {
  Expr find_tensor       = is_write ? Expr(info->write_tensor) : Expr(info->read_tensor);
  auto find_produce_read = ir::CollectIRNodesWithoutTensor(
      root, [&](const Expr* x) { return x->As<ir::Store>() && x->As<ir::Store>()->tensor == find_tensor; });

  if (find_produce_read.empty()) {
    CHECK(root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>());
    CHECK(root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body.As<Block>());
    info->loc_block = root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body;
    info->loc_pos   = 0;
    return;
  }

  CHECK_EQ(find_produce_read.size(), 1U);
  Expr producer = *(find_produce_read.begin());

  CHECK(root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>());
  CHECK(root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body.As<Block>());
  info->loc_block = root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body;
  for (int i = 0; i < (int)info->loc_block.As<Block>()->stmts.size(); ++i) {
    if (Contains(info->loc_block.As<Block>()->stmts[i], producer)) {
      info->loc_pos = i + 1;
      break;
    }
  }
}

DeviceAPI IRSchedule::GetDeviceAPI() const {
  auto exprs          = this->GetModule().GetExprs();
  auto find_for_nodes = ir::CollectIRNodesWithoutTensor(exprs.front(), [&](const Expr* x) { return x->As<ir::For>(); });
  return (*find_for_nodes.begin()).As<ir::For>()->device_api;
}

Expr IRSchedule::CacheRead(const Expr& block, int read_tensor_index, const std::string& memory_type) {
  CHECK(block.As<ScheduleBlockRealize>());
  auto root = GetRootBlock(block);
  ChangeBodyToBlock::Change(&root);
  Expr read_expr = GetNthAccessExpr(block, read_tensor_index, false);
  CHECK(read_expr.As<ir::Load>());
  auto tensor_indices = read_expr.As<ir::Load>()->indices;
  CacheBlockInfo info;
  info.read_tensor  = read_expr.As<ir::Load>()->tensor.as_tensor_ref();
  info.write_tensor = MakeCacheTensor(info.read_tensor, memory_type);
  info.alloc        = info.write_tensor;

  auto read_buffer_region = CalculateTensorRegions(block, tensor_indices, info.read_tensor, root);
  auto new_block          = MakeCacheBlock(read_buffer_region, &info, memory_type, this->GetDeviceAPI());
  FindInsertionPoint(root, &info, false);
  auto new_root = CacheReadRewriter::Rewrite(root, &info);
  helper_.Replace(root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body,
                  new_root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body);
  return new_block;
}

Expr IRSchedule::CacheWrite(const Expr& block, int write_buffer_index, const std::string& memory_type) {
  CHECK(block.As<ScheduleBlockRealize>());
  auto root = GetRootBlock(block);
  ChangeBodyToBlock::Change(&root);
  Expr write_expr = GetNthAccessExpr(block, write_buffer_index, true);
  CHECK(write_expr.As<ir::Store>());
  Tensor write_tensor = write_expr.As<ir::Store>()->tensor.as_tensor_ref();
  auto tensor_indices = write_expr.As<ir::Store>()->indices;
  CacheBlockInfo info;
  info.read_tensor         = MakeCacheTensor(write_tensor, memory_type);
  info.write_tensor        = write_tensor;
  info.alloc               = info.read_tensor;
  auto write_buffer_region = CalculateTensorRegions(block, tensor_indices, info.write_tensor, root);
  auto new_block           = MakeCacheBlock(write_buffer_region, &info, memory_type, this->GetDeviceAPI());
  FindInsertionPoint(root, &info, true);

  auto new_root = CacheWriteRewriter::Rewrite(root, &info);
  helper_.Replace(root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body,
                  new_root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body);

  auto find_cache_block = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
    return x->As<ir::ScheduleBlockRealize>() && !x->As<ir::ScheduleBlockRealize>()->iter_values.empty() &&
           GetTensor(*x)->name == info.read_tensor->name;
  });

  CHECK_EQ(find_cache_block.size(), 1U);

  return *find_cache_block.begin();
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
  CHECK((src_sref.As<ir::For>() && tgt_stmt.As<ir::For>()) || (src_sref.As<ir::Block>() && tgt_stmt.As<ir::Block>()) ||
        (src_sref.As<ir::ScheduleBlockRealize>() && tgt_stmt.As<ir::ScheduleBlockRealize>()));
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

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Block* op, Expr* expr) override {
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
    VLOG(3) << "Origin Expr is: \n" << i;
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
    for (int i = 0; i < static_cast<int>(if_nodes.size()); ++i) {
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
    CHECK_GE(i, 0) << "The loop index in Reorder should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  this->Reorder(loops_expr);
}

void IRSchedule::Reorder(const Expr& block, const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size()) << "The loop index in Reorder should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Reorder should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  this->Reorder(loops_expr);
}

Expr IRSchedule::GetRootBlock(const Expr& expr) const {
  auto exprs = this->GetModule().GetExprs();
  for (auto& it_expr : exprs) {
    auto find_expr = ir::CollectIRNodesWithoutTensor(it_expr, [&](const Expr* x) { return *x == expr; });
    if (!find_expr.empty()) {
      CHECK(it_expr.As<ir::Block>());
      CHECK_EQ(it_expr.As<ir::Block>()->stmts.size(), 1U);
      CHECK(it_expr.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
      return it_expr.As<ir::Block>()->stmts[0];
    }
  }
  LOG(FATAL) << "Didn't find expr in IRSchedule:\n" << expr;
}

/*!
 * \brief Find producers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its producers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's producers(Load nodes) in root.
 */
std::vector<Expr> GetProducers(const Expr& block, const Expr& root) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(root.As<ir::ScheduleBlockRealize>());
  std::vector<Expr> producers;
  auto compute_body = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
  auto find_load    = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Load>(); });
  for (auto& i : find_load) producers.emplace_back(i);
  return producers;
}

/*!
 * \brief Find consumers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its consumers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's consumers(ScheduleBlockRealize nodes) in root.
 */
std::vector<Expr> GetConsumers(const Expr& block, const Expr& root) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(root.As<ir::ScheduleBlockRealize>());
  std::vector<Expr> consumers;
  std::string block_tensor = GetTensor(block)->name;
  auto find_block          = ir::CollectIRNodesWithoutTensor(
      root, [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>() && *x != block && *x != root; });
  for (auto& i : find_block) {
    CHECK(i.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
    auto block_body = i.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
    auto find_load  = ir::CollectIRNodesWithoutTensor(block_body, [&](const Expr* x) {
      return x->As<ir::Load>() && x->As<ir::Load>()->tensor.as_tensor_ref()->name == block_tensor;
    });
    if (!find_load.empty()) consumers.emplace_back(i);
  }
  return consumers;
}

/*!
 * \brief Check if the params of ComputeAt is validate.
 * \param block The block node we want to move in ComputeAt.
 * \param loop The for node we want to put the block under in ComputeAt.
 * \param root The root ScheduleBlockRealize node of block and loop.
 */
void CheckComputeAtValidation(const Expr& block, const Expr& loop, const Expr& root) {
  auto find_block = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) { return *x == block; });
  CHECK(!find_block.empty()) << "Didn't find block in root!";

  auto find_loop = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) { return *x == loop; });
  CHECK(!find_loop.empty()) << "Didn't find loop in root!";

  auto find_block_in_loop = ir::CollectIRNodesWithoutTensor(loop, [&](const Expr* x) { return *x == block; });
  CHECK(find_block_in_loop.empty()) << "loop should not be block's ancestor!";
}

/*!
 * \brief Insert a new ScheduleBlockRealize in a loop's body(under its IfThenElse Node, if any)
 * \param for_loop The for loop whose body we want to modify
 * \param insertion The ScheduleBlockRealize we want to insert
 */
void InsertBlock(Expr& for_loop, const Expr& insertion) {
  CHECK(for_loop.As<ir::For>());
  CHECK(for_loop.As<ir::For>()->body.As<Block>());
  Expr& block = for_loop.As<ir::For>()->body;
  if (block.As<Block>()->stmts[0].As<IfThenElse>()) {
    CHECK(block.As<Block>()->stmts[0].As<IfThenElse>()->true_case.As<Block>());
    Expr& insert_block = block.As<Block>()->stmts[0].As<IfThenElse>()->true_case;
    insert_block.As<Block>()->stmts.insert(insert_block.As<Block>()->stmts.begin(), insertion);
  } else {
    block.As<Block>()->stmts.insert(block.As<Block>()->stmts.begin(), insertion);
  }
}

// The struct used to reconstruct the new For node to replace the old For node.
struct LoopReconstructor : public ir::IRMutator<> {
 public:
  explicit LoopReconstructor(const Expr& root, const Expr& block, const Expr& loop)
      : root_(root), block_(block), loop_(loop) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  void MakeNewLoop(const std::vector<std::pair<Expr, Expr>>& iter_doms) {
    int n_iters = iter_doms.size();
    std::vector<Var> loop_vars;
    std::vector<Expr> loop_extents;
    std::vector<Expr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    for (int i = 0; i < n_iters; ++i) {
      auto iter_dom = iter_doms[i];
      if (iter_dom.second != Expr(1)) {
        Var var("ax" + std::to_string(loop_vars.size()), Int(32));
        loop_vars.push_back(var);
        loop_extents.push_back(iter_dom.second);
        iter_values.push_back(common::AutoSimplify(iter_dom.first) + var);
      } else {
        iter_values.push_back(common::AutoSimplify(iter_dom.first));
      }
    }
    auto schedule_block_node = block_.As<ir::ScheduleBlockRealize>()->schedule_block;
    new_block_               = ScheduleBlockRealize::Make(std::move(iter_values), std::move(schedule_block_node));
    Expr loop_body           = new_block_;
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      auto loop_var    = loop_vars[i];
      auto loop_extent = loop_extents[i];
      if (!loop_body.As<ir::Block>()) loop_body = Block::Make({loop_body});
      loop_body = For::Make(loop_var,
                            Expr(0),
                            loop_extent,
                            loop_.As<ir::For>()->for_type(),
                            loop_.As<ir::For>()->device_api,
                            std::move(loop_body));
    }
    new_loop_ = optim::IRCopy(loop_);
    InsertBlock(new_loop_, loop_body);
    return;
  }

 private:
 public:
  /*! \brief The root block */
  Expr root_;
  /*! \brief The given block to be moved */
  Expr block_;
  /*! \brief The given loop the block and its loop nest to be put under */
  Expr loop_;
  /*! \brief The new loop to replace the original loop */
  Expr new_loop_{nullptr};
  /*! \brief The new block realize to the moved block */
  Expr new_block_{nullptr};
  /*! \brief The plan to remove the given block by replacing this loop/block in the AST */
  Expr source_expr{nullptr};
  /*! \brief The plan to remove the given block by replacing to this loop/block in the AST */
  Expr target_expr{nullptr};
};

// The struct used to remove the original block in ComputeAt.
struct LeafBlockRemovalPlan : public ir::IRMutator<> {
  LeafBlockRemovalPlan(const Expr& block, Expr* source_expr, Expr* target_expr)
      : block_(block), source_expr_(source_expr), target_expr_(target_expr) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize* expr, Expr* op) override {
    if (*op == block_) {
      find_block = true;
      return;
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (*op == block_) {
      find_block = true;
      return;
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Block* expr, Expr* op) override {
    if (expr->stmts.size() > 1U) {
      int block_index = -1;
      for (int i = 0; i < expr->stmts.size(); ++i) {
        auto keep_flag = find_block;
        find_block     = false;
        auto* node     = op->As<ir::Block>();
        IRMutator::Visit(&node->stmts[i], &node->stmts[i]);
        if (find_block) {
          if (depth == 0) {
            *source_expr_ = *op;
            block_index   = i;
          }
          depth++;
        }
        find_block = find_block || keep_flag;
      }
      if (block_index != -1) {
        std::vector<Expr> new_stmts;
        for (int i = 0; i < expr->stmts.size(); ++i) {
          if (i == block_index)
            continue;
          else
            new_stmts.push_back(expr->stmts[i]);
        }
        auto target_block = ir::Block::Make(new_stmts);
        *target_expr_     = target_block;
      }
    } else {
      IRMutator::Visit(expr, op);
    }
  }

 private:
  bool find_block{false};
  int depth{0};
  const Expr& block_;
  Expr* source_expr_;
  Expr* target_expr_;
};

void IRSchedule::SetBuffer(Expr& block, const std::string& memory_type) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::CollectIRNodesWithoutTensor(block, [&](const Expr* x) { return x->As<ir::Store>(); });
  CHECK(!find_tensor.empty()) << "Didn't find Store in block!";
  CHECK_EQ(find_tensor.size(), 1U) << "One block should only have one Store node!(except for root block)";
  auto& tensor = (*find_tensor.begin()).As<ir::Store>()->tensor;
  tensor.as_tensor_ref()->WithBuffer(memory_type, "_" + tensor.as_tensor_ref()->name + "_temp_buffer");

  auto exprs = this->GetModule().GetExprs();
  for (auto& it_expr : exprs) {
    auto find_tensor = ir::CollectIRNodes(
        it_expr, [&](const Expr* x) { return x->as_tensor() && x->as_tensor()->name == tensor.as_tensor_ref()->name; });
    for (auto& t : find_tensor) {
      CHECK(t.as_tensor());
      t.as_tensor_ref()->Bind(tensor.as_tensor_ref()->buffer);
    }
  }
}

void IRSchedule::MergeExprs() {
  auto exprs = this->GetModule().GetExprs();
  if (exprs.size() == 1U) return;
  CHECK(exprs[0].As<ir::Block>());
  CHECK_EQ(exprs[0].As<ir::Block>()->stmts.size(), 1U);
  CHECK(exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
  CHECK(exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
  std::vector<Expr> merged_block;
  merged_block.push_back(
      exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body);
  VLOG(3) << "Before merging, exprs[0] is : " << exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
    auto root_block = ir::CollectIRNodes(exprs[i], [&](const Expr* x) {
      return x->As<ir::ScheduleBlockRealize>() && x->As<ir::ScheduleBlockRealize>()->iter_values.empty();
    });
    CHECK_EQ(root_block.size(), 1U);
    for (auto& it_block : root_block) {
      auto& block_body = it_block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
      merged_block.push_back(block_body);
    }
  }
  for (auto& block : merged_block) {
    VLOG(3) << "in merged_block, it has " << block;
  }
  auto merged_expr = ir::Block::Make(merged_block);
  exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body =
      merged_expr;
  VLOG(3) << "After merging, exprs[0] is : " << exprs[0];
  exprs.erase(exprs.begin() + 1, exprs.end());
  this->SetExprs(exprs);
}

/*!
 * \brief Make a union of two range. The detailed function is :
 * new_range.min = min(range1.min, range2.min)
 * new_range.extent = max(range1.min + range1.extent, range2.min + range2.extent) - new_range.min
 * Notice that the pair<Expr, Expr> indicates a range's min and extent.
 * \param range1 The first range
 * \param range2 The second range
 * \return The union of these two ranges
 */
std::pair<Expr, Expr> RangeUnion(const std::pair<Expr, Expr>& range1, const std::pair<Expr, Expr>& range2) {
  Expr new_min    = common::AutoSimplify(Min::Make(range1.first, range2.first));
  Expr new_extent = common::AutoSimplify(
      common::AutoSimplify(Max::Make(range1.first + range1.second, range2.first + range2.second)) - new_min);
  return std::make_pair(new_min, new_extent);
}

/*!
 * \brief Calculate the required buffer region given a block and its consumers.
 * For example, if block is :
 * B[i0, j0] = A[i0, j0]
 * loop is :
 * for (i, 0, 64) {
 *   for (j, 0, 64) {
 *     C[i, j] = B[i, j]
 *   }
 * }
 * And consumers is :
 * C[i, j] = B[i, j]
 * Then we get the consumer requires B's region:
 * B[i, j], where:
 * i : [i, i]
 * j : [0, 64]
 * \param block The ScheduleBlockRealize node begin required
 * \param loop The loop where we will insert the block under it
 * \param consumers Vector of ScheduleBlockRealize nodes that require the block
 * \return Each index's range of block's tensor. Indicating the buffer region being required.
 */
std::vector<std::pair<Expr, Expr>> CalculateRequiredRegions(const Expr& block,
                                                            const Expr& loop,
                                                            const std::vector<Expr>& consumers,
                                                            const Expr& root) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  std::string block_tensor = GetTensor(block)->name;
  std::vector<std::pair<Expr, Expr>> required_buffer_range;
  CHECK(loop.As<ir::For>());
  for (auto& i : consumers) {
    CHECK(i.As<ir::ScheduleBlockRealize>());
    Expr block_body = optim::IRCopy(i.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body);
    auto iter_var   = i.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
    auto iter_value = i.As<ir::ScheduleBlockRealize>()->iter_values;
    ReplaceExpr(&block_body, iter_var, iter_value);
    // Notice that we look for For nodes in loop's body instead of loop itself.
    auto find_loops = ir::CollectIRNodesWithoutTensor(
        loop.As<ir::For>()->body, [&](const Expr* x) { return x->As<ir::For>() && Contains(*x, i); });
    auto find_load = ir::CollectIRNodesWithoutTensor(block_body, [&](const Expr* x) {
      return x->As<ir::Load>() && x->As<ir::Load>()->tensor.as_tensor_ref()->name == block_tensor;
    });
    for (auto& load : find_load) {
      CHECK(load.As<ir::Load>());
      auto indices = load.As<ir::Load>()->indices;
      if (find_loops.empty()) {
        for (int i = 0; i < indices.size(); ++i) {
          if (i >= required_buffer_range.size())
            required_buffer_range.push_back(std::make_pair(indices[i], Expr(1)));
          else
            required_buffer_range[i] = RangeUnion(required_buffer_range[i], std::make_pair(indices[i], Expr(1)));
        }
      } else {
        for (int i = 0; i < indices.size(); ++i) {
          Expr indice_min = optim::IRCopy(indices[i]);
          Expr indice_max = optim::IRCopy(indices[i]);
          std::vector<Var> loop_vars;
          std::vector<Expr> vars_min;
          std::vector<Expr> vars_max;
          for (auto& for_loop : find_loops) {
            loop_vars.push_back(for_loop.As<ir::For>()->loop_var);
            vars_min.push_back(for_loop.As<ir::For>()->min);
            vars_max.push_back(for_loop.As<ir::For>()->min + for_loop.As<ir::For>()->extent);
          }
          Expr mod_extent(0);
          if (indice_min.As<Mod>() && indice_min.As<Mod>()->b().is_constant()) mod_extent = indice_min.As<Mod>()->b();
          ReplaceExpr(&indice_min, loop_vars, vars_min);
          ReplaceExpr(&indice_max, loop_vars, vars_max);
          Expr indice_extent;
          // If a index keeps constant, its extent should be 1.
          if (common::AutoSimplify(indice_min) == common::AutoSimplify(indice_max)) {
            if (common::is_zero(mod_extent)) {
              indice_extent = Expr(1);
            } else {
              indice_extent = mod_extent;
            }
          } else {
            indice_extent = common::AutoSimplify(common::AutoSimplify(indice_max) - common::AutoSimplify(indice_min));
          }
          if (indice_extent.is_constant() && indice_extent.get_constant() < 0) {
            indice_min    = common::AutoSimplify(indice_max);
            indice_extent = Expr(-indice_extent.get_constant());
          }
          if (i >= required_buffer_range.size()) {
            required_buffer_range.push_back(std::make_pair(indice_min, indice_extent));
          } else {
            required_buffer_range[i] = RangeUnion(required_buffer_range[i], std::make_pair(indice_min, indice_extent));
          }
        }
      }
    }
  }
  int iter_size = block.As<ir::ScheduleBlockRealize>()->iter_values.size();
  if (iter_size > required_buffer_range.size()) {
    for (int i = required_buffer_range.size(); i < iter_size; ++i) {
      CHECK(block.As<ir::ScheduleBlockRealize>()->iter_values[i].as_var() ||
            block.As<ir::ScheduleBlockRealize>()->iter_values[i].is_constant());
      if (block.As<ir::ScheduleBlockRealize>()->iter_values[i].as_var()) {
        auto find_for_loops = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
          return x->As<ir::For>() && x->As<ir::For>()->loop_var->name ==
                                         block.As<ir::ScheduleBlockRealize>()->iter_values[i].as_var_ref()->name;
        });
        CHECK_EQ(find_for_loops.size(), 1U);
        required_buffer_range.push_back(std::make_pair((*find_for_loops.begin()).As<ir::For>()->min,
                                                       (*find_for_loops.begin()).As<ir::For>()->extent));
      } else {
        int cons = (int)block.As<ir::ScheduleBlockRealize>()->iter_values[i].is_constant();
        required_buffer_range.push_back(std::make_pair(Expr(cons), Expr(1)));
      }
    }
  }
  return required_buffer_range;
}

void IRSchedule::ComputeAt(const Expr& block, const Expr& loop) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(loop.As<ir::For>());
  Expr root      = this->GetRootBlock(block);
  auto producers = GetProducers(block, root);
  auto consumers = GetConsumers(block, root);
  CheckComputeAtValidation(block, loop, root);
  LoopReconstructor reconstructor(root, block, loop);
  LeafBlockRemovalPlan remove_plan(block, &reconstructor.source_expr, &reconstructor.target_expr);
  remove_plan(&root);
  auto iter_doms = CalculateRequiredRegions(block, loop, consumers, root);
  for (auto& i : iter_doms) VLOG(3) << "CalculateRequiredRegions is : " << i.first << " to " << i.second;
  reconstructor.MakeNewLoop(iter_doms);
  helper_.Replace(reconstructor.source_expr, reconstructor.target_expr);
  helper_.Replace(reconstructor.loop_, reconstructor.new_loop_);
  return;
}

void IRSchedule::SimpleComputeAt(const Expr& block, const Expr& loop) {
  VLOG(3) << "Begin SimpleComputeAt of block:\n" << block << " and loop:\n" << loop;
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(loop.As<ir::For>());
  std::vector<Expr> block_loops = this->GetLoops(block);
  Expr root                     = this->GetRootBlock(block);
  auto loops                    = GetLoopsOfExpr(loop, root);
  auto this_loop                = loop;
  auto block_name               = GetTensor(block)->name;
  auto this_block               = block;
  if (GetLoopExtent(loops[0]) == 1 && GetLoopExtent(block_loops[0]) != 1) {
    this->Split(block_loops[0], {1, -1});
    this_block = this->GetBlock(block_name);
  } else if (GetLoopExtent(loops[0]) != 1 && GetLoopExtent(block_loops[0]) == 1) {
    auto splited = this->Split(loops[0], {1, -1});
    this_loop    = splited[1];
  }

  block_loops = this->GetLoops(this_block);
  root        = this->GetRootBlock(this_block);
  loops       = GetLoopsOfExpr(this_loop, root);

  CHECK_LE(loops.size(), block_loops.size());

  std::vector<Var> replaced_var;
  std::vector<Expr> substitute_expr;
  for (int i = 0; i < loops.size(); ++i) {
    CHECK_EQ(GetLoopExtent(loops[i]), GetLoopExtent(block_loops[i]));
    if (block_loops[i].As<ir::For>()->bind_info().valid() && !loops[i].As<ir::For>()->bind_info().valid()) {
      loops[i].As<ir::For>()->set_bind_info(block_loops[i].As<ir::For>()->bind_info());
    }
    replaced_var.push_back(block_loops[i].As<ir::For>()->loop_var);
    substitute_expr.push_back(Expr(loops[i].As<ir::For>()->loop_var));
  }
  Expr result =
      loops.size() < block_loops.size() ? optim::IRCopy(block_loops[loops.size()]) : optim::IRCopy(this_block);
  ReplaceExpr(&result, replaced_var, substitute_expr);
  Expr new_loop                = optim::IRCopy(this_loop);
  new_loop.As<ir::For>()->body = ir::Block::Make({result, new_loop.As<ir::For>()->body});
  Expr source_expr{nullptr};
  Expr target_expr{nullptr};
  LeafBlockRemovalPlan remove_plan(result, &source_expr, &target_expr);
  remove_plan(&root);
  helper_.Replace(source_expr, target_expr);
  helper_.Replace(this_loop, new_loop);
  return;
}

/*!
 * \brief The base class of the inliner, which handles:
 * 1) Remove the block to be lined
 * 2) Maintain a list of index variables and their substition of the buffer being inlined
 */
class BaseInliner : public ir::IRMutator<> {
 protected:
  explicit BaseInliner(const Tensor& inlined_tensor, const Expr& inlined_store)
      : inlined_tensor_(inlined_tensor), inlined_store_(inlined_store) {}

 public:
  void operator()(Expr* expr) {
    IRMutator::Visit(&tgt_stmt, &tgt_stmt);
    IRMutator::Visit(expr, expr);
  }

 private:
  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == src_stmt) {
      *op = tgt_stmt;
      return;
    }
    IRMutator::Visit(expr, op);
  }

 protected:
  //! Check if indices are validate. If so, set idx_vars_ properly.
  bool UpdateAndCheckIndexVars(const std::vector<Expr>& indices, int expected_ndim) {
    int n = indices.size();
    if (n != expected_ndim) {
      return false;
    }
    std::vector<Var> result;
    result.reserve(n);
    for (auto& i : indices) {
      if (i.as_var()) {
        result.push_back(i.as_var_ref());
      } else {
        return false;
      }
    }
    int n_distinct = std::set<Var, CompVar>(result.begin(), result.end()).size();
    if (n != n_distinct) {
      return false;
    }
    if (idx_vars_.empty()) {
      idx_vars_ = std::move(result);
    } else {
      if (idx_vars_.size() != result.size()) return false;
      for (int i = 0; i < result.size(); ++i) {
        if (Expr(idx_vars_[i]) != Expr(result[i])) return false;
      }
    }
    return true;
  }

  void SetIndexSubstitution(const std::vector<Expr>& indices) {
    CHECK_EQ(indices.size(), idx_vars_.size());
    int n = idx_vars_.size();
    idx_sub_var_.reserve(n);
    idx_sub_expr_.reserve(n);
    for (int i = 0; i < n; ++i) {
      idx_sub_var_.push_back(idx_vars_[i]);
      idx_sub_expr_.push_back(indices[i]);
    }
  }

 protected:
  //! The tensor to be inlined
  Tensor inlined_tensor_{nullptr};
  //! The body of the block to be inlined
  Expr inlined_store_{nullptr};
  //! The indices used for indexing the buffer to be inlined
  std::vector<Var> idx_vars_;
  //! Replacing vars(idx_sub_var_) in indices to corresponding expr(idx_sub_expr_)
  std::vector<Var> idx_sub_var_;
  std::vector<Expr> idx_sub_expr_;

 public:
  /*!
   * \brief The Expr to be replaced when removing the block
   * \note The pair (src_stmt, tgt_stmt) are produced by LeafBlockRemovalPlan
   */
  Expr src_stmt{nullptr};
  //! The Expr to replace the original one when removing the block
  Expr tgt_stmt{nullptr};
};

/*!
 * \brief Helper to inline the producer block into its consumer(s)
 * The derived class implements:
 * Substitute `Load` on the tensor to be inlined to its value calculation in the producer block
 */
class ComputeInliner : public BaseInliner {
 public:
  explicit ComputeInliner(const Tensor& inlined_tensor, const Expr& inlined_store)
      : BaseInliner(inlined_tensor, inlined_store) {}

  bool BodyPatternAllowInline() {
    if (!inlined_store_.defined()) {
      return false;
    }
    CHECK(inlined_store_.As<Store>());
    auto find_vars = ir::CollectIRNodesWithoutTensor(inlined_store_, [&](const Expr* x) { return x->as_var(); });
    std::set<Var, CompVar> vars_set;
    for (auto& i : find_vars) vars_set.insert(i.as_var_ref());
    int n_vars = vars_set.size();
    if (!UpdateAndCheckIndexVars(inlined_store_.As<Store>()->indices, n_vars)) {
      return false;
    }
    return true;
  }

 private:
  void Visit(const ir::Load* expr, Expr* op) override {
    if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
      *op = ReplaceInlinedTensor(op);
      return;
    }
    IRMutator::Visit(expr, op);
  }

  //! Replace the 'Load' node on the tensor to 'Load' node of its producers.
  Expr ReplaceInlinedTensor(Expr* load) {
    CHECK(load->As<ir::Load>());
    SetIndexSubstitution(load->As<ir::Load>()->indices);
    Expr value_copy = optim::IRCopy(inlined_store_.As<Store>()->value);
    ReplaceExpr(&value_copy, idx_sub_var_, idx_sub_expr_);
    return value_copy;
  }
};

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block, const Expr& root) {
  CHECK(schedule_block.As<ir::ScheduleBlockRealize>());
  auto compute_body = schedule_block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
  // 1. Check the schedule block to be inlined is not a reduce tensor.
  auto find_store = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Store>(); });
  CHECK_EQ(find_store.size(), 1U);
  Expr tensor = (*find_store.begin()).As<ir::Store>()->tensor;
  CHECK(!tensor.as_tensor_ref()->is_reduce_tensor());
  // 2. Check this schedule block is the only writer of the tensor.
  find_store = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
    return x->As<ir::Store>() && (x->As<ir::Store>()->tensor).as_tensor_ref()->name == tensor.as_tensor_ref()->name;
  });
  CHECK_EQ(find_store.size(), 1U);
  // 3. Check there is no overlap between the buffers the schedule block reads and writes.
  auto find_load = ir::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Load>() && x->As<ir::Load>()->tensor == tensor; });
  CHECK(find_load.empty());
  return (*find_store.begin());
}

void IRSchedule::ComputeInline(const Expr& schedule_block) {
  CHECK(schedule_block.As<ir::ScheduleBlockRealize>());
  Expr root  = this->GetRootBlock(schedule_block);
  Expr store = CheckComputeInlineValidationAndGetStore(schedule_block, root);
  ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(), store);
  CHECK(inliner.BodyPatternAllowInline());
  // Create a plan that removes the block to be inlined
  LeafBlockRemovalPlan remove_plan(schedule_block, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  inliner(&root);
  return;
}

std::vector<Expr> ScheduleHelper::GetLoops(const Expr& block) const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;

  std::set<std::string> loops_name;
  for (auto& iter_val : block.As<ir::ScheduleBlockRealize>()->iter_values) {
    auto vars = ir::CollectIRNodes(iter_val, [&](const Expr* x) { return x->is_var(); });
    for (auto& iter_var : vars) loops_name.insert(iter_var.as_var_ref()->name);
  }

  for (auto& it_expr : exprs) {
    auto find_block = ir::CollectIRNodes(it_expr, [&](const Expr* x) {
      return x->As<ir::ScheduleBlockRealize>() &&
             x->As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>() &&
             x->As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name == block_name;
    });
    if (!find_block.empty()) {
      if (!result.empty()) LOG(FATAL) << "Find block with name: \n" << block_name << " appeared in more than one AST!";
      auto loop_nodes = ir::CollectIRNodes(it_expr, [&](const Expr* x) {
        return x->As<ir::For>() && loops_name.count(x->As<ir::For>()->loop_var->name) != 0;
      });
      for (auto& it_for : loop_nodes) {
        if (Contains(it_for, block)) result.push_back(it_for);
      }
    }
  }
  if (result.empty())
    LOG(FATAL) << "Didn't find Loops containing ScheduleBlock with name: \n" << block_name << " in ModuleExepr.";
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
    auto block_nodes = ir::CollectIRNodes(it_expr, [&](const Expr* x) {
      if (x->As<ir::ScheduleBlockRealize>() && !x->As<ir::ScheduleBlockRealize>()->iter_values.empty())
        result.push_back(*x);
      return x->As<ir::ScheduleBlockRealize>() && !x->As<ir::ScheduleBlockRealize>()->iter_values.empty();
    });
  }
  CHECK(!result.empty()) << "Didn't find blocks in expr.";
  for (auto& it_expr : exprs) {
    VLOG(3) << "it_expr is : " << it_expr;
  }
  return result;
}

Expr ScheduleHelper::GetBlock(const std::string& block_name) const {
  Expr result;
  std::vector<Expr> all_blocks = this->GetAllBlocks();
  for (auto& it_block : all_blocks) {
    if (GetTensor(it_block)->name == block_name) result = it_block;
  }
  if (!result.defined()) LOG(FATAL) << "Didn't find a block with name " << block_name << " in this ModuleExpr!";
  return result;
}

void SetCudaAxisInfo(Expr* lowered_func) {
  if (!lowered_func->as_lowered_func()) {
    LOG(ERROR) << "The input of SetCudaAxisInfo should be lowered_func!";
    return;
  }

  auto func_body = lowered_func->as_lowered_func_ref()->body;
  CudaAxisInfo info;

  auto block_nodes                                    = ir::CollectIRNodes(func_body, [&](const Expr* x) {
    if (x->As<ir::For>() && x->As<ir::For>()->bind_info().valid()) {
      auto bind_info = x->As<ir::For>()->bind_info();
      info.set_valid(true);
      if (bind_info.for_type == ForType::GPUThread) {
        CHECK(common::is_zero(x->As<ir::For>()->min));
        CHECK(x->As<ir::For>()->extent.is_constant());
        int range = x->As<ir::For>()->extent.get_constant();
        range     = range > info.block_dim(bind_info.offset) ? range : info.block_dim(bind_info.offset);
        VLOG(3) << "Set block dim[" << bind_info.offset << "] with range " << range;
        info.set_block_dim(bind_info.offset, range);
      } else if (bind_info.for_type == ForType::GPUBlock) {
        CHECK(common::is_zero(x->As<ir::For>()->min));
        CHECK(x->As<ir::For>()->extent.is_constant());
        int range = x->As<ir::For>()->extent.get_constant();
        range     = range > info.grid_dim(bind_info.offset) ? range : info.grid_dim(bind_info.offset);
        info.set_grid_dim(bind_info.offset, range);
        VLOG(3) << "Set grid dim[" << bind_info.offset << "] with range " << range;
      } else {
        LOG(FATAL) << "The for loop's bind info should be gpu block or thread!";
      }
    }
    return (x->As<ir::For>() && x->As<ir::For>()->bind_info().valid());
  });
  lowered_func->as_lowered_func_ref()->cuda_axis_info = info;
}

void IRSchedule::CopyTransformAndLoopInfo(const std::string& block_name, const std::string& block_target_name) {
  auto block        = this->GetBlock(block_name);
  auto block_target = this->GetBlock(block_target_name);
  this->CopyTransformAndLoopInfo(block, block_target);
}

void IRSchedule::CopyTransformAndLoopInfo(const Expr& block, const Expr& block_target) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block_target.As<ir::ScheduleBlockRealize>());
  auto exprs = this->GetModule().GetExprs();
  CHECK_EQ(exprs.size(), 1U);
  auto expr            = exprs[0];
  auto vars            = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
  auto vars_target     = block_target.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
  auto old_iter_values = block.As<ir::ScheduleBlockRealize>()->iter_values;
  auto iter_values_target = block_target.As<ir::ScheduleBlockRealize>()->iter_values;
  std::vector<Expr> new_iter_values;
  for (int i = 0; i < vars.size() && i < vars_target.size(); ++i) {
    CHECK(vars[i]->upper_bound.defined() && vars_target[i]->upper_bound.defined());
    if (vars[i]->upper_bound.is_constant() && vars_target[i]->upper_bound.is_constant() &&
        vars[i]->upper_bound.get_constant() == vars_target[i]->upper_bound.get_constant() && !vars[i]->is_reduce_axis &&
        !vars_target[i]->is_reduce_axis) {
      new_iter_values.push_back(iter_values_target[i]);
      VLOG(3) << "new_iter_values.push_back " << iter_values_target[i];
    } else
      break;
  }

  if (new_iter_values.empty())
    LOG(FATAL) << "Cannot CopyTransformAndLoopInfo since shape[0] of source and target is not equal! "
               << vars[0]->upper_bound << " v.s " << vars_target[0]->upper_bound;

  int changed_loop_num = new_iter_values.size();
  std::set<std::string> used_target_loop_vars;
  for (auto& iter_val : new_iter_values) {
    auto find_partial_loop = ir::CollectIRNodesWithoutTensor(iter_val, [&](const Expr* x) {
      if (x->as_var()) used_target_loop_vars.insert(x->as_var_ref()->name);
      return x->as_var();
    });
  }
  CHECK(!used_target_loop_vars.empty());
  std::vector<Expr> used_target_loops;
  auto expr_copy = optim::IRCopy(expr);
  for (auto& var : used_target_loop_vars) {
    auto find_loop_var = ir::CollectIRNodesWithoutTensor(expr_copy, [&](const Expr* x) {
      return x->As<ir::For>() && x->As<ir::For>()->loop_var->name == var && Contains(*x, block_target);
    });
    CHECK(!find_loop_var.empty());
    CHECK_EQ(find_loop_var.size(), 1U);
    used_target_loops.push_back(*find_loop_var.begin());
    VLOG(3) << "used_target_loops push_back " << used_target_loops.back();
  }
  std::sort(used_target_loops.begin(), used_target_loops.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });
  for (int i = new_iter_values.size(); i < old_iter_values.size(); ++i) {
    CHECK(old_iter_values[i].as_var());
    new_iter_values.push_back(old_iter_values[i]);
  }
  Expr new_loop;
  VLOG(3) << "changed_loop_num is : " << changed_loop_num;
  VLOG(3) << "old_iter_values.size() is : " << old_iter_values.size();
  if (changed_loop_num >= (int)old_iter_values.size()) {
    new_loop                                             = optim::IRCopy(block);
    new_loop.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  } else {
    CHECK(old_iter_values[changed_loop_num].as_var());
    auto old_var           = old_iter_values[changed_loop_num].as_var_ref();
    auto find_partial_loop = ir::CollectIRNodesWithoutTensor(expr, [&](const Expr* x) {
      return x->As<ir::For>() && x->As<ir::For>()->loop_var->name == old_var->name && Contains(*x, block);
    });
    CHECK(!find_partial_loop.empty());
    CHECK_EQ(find_partial_loop.size(), 1U);
    new_loop = optim::IRCopy(*find_partial_loop.begin());
    auto find_schedule_block =
        ir::CollectIRNodesWithoutTensor(new_loop, [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); });
    CHECK(!find_schedule_block.empty());
    CHECK_EQ(find_schedule_block.size(), 1U);
    Expr sch_block                                        = (*find_schedule_block.begin());
    sch_block.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  }
  VLOG(3) << "new_loop is : " << new_loop;
  CHECK(!used_target_loops.empty());
  Expr res;
  if (used_target_loops.size() == 1) {
    auto for_loop = used_target_loops[0].As<ir::For>();
    res           = For::Make(for_loop->loop_var,
                    for_loop->min,
                    for_loop->extent,
                    for_loop->for_type(),
                    for_loop->device_api,
                    new_loop,
                    for_loop->vectorize_info(),
                    for_loop->bind_info());
  } else {
    Expr outer_loop                = used_target_loops.front();
    Expr inner_loop                = used_target_loops.back();
    inner_loop.As<ir::For>()->body = Block::Make({new_loop});
    res                            = outer_loop;
  }
  VLOG(3) << "res is : " << res;
  std::vector<Expr> all_loops = this->GetLoops(block);
  CHECK(!all_loops.empty());
  helper_.Replace(all_loops[0], res);
}

}  // namespace ir
}  // namespace cinn
