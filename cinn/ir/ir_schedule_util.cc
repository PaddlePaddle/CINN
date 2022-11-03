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

#include "cinn/ir/ir_schedule_util.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/lang/compute.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace ir {

Tensor GetTensor(const Expr& block) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::CollectIRNodesWithoutTensor(
      block, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  CHECK_EQ(find_tensor.size(), 1U) << "One block should only have one Store node!(except for root block)";
  CHECK((*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor());
  Tensor tensor = (*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor_ref();
  return tensor;
}

Tensor GetReadTensor(const Expr& block, int index) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::CollectIRNodesWithoutTensor(
      block, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
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

bool Contains(const Expr& container, const Expr& expr) {
  auto find_expr = ir::CollectIRNodesWithoutTensor(
      container, [&](const Expr* x) { return (x->node_type() == expr.node_type() && *x == expr); }, true);
  return (!find_expr.empty());
}

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

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis) {
  auto* rf_for = rf_loop.As<ir::For>();
  CHECK(rf_for) << "Expr param of Rfactor must be For node! Please check.";
  // check the rf_loop only has one schedule block
  auto block_nodes = ir::CollectIRNodesWithoutTensor(
      rf_loop, [&](const Expr* x) { return x->As<ScheduleBlockRealize>(); }, true);
  CHECK_EQ(block_nodes.size(), 1U) << "Rfactor Loop should only have one schedule block";
  auto find_store = ir::CollectIRNodesWithoutTensor(
      rf_loop, [&](const Expr* x) { return x->As<Store>(); }, true);
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

std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root) {
  auto loop_nodes =
      ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) { return x->As<ir::For>() && Contains(*x, expr); });
  std::vector<Expr> result(loop_nodes.begin(), loop_nodes.end());
  if (result.empty()) LOG(FATAL) << "Didn't find expr's : \n" << expr << "\n loops in root : \n" << root;
  std::sort(result.begin(), result.end(), [&](Expr i, Expr j) {
    return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
  });
  return result;
}

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

Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type) {
  auto cache_tensor = lang::Compute(
      tensor->shape,
      [=](const std::vector<Expr>& dims) { return tensor(dims); },
      tensor->name + "_" + memory_type + "_temp_buffer");
  cache_tensor->WithBuffer(memory_type);
  return cache_tensor;
}

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
    Var loop_var(common::UniqName("cache_ax" + std::to_string(loop_vars.size())));
    // Var loop_var("ax" + std::to_string(loop_vars.size()));
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

std::vector<Expr> GetProducers(const Expr& block, const Expr& root) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(root.As<ir::ScheduleBlockRealize>());
  std::vector<Expr> producers;
  auto compute_body = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
  auto find_load    = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Load>(); });
  for (auto& i : find_load) producers.emplace_back(i);
  return producers;
}

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

void CheckComputeAtValidation(const Expr& block, const Expr& loop, const Expr& root) {
  auto find_block = ir::CollectIRNodesWithoutTensor(
      root, [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>() && *x == block; }, true);
  CHECK(!find_block.empty()) << "Didn't find block in root!";

  auto find_loop = ir::CollectIRNodesWithoutTensor(
      root, [&](const Expr* x) { return x->As<ir::For>() && *x == loop; }, true);
  CHECK(!find_loop.empty()) << "Didn't find loop in root!";

  auto find_block_in_loop = ir::CollectIRNodesWithoutTensor(
      loop, [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>() && *x == block; }, true);
  CHECK(find_block_in_loop.empty()) << "loop should not be block's ancestor!";
}

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

std::pair<Expr, Expr> RangeUnion(const std::pair<Expr, Expr>& range1, const std::pair<Expr, Expr>& range2) {
  Expr new_min    = common::AutoSimplify(Min::Make(range1.first, range2.first));
  Expr new_extent = common::AutoSimplify(
      common::AutoSimplify(Max::Make(range1.first + range1.second, range2.first + range2.second)) - new_min);
  return std::make_pair(new_min, new_extent);
}

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

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block, const Expr& root) {
  CHECK(schedule_block.As<ir::ScheduleBlockRealize>());
  auto compute_body = schedule_block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->body;
  // 1. Check the schedule block to be inlined is not a reduce tensor.
  auto find_store = ir::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  CHECK_EQ(find_store.size(), 1U);
  Expr tensor = (*find_store.begin()).As<ir::Store>()->tensor;
  CHECK(!tensor.as_tensor_ref()->is_reduce_tensor());
  // 2. Check this schedule block is the only writer of the tensor.
  find_store = ir::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) {
        return x->As<ir::Store>() && (x->As<ir::Store>()->tensor).as_tensor_ref()->name == tensor.as_tensor_ref()->name;
      },
      true);
  CHECK_EQ(find_store.size(), 1U);
  // 3. Check there is no overlap between the buffers the schedule block reads and writes.
  auto find_load = ir::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Load>() && x->As<ir::Load>()->tensor == tensor; });
  CHECK(find_load.empty());
  return (*find_store.begin());
}

bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name) {
  for (auto& expr : exprs) {
    auto find_expr = ir::CollectIRNodesWithoutTensor(
        expr, [&](const Expr* x) { return x->As<_Var_>() && x->As<_Var_>()->name == var_name; }, true);
    if (!find_expr.empty()) return true;
  }
  return false;
}

}  // namespace ir
}  // namespace cinn