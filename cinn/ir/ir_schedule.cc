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

// Self-defined operator to support std::set<Expr>
struct CompExpr {
  bool operator()(const Expr& left, const Expr& right) const {
    return utils::GetStreamCnt(left) < utils::GetStreamCnt(right);
  }
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
    CHECK_LT(i, (int)all_loops.size()) << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

void IRSchedule::MutateForType(Expr& loop, ForType for_type, int factor) {
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

void IRSchedule::Parallel(Expr& loop) { MutateForType(loop, ForType::Parallel); }

void IRSchedule::Vectorize(Expr& loop, int factor) {
  CHECK_GT(factor, 0) << "vectorize factor should be more than 0";
  MutateForType(loop, ForType::Vectorized, factor);
}

void IRSchedule::Unroll(Expr& loop) { MutateForType(loop, ForType::Unrolled); }

void IRSchedule::Bind(Expr& loop, const std::string& thread_axis) {
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
  if (index.is_constant())
    return std::make_pair(index, index);
  else if (i >= (int)iter_vars.size()) {
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
  for (int i = 0; i < (int)tensor_indices.size(); i++) {
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
    auto find_store = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Store>(); });
    CHECK_LT(index, (int)find_store.size());
    std::vector<Expr> store_vec(find_store.begin(), find_store.end());
    Expr store_index = store_vec[index];
    CHECK(store_index.As<ir::Store>());
    CHECK(store_index.As<ir::Store>()->tensor.as_tensor());
    return store_index;
  } else {
    auto find_load = ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Load>(); });
    CHECK_LT(index, (int)find_load.size());
    std::vector<Expr> load_vec(find_load.begin(), find_load.end());
    Expr load_index = load_vec[index];
    CHECK(load_index.As<ir::Load>());
    CHECK(load_index.As<ir::Load>()->tensor.as_tensor());
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
    Var var(Expr(0), dim, "v" + std::to_string(block_vars.size()));
    block_vars.push_back(var);
  }
  auto body                  = new_tensor->tensor_store_expanded_body();
  std::vector<Var> axis_vars = common::GenDefaultAxis(new_tensor->domain.size());
  axis_vars.insert(axis_vars.end(), new_tensor->reduce_axis.begin(), new_tensor->reduce_axis.end());
  for (int i = 0; i < axis_vars.size(); i++) {
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
  for (int i = 0; i < (int)info->loc_block.As<Block>()->stmts.size(); i++) {
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
    return x->As<ir::ScheduleBlockRealize>() &&
           x->As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name == info.read_tensor->name;
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
  CHECK((src_sref.As<ir::For>() && tgt_stmt.As<ir::For>()) || (src_sref.As<ir::Block>() && tgt_stmt.As<ir::Block>()));
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
  std::string block_tensor = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
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

  void Visit(const ir::Block* expr, Expr* op) override {
    if (expr->stmts.size() > 1U) {
      int block_index = -1;
      for (int i = 0; i < expr->stmts.size(); i++) {
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
        for (int i = 0; i < expr->stmts.size(); i++) {
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

void IRSchedule::SetBuffer(const Expr& block, const std::string& memory_type) const {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::CollectIRNodesWithoutTensor(block, [&](const Expr* x) { return x->As<ir::Store>(); });
  CHECK(!find_tensor.empty()) << "Didn't find Store in block!";
  CHECK_EQ(find_tensor.size(), 1U) << "One block should only have one Store node!(except for root block)";
  Tensor tensor = (*find_tensor.begin()).As<ir::Store>()->tensor.as_tensor_ref();
  tensor->WithBuffer(memory_type, "_" + tensor->name + "_temp_buffer");
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
                                                            const std::vector<Expr>& consumers) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  std::string block_tensor = block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
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
        for (int i = 0; i < indices.size(); i++) {
          if (i >= required_buffer_range.size())
            required_buffer_range.push_back(std::make_pair(indices[i], Expr(1)));
          else
            required_buffer_range[i] = RangeUnion(required_buffer_range[i], std::make_pair(indices[i], Expr(1)));
        }
      } else {
        for (int i = 0; i < indices.size(); i++) {
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
          if (common::AutoSimplify(indice_min) == common::AutoSimplify(indice_max))
            if (common::is_zero(mod_extent))
              indice_extent = Expr(1);
            else
              indice_extent = mod_extent;
          else
            indice_extent = common::AutoSimplify(common::AutoSimplify(indice_max) - common::AutoSimplify(indice_min));
          if (indice_extent.is_constant() && indice_extent.get_constant() < 0) {
            indice_min    = common::AutoSimplify(indice_max);
            indice_extent = Expr(-indice_extent.get_constant());
          }
          if (i >= required_buffer_range.size())
            required_buffer_range.push_back(std::make_pair(indice_min, indice_extent));
          else
            required_buffer_range[i] = RangeUnion(required_buffer_range[i], std::make_pair(indice_min, indice_extent));
        }
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
  auto iter_doms = CalculateRequiredRegions(block, loop, consumers);
  for (auto& i : iter_doms) LOG(INFO) << "CalculateRequiredRegions is : " << i.first << " to " << i.second;
  reconstructor.MakeNewLoop(iter_doms);
  helper_.Replace(reconstructor.source_expr, reconstructor.target_expr);
  helper_.Replace(reconstructor.loop_, reconstructor.new_loop_);
  return;
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
      for (auto& it_for : loop_nodes) {
        if (Contains(it_for, block)) result.push_back(it_for);
      }
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
    auto block_nodes = ir::CollectIRNodes(it_expr, [&](const Expr* x) {
      return x->As<ir::ScheduleBlockRealize>() && !x->As<ir::ScheduleBlockRealize>()->iter_values.empty();
    });
    for (auto& it_block : block_nodes) result.push_back(it_block);
  }
  CHECK(!result.empty());
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
