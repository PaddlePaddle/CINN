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
#include <string>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace ir {

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

struct FindLoopsVisitor {
  FindLoopsVisitor(const Expr& block) : block_(block) {}

  std::vector<Expr> operator()(const Expr* expr) {
    CHECK(block_.As<ir::ScheduleBlockRealize>());
    visit_end = false;
    Visit(expr);
    return result;
  }

 private:
  void Visit(const Expr* expr) {
    if (visit_end || !expr->defined()) return;
    if (expr->As<ir::For>()) {
      father_loops.emplace_back(*expr);
      Visit(&(expr->As<ir::For>()->body));
      father_loops.pop_back();
    } else if (expr->As<ir::ScheduleBlockRealize>()) {
      if (!expr->As<ir::ScheduleBlockRealize>()->iter_values.empty() && (*expr == block_)) {
        result    = father_loops;
        visit_end = true;
        return;
      } else {
        Visit(&(expr->As<ir::ScheduleBlockRealize>()->schedule_block));
      }
    } else if (expr->As<ir::ScheduleBlock>()) {
      Visit(&(expr->As<ir::ScheduleBlock>()->body));
    } else if (expr->As<ir::Block>()) {
      for (auto& n : expr->As<ir::Block>()->stmts) Visit(&n);
    } else if (expr->As<ir::IfThenElse>()) {
      Visit(&(expr->As<ir::IfThenElse>()->true_case));
      Visit(&(expr->As<ir::IfThenElse>()->false_case));
    }
  }

  std::vector<Expr> father_loops{};
  std::vector<Expr> result{};
  bool visit_end{false};
  const Expr& block_;
};

/**
 * \brief Given a ScheduleBlockRealize node, return the Store tensor in its body.
 * @param block The given ScheduleBlockRealize node
 * @return The Store tensor in block
 */
Tensor GetTensor(const Expr& block);

struct FindBlocksVisitor {
  FindBlocksVisitor(const std::string& block_name = "") : block_name_(block_name) {}

  std::vector<Expr> operator()(const Expr* expr) {
    Visit(expr);
    return result;
  }

 private:
  void Visit(const Expr* expr) {
    if (!expr->defined()) return;
    if (!block_name_.empty() && !result.empty()) return;
    if (expr->As<ir::For>()) {
      Visit(&(expr->As<ir::For>()->body));
    } else if (expr->As<ir::ScheduleBlockRealize>()) {
      if (!expr->As<ir::ScheduleBlockRealize>()->iter_values.empty()) {
        if (block_name_.empty() || GetTensor(*expr)->name == block_name_) {
          result.emplace_back(*expr);
        }
      } else {
        Visit(&(expr->As<ir::ScheduleBlockRealize>()->schedule_block));
      }
    } else if (expr->As<ir::ScheduleBlock>()) {
      Visit(&(expr->As<ir::ScheduleBlock>()->body));
    } else if (expr->As<ir::Block>()) {
      for (auto& n : expr->As<ir::Block>()->stmts) Visit(&n);
    } else if (expr->As<ir::IfThenElse>()) {
      Visit(&(expr->As<ir::IfThenElse>()->true_case));
      Visit(&(expr->As<ir::IfThenElse>()->false_case));
    }
  }
  std::string block_name_;
  std::vector<Expr> result{};
};

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
 * \brief Given a ScheduleBlockRealize node, return the index-th Load tensor in its body.
 * @param block The given ScheduleBlockRealize node
 * @param index The index of Load tensor
 * @return The index-th Load tensor in block
 */
Tensor GetReadTensor(const Expr& block, int index);

/**
 * \brief Given a For node, return its extent as int.
 * @param loop The given For node
 * @return The extent of For node
 */
int GetLoopExtent(const Expr& loop);

/**
 * \brief Given a vector of Exors, return whether they contain a var with specific name.
 * @param exprs The given vector of Exprs
 * @param var_name The name of specific var
 * @return Whether there is a Var with the same name as var_name
 */
bool ContainVar(const std::vector<Expr>& exprs, const std::string& var_name);

/**
 * \brief Given a _LoweredFunc_, set its cuda_axis_info based on its func_body.
 * @param lowered_func A pointer to the given _LoweredFunc_
 */
void SetCudaAxisInfo(Expr* lowered_func);

/*!
 * \brief Check if a Expr node contains a ScheduleBlockRealize node.
 * \param container The container Expr node.
 * \param expr The node we want to find.
 * \return If the container contains the expr.
 */
bool Contains(const Expr& container, const Expr& expr);

/**
 * \brief Given a For loop, return the next For loop in its body.
 * @param for_loop The given For loop.
 * @return The next For loop.
 */
Expr GetNextForLoop(const Expr& for_loop);

/**
 * \brief Given two For loops, return all ir::IfThenElse nodes between them.
 * @param top The given top For loop.
 * @param bottom The given bottom For loop.
 * @return All ir::IfThenElse nodes between them.
 */
std::vector<Expr> GetIfThenElseInRange(const Expr& top, const Expr& bottom);

/**
 * Replace Vars in replaced to Exprs in candidates in source. Vars -> Exprs is one-to-one correspondence.
 * @param source The Expr we will implement the change.
 * @param replaced The Vars to be replaced.
 * @param candidates The Exprs to replace Vars in replaced.
 */
void ReplaceExpr(Expr* source, const std::vector<Var>& replaced, const std::vector<Expr>& candidates);

/**
 * Validate the factors param of Split. We will check if factors are validate and change -1 to positive integer.
 * @param factors The original factors.
 * @param total_extent The extent of the loop to be splitted.
 * @param return The valiated factors.
 */
std::vector<int> ValidateFactors(const std::vector<int>& factors, int total_extent);

void CHECKRfactorValidation(const Expr& rf_loop, int rf_axis);

/**
 * Return loops that contain the expr.
 * @param expr The expr.
 * @param root The root of the whole AST.
 * @param return Loops in AST that contain the expr.
 */
std::vector<Expr> GetLoopsOfExpr(const Expr& expr, const Expr& root);

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
                               int i);

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
                                            const Tensor& tensor);

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
                                                          const Expr& root);

/**
 * Return n-th access tensor in block
 * @param block The ScheduleBlockRealize.
 * @param index The index indicating which tensor we want to get.
 * @param is_write We want to get write tensor or read tensor.
 * @param return The n-th access tensor in block. Should be ir::Store(is_write) or ir::Load(!is_write).
 */
Expr GetNthAccessExpr(const Expr& block, int index, bool is_write);

/**
 * Make a tensor's cache tensor.
 * @param tensor The original tensor.
 * @param memory_type The memory type of the cache tensor.
 * @param return The tensor's cache tensor.
 */
Tensor MakeCacheTensor(const Tensor& tensor, const std::string& memory_type);

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
                    DeviceAPI device_api);

/**
 * Fidn cache tensor block's insertion point in the whole AST(root).
 * @param root The whole AST.
 * @param info The information of cache block.
 * @param is_write Are we inserting a write cache tensor or a read cache tensor.
 */
void FindInsertionPoint(Expr& root, CacheBlockInfo* info, bool is_write);

/**
 * \brief Given a vector of For loops, return a set of them.
 * @param loops The given vector of For loops.
 * @return A set containing all the For loops in loops.
 */
const std::set<Expr, CompExpr> CollectLoopsToSet(const std::vector<Expr>& loops);

/**
 * \brief Given a set of For loops, return the boundary among them.
 * @param loop_set The given set of For loops.
 * @return A pair of the boundary among For loops.(The top For and bottom For)
 */
std::pair<Expr, Expr> GetBoundaryOfReorderRange(const std::set<Expr, CompExpr>& loop_set);

/**
 * \brief Given two For loops, return all loops between them.
 * @param top The top For loop.
 * @param bottom The bottom For loop.
 * @return A vector containing all For loops between the boundary, stored in ascending order.
 */
std::vector<Expr> GetLoopsInRange(const Expr& top, const Expr& bottom);

/**
 * \brief Given params, construct a new loop.
 */
Expr ConstructNewLoopChain(const std::vector<Expr>& chain,
                           const std::vector<Expr>& ordered_loops,
                           const std::set<Expr, CompExpr>& loop_set,
                           std::vector<Expr>& if_nodes);

/*!
 * \brief Find producers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its producers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's producers(Load nodes) in root.
 */
std::vector<Expr> GetProducers(const Expr& block, const Expr& root);

/*!
 * \brief Find consumers of block in root.
 * \param block The ScheduleBlockRealize node we want to find its consumers.
 * \param root The root ScheduleBlockRealize node.
 * \return block's consumers(ScheduleBlockRealize nodes) in root.
 */
std::vector<Expr> GetConsumers(const Expr& block, const Expr& root);

/*!
 * \brief Check if the params of ComputeAt is validate.
 * \param block The block node we want to move in ComputeAt.
 * \param loop The for node we want to put the block under in ComputeAt.
 * \param root The root ScheduleBlockRealize node of block and loop.
 */
void CheckComputeAtValidation(const Expr& block, const Expr& loop, const Expr& root);

/*!
 * \brief Insert a new ScheduleBlockRealize in a loop's body(under its IfThenElse Node, if any)
 * \param for_loop The for loop whose body we want to modify
 * \param insertion The ScheduleBlockRealize we want to insert
 */
void InsertBlock(Expr& for_loop, const Expr& insertion);

/*!
 * \brief Make a union of two range. The detailed function is :
 * new_range.min = min(range1.min, range2.min)
 * new_range.extent = max(range1.min + range1.extent, range2.min + range2.extent) - new_range.min
 * Notice that the pair<Expr, Expr> indicates a range's min and extent.
 * \param range1 The first range
 * \param range2 The second range
 * \return The union of these two ranges
 */
std::pair<Expr, Expr> RangeUnion(const std::pair<Expr, Expr>& range1, const std::pair<Expr, Expr>& range2);

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
                                                            const Expr& root);

Expr CheckComputeInlineValidationAndGetStore(const Expr& schedule_block, const Expr& root);

}  // namespace ir
}  // namespace cinn
