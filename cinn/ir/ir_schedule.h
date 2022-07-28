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

/**
 * A struct representing a module that contains Expr. This struct is only used in Schedule process.
 */
class ModuleExpr {
 public:
  ModuleExpr()                           = default;
  ModuleExpr(const ModuleExpr& mod_expr) = default;
  ModuleExpr(ModuleExpr&& mod_expr)      = default;

  ModuleExpr& operator=(const ModuleExpr& mod_expr) = default;

  explicit ModuleExpr(const std::vector<Expr>& exprs) : exprs_(exprs) {}
  explicit ModuleExpr(std::vector<Expr>&& exprs) : exprs_(std::move(exprs)) {}

  //! Get all the Expr in this ModuleExpr.
  std::vector<Expr> GetExprs() { return exprs_; }

  std::vector<Expr> GetExprs() const { return exprs_; }

  void SetExprs(const std::vector<Expr>& exprs) { exprs_ = exprs; }

 private:
  //! Exprs stored in ModuleExpr. Each one is an AST, representing a computation kernel.
  std::vector<Expr> exprs_;
};

/**
 * A struct helps to implment Schedule primitives.
 */
class ScheduleHelper {
 public:
  ScheduleHelper() = default;
  explicit ScheduleHelper(const ModuleExpr& module_expr, bool debug_flag = false)
      : module_expr_(module_expr), debug_flag_(debug_flag) {}

  //! Set the debug flag.
  void SetDebugFlag(bool debug_flag) { debug_flag_ = debug_flag; }

  /**
   * In IR stored in ModuleExpr, replace a specified Expr node src_sref to another Expr node tgt_stmt.
   * @param src_sref The IR node to be replaced.
   * @param tgt_stmt The IR node to replace the original one.
   */
  void Replace(const Expr& src_sref, const Expr& tgt_stmt);

  /**
   * \brief Get all the loops of specific Block stored in ModuleExpr.
   * @param block The block we find loop in.
   * @return Loops of the block.
   */
  std::vector<Expr> GetLoops(const Expr& block) const;

  /**
   * \brief Get all the loops of specific Block stored in ModuleExpr.
   * @param block_name Name of the block.
   * @return Loops of the block.
   */
  std::vector<Expr> GetLoops(const std::string& block_name) const;

  //! Get all blocks stored in this ModuleExpr.
  std::vector<Expr> GetAllBlocks() const;

  //! Get a block with the specific name.
  Expr GetBlock(const std::string& block_name) const;

  //! Get the ModuleExpr stored in ScheduleHelper.
  ModuleExpr GetModule() const { return module_expr_; }

  void SetExprs(const std::vector<Expr>& exprs) { module_expr_.SetExprs(exprs); }

 private:
  ModuleExpr module_expr_;
  bool debug_flag_{false};
};

/**
 * A struct containing all the schedule primitives. Each shedule primitive is a member function of IRSchedule.
 * Schedule primitves are implmented by ScheduleHelper manipulating the AST - IR(Expr).
 */
class IRSchedule {
 public:
  IRSchedule() = default;
  explicit IRSchedule(const ModuleExpr& modexpr, bool debug_flag = false);

  void SetExprs(const std::vector<Expr>& exprs) { helper_.SetExprs(exprs); }

  /**
   * \brief Get all the loops of specific Block stored in ModuleExpr.
   * @param block The block we find loop in.
   * @return Loops of the block.
   */
  std::vector<Expr> GetLoops(const Expr& block) const { return helper_.GetLoops(block); }

  /**
   * \brief Get all the loops of specific Block stored in ModuleExpr.
   * @param block_name Name of the block.
   * @return Loops of the block.
   */
  std::vector<Expr> GetLoops(const std::string& block_name) const { return helper_.GetLoops(block_name); }

  //! Get all blocks stored in this ModuleExpr.
  std::vector<Expr> GetAllBlocks() const { return helper_.GetAllBlocks(); }

  //! Get a block with the specific name.
  Expr GetBlock(const std::string& block_name) const { return helper_.GetBlock(block_name); }

  /**
   * \brief Split a for loop into multiple loops, based on the factors.
   * @param loop The loop to be splited.
   * @param factors The factors we used to split the loop.
   * @return The splited loops.
   */
  std::vector<Expr> Split(const Expr& loop, const std::vector<int>& factors);

  /**
   * \brief Split a for loop into multiple loops, based on the factors.
   * @param block_name Name of the block we want to modify.
   * @param loop_index Index of the loop to be splited.
   * @param factors The factors we used to split the loop.
   * @return The splited loops.
   */
  std::vector<Expr> Split(const std::string& block_name, int loop_index, const std::vector<int>& factors);

  /**
   * \brief Fuse for loops and return the fused loop.
   * @param loops All the loops to be fused, stored in ascending order.
   * @return The fused loop.
   */
  Expr Fuse(const std::vector<Expr>& loops);

  /**
   * \brief Fuse for loops and return the fused loop.
   * @param block_name Name of the block we want to modify.
   * @param loops_index Indices of the loops to be fused, stored in ascending order.
   * @return The fused loop.
   */
  Expr Fuse(const std::string& block_name, const std::vector<int>& loops_index);

  /**
   * \brief Fuse for loops and return the fused loop.
   * @param block The block we want to modify.
   * @param loops_index Indices of the loops to be fused, stored in ascending order.
   * @return The fused loop.
   */
  Expr Fuse(const Expr& block, const std::vector<int>& loops_index);

  /**
   * \brief Move a block's location under a loop.
   * @param block The block we want to move its computation location.
   * @param loop The loop we will move the block to.
   */
  void ComputeAt(const Expr& block, const Expr& loop);

  /**
   * \brief Move a block's location under a loop without considering their dependency.
   * @param block The block we want to move its computation location.
   * @param loop The loop we will move the block to.
   */
  void SimpleComputeAt(const Expr& block, const Expr& loop);

  /**
   * \brief Find an expr's root ScheduleBlockRealize node
   * @param expr The expr node.
   * @return Its root ScheduleBlockRealize node.
   */
  Expr GetRootBlock(const Expr& expr) const;

  /**
   * \brief Find a buffer that is being read, and create its cache.
   * @param block Block that reads the buffer.
   * @param read_buffer_index Index of the buffer being read in block.
   * @param memory_type String that indicates the buffer's storage scope.
   * @return The buffer's cache.
   */
  Expr CacheRead(const Expr& block, int read_buffer_index, const std::string& memory_type);

  /**
   * \brief Find a buffer that is being written, and create its cache.
   * @param block Block that writes the buffer.
   * @param write_buffer_index Index of the buffer being written in block.
   * @param memory_type String that indicates the buffer's storage scope.
   * @return The buffer's cache.
   */
  Expr CacheWrite(const Expr& block, int write_buffer_index, const std::string& memory_type);

  /**
   * \brief Add SyncThreads statements in AST.
   * @param ir_node The insertion point in AST.
   * @param after_node Whether to insert the statement after the insertion point. When it is True, we will insert the
   * SyncThreads statement after the insertion IR. When it is False, we will insert the SyncThreads statement before the
   * insertion IR.
   */
  void SyncThreads(const Expr& ir_node, bool after_node = true);

  /*!
   * \brief Set a tensor's buffer type(memory_type)
   * \param block The ScheduleBlockRealize corresponding to an unique tensor.
   * \param memory_type The memory type we want to set. Should be "local", "shared" or "global".
   */
  void SetBuffer(Expr& block, const std::string& memory_type, bool fixed = false);

  /**
   * \brief Reorder the loops in the order of vector.
   * @param loops The loops to be reordered.
   */
  void Reorder(const std::vector<Expr>& loops);

  /**
   * \brief Reorder the loops in the order of vector elements.
   * @param block_name Name of the block we want to modify.
   * @param loops_index Indices of loops to be reordered.
   */
  void Reorder(const std::string& block_name, const std::vector<int>& loops_index);

  /**
   * \brief Reorder the loops in the order of vector elements.
   * @param block The block we want to modify.
   * @param loops_index Indices of loops to be reordered.
   */
  void Reorder(const Expr& block, const std::vector<int>& loops_index);

  /**
   * Get the device api of this IRSchedule.
   * @param return The device api of this IRSchedule.
   */
  DeviceAPI GetDeviceAPI() const;

  /**
   * \brief Change forloop to be parallelized/vectorized/unrolled.
   * @param loop The forloop to parallel/vectorize/unroll.
   * @param for_type the target forloop type.
   */
  void MutateForType(const Expr& loop, ForType for_type, int factor = -1);

  /**
   * \brief Parallelize the given loop.
   * @param loop the loop to parallel.
   */
  void Parallel(const Expr& loop);

  /**
   * \brief Vectorize the given loop.
   * @param loop the loop to vectorize.
   * @param factor the vectorized factor.
   */
  void Vectorize(const Expr& loop, int factor);

  /**
   * \brief Unroll the given loop.
   * @param loop the loop to unroll.
   */
  void Unroll(const Expr& loop);

  /**
   * \brief Mark an schedule block as inlined.
   * @param schedule_block the schedule block to be inlined.
   */
  void ComputeInline(const Expr& schedule_block);

  /**
   * \brief Bind the loop to the given thread axis.
   * @param loop the loop to Bind.
   * @param thread_axis the name of the thread axis to be bound to the loop.
   */
  void Bind(const Expr& loop, const std::string& thread_axis);

  //! Copy another block's schedule transform.
  void CopyTransformAndLoopInfo(const Expr& block, const Expr& block_target);

  void CopyTransformAndLoopInfo(const std::string& block_name, const std::string& block_target_name);

  /**
   * \brief Factorize the reduction block by the given loop. The block will be split into two blocks: rfactor block and
   * final write-back block.
   * @param rf_loop the reduce loop to do rfactor transformation.
   * @param rf_axis the axis where the new generated loop is placed in the rfactor block.
   * @return The new created rfactor tensor.
   *
   * For example, input the block:
   * \code
   * for (i, 0, 10)      // serial loop
   *   B_init[i] = 0
   *   for (j, 0, 20)    // reduce loop
   *      for (k, 0, 30) // reduce loop
   *         B[i] = B[i] + A[i, j, k]
   * \endcode
   *
   * If the rfactor loop is k and rf_axis is 0, the rfactor transformation is divided into 2 steps:
   * 1. get the rfactor block where the reduce loop k is transformed to the serial loop with no accumalation and a new
   * rfactor tensor is created. The axis k will be placed in the rf_axis of the new rf_tensor. The rf_block is as
   * follows:
   * \code
   * for (rf_k, 0, 30)      // rfactor loop k is transformed to the serial loop.
   *   for (i, 0, 10)       // serial loop for (j, 0, 20) // reduce loop
   *     rf_B_init[rf_k, i] = 0
   *     for (j, 0, 20)     // reduce loop
   *       rf_B[rf_k, i] = rf_B[rf_k, i] + A[i, j, rf_k]
   * \endcode
   * 2. do reduction of the rfactor loop k to get the final result block:
   * \code
   *   for (i, 0, 10)    // serial loop
   *      B_init[i] = 0
   *      for (k, 0, 30)
   *        B[i] = B[i] + rf_B[k, i]
   * \endcode
   */
  Expr Rfactor(const Expr& rf_loop, int rf_axis);

  //! Get the ModuleExpr stored in ScheduleHelper.
  ModuleExpr GetModule() const { return helper_.GetModule(); }

  void MergeExprs();

 private:
  ScheduleHelper helper_;
};

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
  void operator()(Expr* expr);

 private:
  void Visit(const ir::Block* expr, Expr* op) override;

 protected:
  //! Check if indices are validate. If so, set idx_vars_ properly.
  bool UpdateAndCheckIndexVars(const std::vector<Expr>& indices, int expected_ndim);

  void SetIndexSubstitution(const std::vector<Expr>& indices);

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

  bool BodyPatternAllowInline();

 private:
  void Visit(const ir::Load* expr, Expr* op) override;

  //! Replace the 'Load' node on the tensor to 'Load' node of its producers.
  Expr ReplaceInlinedTensor(Expr* load);
};

// The struct used to remove the original block in ComputeAt.
class LeafBlockRemovalPlan : public ir::IRMutator<> {
 public:
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

}  // namespace ir
}  // namespace cinn
