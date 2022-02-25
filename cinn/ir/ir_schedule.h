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
#include <vector>

#include "cinn/ir/ir_base.h"

namespace cinn {
namespace ir {

/**
 * A struct representing a module that contains Expr. This struct is only used in Schedule process.
 */
class ModuleExpr {
 public:
  ModuleExpr() = default;
  explicit ModuleExpr(const std::vector<Expr>& init_exprs) : init_exprs_(init_exprs) {}

  //! Get all the Expr in this ModuleExpr.
  std::vector<Expr> GetExprs() const { return init_exprs_; }

 private:
  //! Exprs stored in ModuleExpr. Each one is an AST, representing a computation kernel.
  std::vector<Expr> init_exprs_;
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
  void Replace(Expr& src_sref, const Expr& tgt_stmt);

  //! Get all the loops in AST stored in ModuleExpr.
  std::vector<Expr> GetLoops() const;

  //! Get the ModuleExpr stored in ScheduleHelper.
  ModuleExpr GetModule() const { return module_expr_; }

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

  //! Get all the loops in IR(Expr)/AST stored in ModuleExpr.
  std::vector<Expr> GetLoops() const { return helper_.GetLoops(); }

  /**
   * \brief Split a for loop into multiple loops, based on the factors.
   * @param loop The loop to be splited.
   * @param factors The factors we used to split the loop.
   * @return The splited loops.
   */
  std::vector<Expr> Split(Expr& loop, const std::vector<int>& factors);

  /**
   * \brief Fuse for loops and return the fused loop.
   * @param loops All the loops to be fused, stored in ascending order.
   * @return The fused loop.
   */
  Expr Fuse(std::vector<Expr>& loops);

  //! Get the ModuleExpr stored in ScheduleHelper.
  ModuleExpr GetModule() { return helper_.GetModule(); }

 private:
  ScheduleHelper helper_;
};

}  // namespace ir
}  // namespace cinn
