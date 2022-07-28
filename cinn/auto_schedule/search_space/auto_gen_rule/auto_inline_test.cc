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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/cinn.h"
#include "cinn/ir/function_base.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

TEST(AutoInline, SingleLoopInline) {
  srand(0);
  Context::Global().ResetNameId();
  Target target = common::DefaultHostTarget();

  Expr M(32);

  Placeholder<float> A("A", {M});
  ir::Tensor B = Compute(
      {M}, [&](Var i) { return A(i) * ir::Expr(2.f); }, "B");
  ir::Tensor C = Compute(
      {M}, [&](Var i) { return B(i) + ir::Expr(1.f); }, "C");

  poly::StageMap stages = CreateStages({A, B, C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestAutoInline_SingleLoopInline", stages, {A, C}, {}, {}, nullptr, target, true);

  VLOG(6) << "Expr after lowering:";
  VLOG(6) << funcs[0]->body;

  /*
   * We have to use ComputeAt to put two Tensor loops together to create IR
   * test case for AutoInline.
   */
  ir::IRSchedule ir_sch(ir::ModuleExpr(std::vector<ir::Expr>{funcs[0]->body}));
  ir::Expr block_b            = ir_sch.GetBlock("B");
  std::vector<ir::Expr> loops = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[0]);

  ir::ModuleExpr mod_expr_before_inline = ir_sch.GetModule();
  VLOG(6) << "Expr after ComputeAt:";
  VLOG(6) << mod_expr_before_inline.GetExprs()[0];

  AutoInline auto_inline(target, {"C"});
  EXPECT_EQ(auto_inline.Init(mod_expr_before_inline), RuleApplyType::kApply);
  EXPECT_EQ(auto_inline.NumberApplicable(), 1);

  ir::ModuleExpr mod_expr_after_inline = auto_inline.ApplyRandomly();
  std::vector<ir::Expr> exprs          = mod_expr_after_inline.GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  std::stringstream ss;
  ss << exprs[0];

  std::string expr_str = ss.str();
  VLOG(6) << "After AutoInline:";
  VLOG(6) << expr_str;

  std::string target_str = R"ROC({
  ScheduleBlock(root)
  {
    {
      for (i, 0, 32)
      {
        ScheduleBlock(C)
        {
          i0 = axis.bind(i)
          read_buffers(_A[i0(0:32)])
          write_buffers(_C[i0(0:32)])
          C[i0] = (1 + (2 * A[i0]))
        }
      }
    }
  }
})ROC";
  EXPECT_EQ(expr_str, target_str);

  // Cannot inline above expr again
  EXPECT_EQ(auto_inline.Init(mod_expr_after_inline), RuleApplyType::kCannotApply);
}

}  // namespace auto_schedule
}  // namespace cinn
