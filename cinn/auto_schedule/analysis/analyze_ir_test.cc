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

#include "cinn/auto_schedule/analysis/analyze_ir.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <vector>

#include "cinn/common/context.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

TEST(AnalyzeIr, AnalyzeScheduleBlockReadWriteBuffer) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  poly::StageMap stages = poly::CreateStages({A, B});
  std::vector<ir::LoweredFunc> funcs =
      cinn::lang::LowerVec("test_vectorize", stages, {A, B}, {}, {}, nullptr, target, true);

  ASSERT_FALSE(funcs.empty());
  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Analyzing for Expr:";
  VLOG(6) << ast_expr;

  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  std::vector<ir::Expr> all_block_realizes = ir_sch.GetAllBlocks();
  ASSERT_EQ(all_block_realizes.size(), 1UL);

  ir::ScheduleBlockRealize* sche_block_realize = all_block_realizes[0].As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sche_block                = sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
  AnalyzeScheduleBlockReadWriteBuffer(sche_block);

  /*
   * the sche_block_realize will be:
   * ScheduleBlock(B)
   * {
   *   i0, i1 = axis.bind(i, j)
   *   read_buffers(_A[])
   *   write_buffers(_B[])
   *   B[i0, i1] = A[i0, i1]
   * }
   */
  ASSERT_EQ(sche_block->read_buffers.size(), 1UL);

  ASSERT_EQ(sche_block->write_buffers.size(), 1UL);
}

}  // namespace auto_schedule
}  // namespace cinn
