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

#include <gtest/gtest.h>

#include "cinn/common/context.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

TEST(AnalyzeIr, AnalyzeScheduleBlockReadWriteBuffer) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  ir::Tensor B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  poly::StageMap stages = CreateStages({A, B});
  std::vector<ir::LoweredFunc> funcs =
      cinn::lang::LowerVec("test_vectorize", stages, {A, B}, {}, {}, nullptr, target, true);

  ASSERT_FALSE(funcs.empty());
  ir::Expr ast_expr = func[0]->body;
  // std::vector<Expr> vec_ast{ast_expr};
  // ir::ModuleExpr mod_expr(vec_ast);
  // ir::IRSchedule ir_sch(mod_expr);

  std::string origin = utils::GetStreamCnt(func[0]);
  std::cout << origin << std::endl;
}

}  // namespace auto_schedule
}  // namespace cinn
