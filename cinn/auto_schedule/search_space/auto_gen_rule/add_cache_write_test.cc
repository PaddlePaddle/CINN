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

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_write.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class TestAddCacheWriteWith2DMatmul : public TestAutoGenRuleBase {
 public:
  void SetUp() override {
    srand(0);
    Context::Global().ResetNameId();
  }

  std::vector<ir::LoweredFunc> GenLoweredFuncs() override {
    CHECK_EQ(input_shapes_.size(), 2);
    CHECK_EQ(output_shapes_.size(), 1);
    const int M = input_shapes_[0][0];
    const int K = input_shapes_[0][1];
    const int N = input_shapes_[1][1];
    return Lower2DMatmul(M, K, N);
  }

  void CheckPrecision(const ir::Module& ir_module) override {
    // Compile to machine code
    backend_compier_->Build(ir_module);
    auto test_func_ptr = reinterpret_cast<void (*)(void**, int32_t)>(backend_compier_->Lookup(func_name_));
    // check result in precision
    CheckResult(test_func_ptr, expected_func_matmul, {"A", "B"}, {"C"}, input_shapes_, output_shapes_, target_);
  }
};

TEST_F(TestAddCacheWriteWith2DMatmul, Init) {
  // matmul case
  ir::IRSchedule ir_schedule_matmul = Initialize("matmul_apply_add_cache_write", {{32, 32}, {32, 32}}, {{32, 32}});
  std::vector<ir::Expr> func_bodys  = ir_schedule_matmul.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  AddCacheWrite add_cache_write(target_);
  auto apply_type = add_cache_write.Init(&ir_schedule_matmul);
#ifdef CINN_WITH_CUDA
  EXPECT_EQ(apply_type, RuleApplyType::kApplyAndSkipAllRules);
#else
  EXPECT_EQ(apply_type, RuleApplyType::kApplyAndSkipThisRule);
#endif
  EXPECT_EQ(add_cache_write.NumberApplicable(), 1);
  add_cache_write.ApplyRandomly();
  VLOG(6) << "Matmul Expr after AddCacheRead: " << func_bodys[0];

  // add case
  Placeholder<float> D("D", {64, 64});
  Placeholder<float> E("E", {64, 64});
  ir::Tensor F = Compute(
      {ir::Expr(64), ir::Expr(64)}, [&](Var i, Var j) { return D(i, j) + E(i, j); }, "F");
  std::vector<ir::LoweredFunc> funcs_add =
      lang::LowerVec("TestAddCacheRead_InitFalse", CreateStages({F}), {F}, {}, {}, nullptr, target_, true);

  ir::IRSchedule ir_schedule_add(ir::ModuleExpr({funcs_add[0]->body}));
  VLOG(6) << "Mat Add Expr before AddCacheRead:\n" << ir_schedule_add.GetModule().GetExprs();
  AddCacheWrite add_cache_write2(target_);
  EXPECT_EQ(add_cache_write2.Init(&ir_schedule_add), RuleApplyType::kCannotApply);
  EXPECT_EQ(add_cache_write2.NumberApplicable(), 0);
}

TEST_F(TestAddCacheWriteWith2DMatmul, BasicApplyOnMatmul) {
  ir::IRSchedule ir_schedule       = Initialize("matmul_apply_add_cache_write", {{32, 32}, {32, 32}}, {{32, 32}});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply AddCacheWrite
  AddCacheWrite add_cache_write(target_);
  add_cache_write.Init(&ir_schedule);
  ASSERT_EQ(add_cache_write.NumberApplicable(), 1);
  add_cache_write.ApplyRandomly();
  VLOG(6) << "Matmul Expr after AddCacheWrite: " << func_bodys[0];

  // build ir::Module and debug source code
  auto build_module = BuildIRModule(func_bodys);
  auto source_code  = GenSourceCode(build_module);
  VLOG(6) << "scheduled source code:\n" << source_code;
  // execute and check precision
  CheckPrecision(build_module);
}

TEST_F(TestAddCacheWriteWith2DMatmul, ApplyOnBlock) {
  ir::IRSchedule ir_schedule       = Initialize("matmul_apply_add_cache_write", {{32, 32}, {32, 32}}, {{32, 32}});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];
  SearchState state(ir_schedule, 0, {});

  // Apply AddCacheWrite
  AddCacheWrite add_cache_write(target_);
  EXPECT_EQ(add_cache_write.AnalyseApplyType(state, "C"), RuleApplyType::kApplyAndSkipAllRules);
  auto new_states             = add_cache_write.ApplyOnBlock(state, "C");
  std::vector<ir::Expr> exprs = new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Matmul Expr after AddCacheWrite: " << exprs[0];

  // build ir::Module and debug source code
  auto build_module = BuildIRModule(func_bodys);
  auto source_code  = GenSourceCode(build_module);
  VLOG(6) << "scheduled source code:\n" << source_code;
  // execute and check precision
  CheckPrecision(build_module);
}

TEST_F(TestAddCacheWriteWith2DMatmul, ApplyOnMatmulWithTiling) {
  ir::IRSchedule ir_schedule       = Initialize("matmul_apply_add_cache_write", {{32, 32}, {32, 32}}, {{32, 32}});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply MultiLevelTiling before AddCacheWrite
  MultiLevelTiling multi_level_tiling(target_);
  multi_level_tiling.Init(&ir_schedule);
  ASSERT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();
  VLOG(6) << "Expr after MultiLevelTiling: " << func_bodys[0];

  // Apply AddCacheWrite
  AddCacheWrite add_cache_write(target_);
  add_cache_write.Init(&ir_schedule);
  ASSERT_EQ(add_cache_write.NumberApplicable(), 1);
  add_cache_write.ApplyRandomly();
  VLOG(6) << "Expr after AddCacheWrite: " << func_bodys[0];

  // build ir::Module and debug source code
  auto build_module = BuildIRModule(func_bodys);
  auto source_code  = GenSourceCode(build_module);
  VLOG(6) << "scheduled source code:\n" << source_code;
  // execute and check precision
  CheckPrecision(build_module);
}

}  // namespace auto_schedule
}  // namespace cinn
