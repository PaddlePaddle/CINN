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

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_read.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/auto_schedule/tests/test_op_builder.h"
#include "cinn/cinn.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class TestAddCacheRead : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names  = {"X", "Y"};
  std::vector<std::string> default_output_names = {"temp_matmul_out"};
};

TEST_F(TestAddCacheRead, Init) {
  Initialize(common::DefaultNVGPUTarget());
  // matmul case
  ir::IRSchedule ir_schedule_matmul = MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})());
  std::vector<ir::Expr> func_bodys  = ir_schedule_matmul.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  AddCacheRead add_cache_read(target_);
  ASSERT_EQ(add_cache_read.Init(&ir_schedule_matmul), RuleApplyType::kApplyAndPruneOtherRules);
  ASSERT_EQ(add_cache_read.NumberApplicable(), 1);
  add_cache_read.ApplyRandomly();
  VLOG(6) << "Matmul Expr after AddCacheRead: " << func_bodys[0];

  // add case
  ir::IRSchedule ir_schedule_add = MakeIRSchedule(AddOpBuilder({64, 64}, {64, 64})());
  VLOG(6) << "Mat Add Expr before AddCacheRead:\n" << ir_schedule_add.GetModule().GetExprs();
  AddCacheRead add_cache_read2(target_);
  EXPECT_EQ(add_cache_read2.Init(&ir_schedule_add), RuleApplyType::kCannotApply);
  EXPECT_EQ(add_cache_read2.NumberApplicable(), 0);
}

TEST_F(TestAddCacheRead, BasicApplyOnMatmul) {
  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})());
  SearchState state(ir_schedule, 0, {});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];
  // Apply AddCacheRead.
  AddCacheRead add_cache_read(target_);
  add_cache_read.Init(&ir_schedule);
  ASSERT_EQ(add_cache_read.NumberApplicable(), 1);
  add_cache_read.ApplyRandomly();
  VLOG(6) << "Matmul Expr after AddCacheRead: " << func_bodys[0];
  // build ir::Module and debug source code
  auto build_module = BuildIRModule(ir_schedule);
  auto source_code  = GenSourceCode(build_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  // ApplyOnBlock
  // Apply AddCacheRead.
  const std::string& applied_block_name = default_output_names.back();
  EXPECT_EQ(add_cache_read.AnalyseApplyType(state, applied_block_name), RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states             = add_cache_read.ApplyOnBlock(state, applied_block_name);
  std::vector<ir::Expr> exprs = new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Matmul Expr after AddCacheRead applied on block: " << exprs[0];
  // build ir::Module and debug source code
  auto build_module_applied_on_block = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code_applied_on_block  = GenSourceCode(build_module_applied_on_block);
  VLOG(6) << "ApplyOnBlock scheduled source code:\n" << source_code_applied_on_block;
  EXPECT_EQ(source_code_applied_on_block, source_code);
  // execute and check precision
  CheckResult(GenExecutableKernel(build_module_applied_on_block),
              GenExecutableKernel(BuildIRModule(
                  MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})(), /* apply_manual_schedule*/ true))),
              default_input_names,
              default_output_names,
              {{32, 32}, {32, 32}},
              {{32, 32}},
              target_);
}

TEST_F(TestAddCacheRead, ApplyOnMatmulWithTiling) {
  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule       = MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})());
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];
  // Apply MultiLevelTiling before AddCacheRead.
  MultiLevelTiling multi_level_tiling(target_);
  multi_level_tiling.Init(&ir_schedule);
  ASSERT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();
  VLOG(6) << "Expr after MultiLevelTiling: " << func_bodys[0];
  // Apply AddCacheRead.
  AddCacheRead add_cache_read(target_);
  add_cache_read.Init(&ir_schedule);
  ASSERT_EQ(add_cache_read.NumberApplicable(), 1);
  add_cache_read.ApplyRandomly();
  VLOG(6) << "Expr after AddCacheRead: " << func_bodys[0];
  // build ir::Module and debug source code
  auto build_module = BuildIRModule(ir_schedule);
  auto source_code  = GenSourceCode(build_module);
  VLOG(6) << "scheduled source code:\n" << source_code;
  // execute and check precision
  CheckResult(GenExecutableKernel(build_module),
              GenExecutableKernel(BuildIRModule(
                  MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})(), /* apply_manual_schedule*/ true))),
              {"A", "B"},
              {"C"},
              {{32, 32}, {32, 32}},
              {{32, 32}},
              target_);

  // ApplyOnBlock
  const std::string& applied_block_name = default_output_names.back();
  ir_schedule                           = MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})());
  SearchState state(ir_schedule, 0, {});
  // Apply MultiLevelTiling before AddCacheRead.
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, applied_block_name), RuleApplyType::kApplyAndPruneOtherRules);
  auto states_after_tiling    = multi_level_tiling.ApplyOnBlock(state, applied_block_name);
  std::vector<ir::Expr> exprs = states_after_tiling[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after MultiLevelTiling applied on block: " << exprs[0];
  // Apply AddCacheRead.
  EXPECT_EQ(add_cache_read.AnalyseApplyType(states_after_tiling[0], applied_block_name),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto states_after_cache_read = add_cache_read.ApplyOnBlock(states_after_tiling[0], applied_block_name);
  exprs                        = states_after_cache_read[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Matmul Expr after AddCacheRead applied on block: " << exprs[0];
  // build ir::Module and debug source code
  build_module = BuildIRModule(states_after_cache_read[0]->ir_schedule);
  source_code  = GenSourceCode(build_module);
  VLOG(6) << "ApplyOnBlock scheduled source code:\n" << source_code;
  // execute and check precision
  CheckResult(GenExecutableKernel(build_module),
              GenExecutableKernel(BuildIRModule(
                  MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})(), /* apply_manual_schedule*/ true))),
              default_input_names,
              default_output_names,
              {{32, 32}, {32, 32}},
              {{32, 32}},
              target_);
}

}  // namespace auto_schedule
}  // namespace cinn
