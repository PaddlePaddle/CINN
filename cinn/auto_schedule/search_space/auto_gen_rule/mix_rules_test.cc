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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_read.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/add_cache_write.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/auto_schedule/tests/test_op_builder.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class TestMixRules : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names  = {"X", "Y"};
  std::vector<std::string> default_output_names = {"temp_matmul_out"};
};

TEST_F(TestMixRules, 2DMatmulOnMultiTilingRelated) {
  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule       = MakeIRSchedule(MatmulOpBuilder({32, 32}, {32, 32})());
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(target_);
  multi_level_tiling.Init(&ir_schedule);
  ASSERT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();
  VLOG(6) << "after MultiLevelTiling Expr:\n" << func_bodys[0];

  // Apply AddCacheWrite
  AddCacheWrite add_cache_write(target_);
  add_cache_write.Init(&ir_schedule);
  ASSERT_EQ(add_cache_write.NumberApplicable(), 1);
  add_cache_write.ApplyRandomly();
  VLOG(6) << "after AddCacheWrite Expr:\n" << func_bodys[0];

  // Apply AddCacheRead.
  AddCacheRead add_cache_read(target_);
  add_cache_read.Init(&ir_schedule);
  ASSERT_EQ(add_cache_read.NumberApplicable(), 1);
  add_cache_read.ApplyRandomly();
  VLOG(6) << "after AddCacheRead Expr:\n" << func_bodys[0];

  // build ir::Module and debug source code
  auto ir_module   = BuildIRModule(ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;
  // execute and check precision
  CheckResult(GenExecutableKernel(ir_module),
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
