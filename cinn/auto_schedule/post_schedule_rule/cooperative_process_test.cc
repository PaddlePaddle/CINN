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

#include "cinn/auto_schedule/post_schedule_rule/cooperative_process.h"

#include <gtest/gtest.h>

#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/ir/ir_printer.h"
#include "tests/program_builder.h"

namespace cinn {
namespace auto_schedule {

class TestCooperativeProcess : public TestAutoGenRuleBase {
 public:
  int fixed_rand_seed = 1;
  std::vector<std::string> default_input_names;
  std::vector<std::string> default_output_names;
};

TEST_F(TestCooperativeProcess, Matmul) {
  default_input_names            = {"X", "Y"};
  default_output_names           = {"temp_matmul_out"};
  std::vector<int32_t> X_shape   = {32, 32};
  std::vector<int32_t> Y_shape   = {32, 32};
  std::vector<int32_t> out_shape = {32, 32};

  int num_blocks_y  = 2;
  int num_blocks_x  = 2;
  int num_threads_y = 8;
  int num_threads_x = 2;
  int steps_k       = 8;

  Initialize(common::DefaultNVGPUTarget());
  frontend::Program matmul_op = tests::OpBuilder("matmul").Build({{"X", X_shape}, {"Y", Y_shape}});
  ir::IRSchedule ir_schedule  = MakeIRSchedule(matmul_op, fixed_rand_seed);

  // split loops
  std::vector<ir::Expr> loops   = ir_schedule.GetLoops("temp_matmul_out");
  std::vector<ir::Expr> k_loops = ir_schedule.Split(loops[2], {steps_k, -1});
  std::vector<ir::Expr> j_loops = ir_schedule.Split(loops[1], {num_blocks_x, num_threads_x, -1});
  std::vector<ir::Expr> i_loops = ir_schedule.Split(loops[0], {num_blocks_y, num_threads_y, -1});
  // reorder to "SSRRS": i0, j0, i1, j1, k0, k1, j2, i2
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.Reorder({loops[0], loops[3], loops[1], loops[4], loops[6], loops[7], loops[2], loops[5]});
  // fuse and bind
  loops                = ir_schedule.GetLoops("temp_matmul_out");
  ir::Expr i1_j1_fused = ir_schedule.Fuse({loops[2], loops[3]});
  ir::Expr i0_j0_fused = ir_schedule.Fuse({loops[0], loops[1]});
  loops                = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.Bind(loops[1], "threadIdx.x");
  ir_schedule.Bind(loops[0], "blockIdx.x");
  // cache read
  ir::Expr out_block     = ir_schedule.GetBlock("temp_matmul_out");
  ir::Expr X_cache_block = ir_schedule.CacheRead(out_block, 1, "shared");
  std::string X_cache_block_name =
      X_cache_block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.ComputeAt(X_cache_block, loops[2]);
  std::vector<ir::Expr> X_cache_loops = ir_schedule.GetLoops(X_cache_block_name);
  ir_schedule.Fuse({X_cache_loops[3], X_cache_loops[4]});
  ir_schedule.Annotate(ir_schedule.GetBlock(X_cache_block_name), ir::attr::cooperative_process, 0);

  out_block              = ir_schedule.GetBlock("temp_matmul_out");
  ir::Expr Y_cache_block = ir_schedule.CacheRead(out_block, 2, "shared");
  std::string Y_cache_block_name =
      Y_cache_block.As<ir::ScheduleBlockRealize>()->schedule_block.As<ir::ScheduleBlock>()->name;
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.ComputeAt(Y_cache_block, loops[2]);
  std::vector<ir::Expr> Y_cache_loops = ir_schedule.GetLoops(Y_cache_block_name);
  ir_schedule.Fuse({Y_cache_loops[3], Y_cache_loops[4]});
  ir_schedule.Annotate(ir_schedule.GetBlock(Y_cache_block_name), ir::attr::cooperative_process, 0);

  // apply CooperativeProcess
  CooperativeProcess cooperative_process;
  cooperative_process.Apply(&ir_schedule);

  auto ast = ir_schedule.GetModule().GetExprs();

  // build ir::Module and debug source code
  auto ir_module   = BuildIRModule(ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  // execute and check precision
  CheckResult(
      GenExecutableKernel(ir_module),
      GenExecutableKernel(BuildIRModule(MakeIRSchedule(matmul_op, fixed_rand_seed, /* apply_manual_schedule*/ true))),
      default_input_names,
      default_output_names,
      {X_shape, Y_shape},
      {out_shape},
      target_);
}

}  // namespace auto_schedule
}  // namespace cinn
