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

#include "cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "cinn/cinn.h"
#include "cinn/frontend/syntax.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "cinn/poly/stage.h"
#include "cinn/utils/string.h"
#include "tests/program_builder.h"

namespace cinn {
namespace auto_schedule {

TEST(MultiLevelTile, SampleSplitTwo) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  MultiLevelTiling multi_level_tiling(target, MultiLevelTiling::kConfigs.at(target.arch));

  for (int i = 0; i < 100; ++i) {
    size_t number_to_split    = rand() % 65535 + 2;  // random number in [2, 2^16]
    std::vector<size_t> split = multi_level_tiling.SampleSplitTwo<size_t>(number_to_split);
    EXPECT_EQ(split.size(), 2UL);
    EXPECT_EQ(split[0] * split[1], number_to_split);
  }
}

TEST(MultiLevelTile, SampleTileSplit) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  MultiLevelTiling multi_level_tiling(target, MultiLevelTiling::kConfigs.at(target.arch));

  for (int i = 0; i < 100; ++i) {
    int number_to_split    = rand() % 65535 + 2;  // random number in [2, 2^16]
    int split_size         = rand() % 5 + 1;      // random in [1, 5]
    std::vector<int> split = multi_level_tiling.SampleTileSplit<int>(number_to_split, split_size);
    EXPECT_EQ(split.size(), static_cast<size_t>(split_size));
    int product = 1;
    for (int num : split) {
      product *= num;
    }
    EXPECT_EQ(product, number_to_split);
  }
}

TEST(MultiLevelTile, SimpleLoops) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(128);

  Placeholder<float> A("A", {M});
  Placeholder<float> B("B", {N});

  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i) + B(j); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMultiLevelTile_SimpleLoops", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  MultiLevelTiling multi_level_tiling(target, MultiLevelTiling::kConfigs.at(target.arch));
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, "C"), RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, "C");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    std::string expr_str = ss.str();
    VLOG(6) << expr_str;
  };

  test_func(&ir_schedule);
  test_func(&new_states[0]->ir_schedule);
}

TEST(MulitLevelTile, MatrixMultiply) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(32);
  Expr K(32);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMultiLevelTile_MatrixMultiply", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  MultiLevelTiling multi_level_tiling(target, MultiLevelTiling::kConfigs.at(target.arch));
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, "C"), RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, "C");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    std::string expr_str = ss.str();
    VLOG(6) << expr_str;
  };

  test_func(&ir_schedule);
  test_func(&new_states[0]->ir_schedule);
}

class TestMultiLevelTiling : public TestAutoGenRuleBase {
 public:
  int fixed_rand_seed = 1;
  std::vector<std::string> default_input_names;
  std::vector<std::string> default_output_names;
};

TEST_F(TestMultiLevelTiling, Matmul) {
  default_input_names            = {"X", "Y"};
  default_output_names           = {"temp_matmul_out"};
  std::vector<int32_t> X_shape   = {32, 32};
  std::vector<int32_t> Y_shape   = {32, 32};
  std::vector<int32_t> out_shape = {32, 32};

  Initialize(common::DefaultNVGPUTarget());
  frontend::Program matmul_op = tests::OpBuilder("matmul").Build({{"X", X_shape}, {"Y", Y_shape}});
  ir::IRSchedule ir_schedule  = MakeIRSchedule(matmul_op, fixed_rand_seed);
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(target_, MultiLevelTiling::kConfigs.at(target_.arch));
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, default_output_names[0]);
  VLOG(6) << "After MultiLevelTiling, state:\n" << new_states[0]->DebugString();
  std::vector<float> target_feature = {1, 15,      15,      0,       0,       0, 0,       0, 0, 0,       0, 0, 0, 0,
                                       0, 0,       0,       16.6724, 15.2854, 0, 0,       0, 0, 0,       0, 0, 0, 0,
                                       0, 4.39232, 1.58496, 0,       0,       0, 2.32193, 0, 0, 2.32193, 0, 0, 0, 0};
  CheckFeature(new_states[0]->ir_schedule, target_feature);

  // build ir::Module and debug source code
  auto ir_module   = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  std::string target_source_code = R"ROC(#include <cstdint>

#define CINN_WITH_CUDA
#include "float16.h"
using cinn::common::float16;

#include "cinn_cuda_runtime_source.cuh"
__global__
void __launch_bounds__(1) fn_matmul_0(const float* __restrict__ X, const float* __restrict__ Y, float* __restrict__ temp_matmul_out)
{
  __shared__ float _X_reshape_shared_temp_buffer [ 64 ];
  __shared__ float _Y_reshape_shared_temp_buffer [ 256 ];
  float _temp_matmul_out_local_temp_buffer [ 256 ];
  float* X_reshape_shared_temp_buffer = _X_reshape_shared_temp_buffer;
  float* Y_reshape_shared_temp_buffer = _Y_reshape_shared_temp_buffer;
  float* temp_matmul_out__reduce_init = _temp_matmul_out_local_temp_buffer;
  float* temp_matmul_out_local_temp_buffer = _temp_matmul_out_local_temp_buffer;
  const float* X_reshape = X;
  const float* Y_reshape = Y;
  if (((int)blockIdx.x < 4)) {
    if (((int)threadIdx.x < 1)) {
      for (int32_t i_2 = 0; i_2 < 1; i_2 += 1) {
        for (int32_t j_2 = 0; j_2 < 1; j_2 += 1) {
          for (int32_t i_3 = 0; i_3 < 1; i_3 += 1) {
            for (int32_t j_3 = 0; j_3 < 1; j_3 += 1) {
              for (int32_t i_4 = 0; i_4 < 8; i_4 += 1) {
                for (int32_t j_4 = 0; j_4 < 32; j_4 += 1) {
                  temp_matmul_out__reduce_init[((256 * i_3) + ((32 * i_4) + ((32 * j_3) + j_4)))] = 0.00000000f;
                };
              };
            };
          };
          for (int32_t reduce_k_0 = 0; reduce_k_0 < 4; reduce_k_0 += 1) {
            for (int32_t ax0_0 = 0; ax0_0 < 8; ax0_0 += 1) {
              for (int32_t ax1_0 = 0; ax1_0 < 32; ax1_0 += 1) {
                Y_reshape_shared_temp_buffer[((32 * ax0_0) + ax1_0)] = Y_reshape[((32 * ax0_0) + ((32 * j_2) + ((256 * reduce_k_0) + ax1_0)))];
              };
            };
            for (int32_t ax0 = 0; ax0 < 8; ax0 += 1) {
              for (int32_t ax1 = 0; ax1 < 8; ax1 += 1) {
                X_reshape_shared_temp_buffer[((8 * ax0) + ((64 * (int)threadIdx.x) + ax1))] = X_reshape[((32 * ax0) + ((256 * (int)blockIdx.x) + ((256 * i_2) + ((8 * reduce_k_0) + ((256 * (int)threadIdx.x) + ax1)))))];
              };
            };
            for (int32_t reduce_k_1 = 0; reduce_k_1 < 1; reduce_k_1 += 1) {
              for (int32_t i_3 = 0; i_3 < 1; i_3 += 1) {
                for (int32_t j_3 = 0; j_3 < 1; j_3 += 1) {
                  for (int32_t reduce_k_2 = 0; reduce_k_2 < 8; reduce_k_2 += 1) {
                    for (int32_t i_4 = 0; i_4 < 8; i_4 += 1) {
                      for (int32_t j_4 = 0; j_4 < 32; j_4 += 1) {
                        temp_matmul_out_local_temp_buffer[((256 * i_3) + ((32 * i_4) + ((32 * j_3) + j_4)))] = (temp_matmul_out_local_temp_buffer[((256 * i_3) + ((32 * i_4) + ((32 * j_3) + j_4)))] + (X_reshape_shared_temp_buffer[((64 * i_3) + ((8 * i_4) + ((8 * r     educe_k_1) + ((64 * (int)threadIdx.x) + reduce_k_2))))] * Y_reshape_shared_temp_buffer[((32 * j_3) + ((256 * reduce_k_1) + ((32 * reduce_k_2) + j_4)))]));
                      };
                    };
                  };
                };
              };
            };
          };
          for (int32_t ax0_1 = 0; ax0_1 < 8; ax0_1 += 1) {
            for (int32_t ax1_1 = 0; ax1_1 < 32; ax1_1 += 1) {
              temp_matmul_out[((32 * ax0_1) + ((256 * (int)blockIdx.x) + ((256 * i_2) + ((32 * j_2) + ((256 * (int)threadIdx.x) + ax1_1)))))] = temp_matmul_out_local_temp_buffer[((32 * ax0_1) + ax1_1)];
            };
          };
        };
      };
    };
  };
})ROC";
  CHECK_EQ(source_code, target_source_code);

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

TEST_F(TestMultiLevelTiling, ReduceSum) {
  default_input_names             = {"X"};
  default_output_names            = {"var_0_tmp"};
  std::vector<int32_t> X_shape    = {1, 16, 32};
  std::vector<int32_t> out_shape  = {1, 16, 1};
  std::vector<int32_t> reduce_dim = {2};

  Initialize(common::DefaultNVGPUTarget());
  frontend::Program reduce_sum_op =
      tests::OpBuilder("reduce_sum").Build({{"X", X_shape}}, {{"dim", reduce_dim}, {"keep_dim", false}});
  ir::IRSchedule ir_schedule = MakeIRSchedule(reduce_sum_op);
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(target_, MultiLevelTiling::kConfigs.at(target_.arch));
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]), RuleApplyType::kCannotApply);
}

TEST_F(TestMultiLevelTiling, Pool2d) {
  default_input_names  = {"input"};
  default_output_names = {"var_0"};
  std::vector<int32_t> input_shape{2, 8, 16, 16};
  std::vector<int32_t> output_shape{2, 8, 8, 8};
  std::string pooling_type = "max";
  std::vector<int> ksize{3, 3};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{1, 1, 1, 1};
  bool ceil_mode                   = false;
  bool exclusive                   = true;
  bool global_pooling              = false;
  std::string data_format          = "NCHW";
  bool adaptive                    = false;
  std::string padding_algorithm    = "EXPLICIT";
  frontend::Program pool2d_program = tests::OpBuilder("pool2d").Build({{"input", input_shape}},
                                                                      {{"pool_type", pooling_type},
                                                                       {"kernel_size", ksize},
                                                                       {"stride_size", strides},
                                                                       {"padding_size", paddings},
                                                                       {"ceil_mode", ceil_mode},
                                                                       {"exclusive", exclusive},
                                                                       {"global_pooling", global_pooling},
                                                                       {"data_format", data_format},
                                                                       {"adaptive", adaptive},
                                                                       {"padding_algorithm", padding_algorithm}});

  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(pool2d_program, fixed_rand_seed);
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling::Config mlt_config = {
      /*bind_axis*/ std::vector<std::string>{"blockIdx.x", "threadIdx.x"},
      /*tile_struct*/ std::string("SSRS"),
      /*read_cache_memory_type*/ std::string("shared"),
      /*read_cache_levels*/ std::vector<int>{3},
      /*write_cache_memory_type*/ std::string("local"),
      /*write_cache_levels*/ std::vector<int>{2},
  };
  MultiLevelTiling multi_level_tiling(target_, mlt_config);
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, default_output_names[0]);
  VLOG(6) << "After MultiLevelTiling, state:\n" << new_states[0]->DebugString();
  std::vector<float> target_feature = {1, 0,       0,       0,       0,       0, 0,       14.8138, 14.17,   0, 14.3399,
                                       0, 0,       13.9249, 13.8139, 0,       0, 15.3152, 14.9916, 0,       0, 0,
                                       0, 0,       0,       0,       0,       0, 0,       4.32193, 1.58496, 0, 0,
                                       0, 4.08746, 0,       0,       6.02237, 0, 0,       0,       0};
  CheckFeature(new_states[0]->ir_schedule, target_feature);

  // build ir::Module and debug source code
  auto ir_module   = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  std::string target_source_code = R"ROC(#include <cstdint>

#define CINN_WITH_CUDA
#include "float16.h"
using cinn::common::float16;

#include "cinn_cuda_runtime_source.cuh"
__global__
void fn_pool2d_0(const float* __restrict__ input, float* __restrict__ pad_temp_0)
{
  for (int32_t i = 0; i < 2; i += 1) {
    for (int32_t j = 0; j < 8; j += 1) {
      for (int32_t k = 0; k < 18; k += 1) {
        for (int32_t a = 0; a < 18; a += 1) {
          pad_temp_0[((2592 * i) + ((324 * j) + ((18 * k) + a)))] = ((((a < 17) && ((a >= 1) && ((k < 17) && (k >= 1))))) ? input[(-17 + ((2048 * i) + ((256 * j) + ((16 * k) + a))))] : -3.40282347e+38f);
        };
      };
    };
  };
}__global__
void __launch_bounds__(4) fn_pool2d_0_1(const float* __restrict__ input, const float* __restrict__ pad_temp_0, float* __restrict__ var_0)
{
  __shared__ float _pad_temp_0_shared_temp_buffer [ 256 ];
  float _var_0_local_temp_buffer [ 16 ];
  float* pad_temp_0_shared_temp_buffer = _pad_temp_0_shared_temp_buffer;
  float* var_0__reduce_init = _var_0_local_temp_buffer;
  float* var_0_local_temp_buffer = _var_0_local_temp_buffer;
  if (((int)blockIdx.x < 16)) {
    if (((int)threadIdx.x < 4)) {
    {
      for (int32_t i_2 = 0; i_2 < 1; i_2 += 1) {
        for (int32_t j_2 = 0; j_2 < 4; j_2 += 1) {
          for (int32_t k_2 = 0; k_2 < 1; k_2 += 1) {
            for (int32_t a_2 = 0; a_2 < 4; a_2 += 1) {
              var_0__reduce_init[((16 * i_2) + ((4 * j_2) + ((4 * k_2) + a_2)))] = -3.40282347e+38f;
            };
          };
        };
      };
      for (int32_t kernel_idx = 0; kernel_idx < 3; kernel_idx += 1) {
        for (int32_t kernel_idx_0 = 0; kernel_idx_0 < 3; kernel_idx_0 += 1) {
          for (int32_t ax0 = 0; ax0 < 4; ax0 += 1) {
            for (int32_t ax1 = 0; ax1 < 7; ax1 += 1) {
              pad_temp_0_shared_temp_buffer[((64 * ax0) + ((16 * (int)threadIdx.x) + ax1))] = pad_temp_0[((((((int)blockIdx.x / 2) / 2) / 2) * 2592) + ((1296 * ((((int)blockIdx.x / 2) / 2) & 1)) + ((144 * (((int)blockIdx.x / 2) & 1)) + ((8 * ((int)blockIdx.x & 1)) + ((324 * ax0) + ((18 * kernel_idx) + ((36 * (int)threadIdx.x) + (ax1 + kernel_idx_0))))))))];
            };
          };
          for (int32_t i_2 = 0; i_2 < 1; i_2 += 1) {
            for (int32_t j_2 = 0; j_2 < 4; j_2 += 1) {
              for (int32_t k_2 = 0; k_2 < 1; k_2 += 1) {
                for (int32_t a_2 = 0; a_2 < 4; a_2 += 1) {
                  var_0_local_temp_buffer[((16 * i_2) + ((4 * j_2) + ((4 * k_2) + a_2)))] = max(var_0_local_temp_buffer[((16 * i_2) + ((4 * j_2) + ((4 * k_2) + a_2)))], pad_temp_0_shared_temp_buffer[((2 * a_2) + ((256 * i_2) + ((64 * j_2) + ((16 * k_2) + (16 * (int)threadIdx.x)))))]);
                };
              };
            };
          };
        };
      };
      for (int32_t ax0_0 = 0; ax0_0 < 4; ax0_0 += 1) {
        for (int32_t ax1_0 = 0; ax1_0 < 4; ax1_0 += 1) {
          var_0[((((((int)blockIdx.x / 2) / 2) / 2) * 512) + ((256 * ((((int)blockIdx.x / 2) / 2) & 1)) + ((32 * (((int)blockIdx.x / 2) & 1)) + ((4 * ((int)blockIdx.x & 1)) + ((64 * ax0_0) + ((8 * (int)threadIdx.x) + ax1_0))))))] = var_0_local_temp_buffer[((4 * ax0_0) + ax1_0)];
        };
      };
    }
    };
  };
})ROC";
  CHECK_EQ(source_code, target_source_code);

  // execute and check precision
  CheckResult(GenExecutableKernel(ir_module),
              GenExecutableKernel(
                  BuildIRModule(MakeIRSchedule(pool2d_program, fixed_rand_seed, /* apply_manual_schedule*/ true))),
              default_input_names,
              default_output_names,
              {input_shape},
              {output_shape},
              target_);
}

}  // namespace auto_schedule
}  // namespace cinn
